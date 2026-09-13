#!/usr/bin/env python3
"""
UC9a: Direct Decision Baseline
Zero-shot image -> discrete driving decision (no question input)
Phase 1: Behavior QA exploration + GT label distribution
Phase 2: Zero-shot inference with imperative prompt
Phase 3: Parsability + accuracy metrics
"""

import json
import os
import re
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# ==================== CONFIG ====================
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DATA_PATH = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm/v1_1_train_nus.json"
IMAGE_ROOT = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm"
OUTPUT_PATH = "/home/xzh5180/Research/vlm-mobility/outputs/usecase9a_direct_decision_baseline_qwen25vl_drivelm.json"

EVAL_FRAMES = {
    "f0f120e4d4b0441da90ec53b16ee169d": [
        "4a0798f849ca477ab18009c3a20b7df2",
        "ffd1bdf020d145759224c629b501d2b2",
        "d9075c2a5f864a2b8abf41e703f4cf1c"
    ],
    "54cdaaae372d421fa4734d66f51a8c48": [
        "542eaf1fc9b34895a9e55fab57cb4cf4",
        "1b45a97a0e5e49fe9cd345dd4bd729c3",
        "d5e16062410f4e329d31a881b28e5c1c"
    ],
    "1977a1c98a6c4eb79fbc2a6dc0da9b0f": [
        "bd8a5e326b804b069d497d29dbf19c2b",
        "7903e67446c64958b0a660f10bdadf19",
        "b6bf5a2bcb094969ace1023f8fe0b9e2"
    ]
}

SPEED_LABELS = ["fast", "medium", "slow", "stop"]
DIR_LABELS = ["straight", "turn_left", "turn_right", "lane_change_left", "lane_change_right"]

# ==================== PROMPT ====================
DECISION_PROMPT = (
    "You are an autonomous vehicle decision system. "
    "Based on this front camera image, classify the vehicle's current driving behavior.\n\n"
    "Output in this exact format only:\n"
    "Speed: [fast/medium/slow/stop]\n"
    "Direction: [straight/turn_left/turn_right/lane_change_left/lane_change_right]"
)

# ==================== LABEL PARSING ====================
def parse_text_to_label(text):
    """Keyword-based parser for natural language behavior descriptions."""
    t = text.lower()

    # Speed: check stop before slow to avoid partial match errors
    speed = None
    if any(w in t for w in ["stop", "stopped", "stationary", "not moving",
                             "halted", "standing still", "standstill"]):
        speed = "stop"
    elif any(w in t for w in ["slow", "slowly", "decelerat", "low speed", "crawl"]):
        speed = "slow"
    elif any(w in t for w in ["fast", "quickly", "high speed", "accelerat", "speeding"]):
        speed = "fast"
    elif any(w in t for w in ["medium", "moderate", "normal speed",
                              "constant speed", "maintain"]):
        speed = "medium"

    # Direction: check lane-change before turn before straight
    direction = None
    if any(w in t for w in ["lane change left", "changing lane to the left",
                             "merge left", "moving to the left lane"]):
        direction = "lane_change_left"
    elif any(w in t for w in ["lane change right", "changing lane to the right",
                               "merge right", "moving to the right lane"]):
        direction = "lane_change_right"
    elif any(w in t for w in ["turn left", "turning left", "left turn"]):
        direction = "turn_left"
    elif any(w in t for w in ["turn right", "turning right", "right turn"]):
        direction = "turn_right"
    elif any(w in t for w in ["straight", "forward", "ahead", "going straight",
                               "moving straight", "continue straight"]):
        direction = "straight"

    return speed, direction


def parse_model_output(text):
    """Parse structured model output. Falls back to keyword scan."""
    t = text.lower()
    speed, direction = None, None

    m = re.search(r"speed:\s*([\w_]+)", t)
    if m and m.group(1).strip() in SPEED_LABELS:
        speed = m.group(1).strip()

    m = re.search(r"direction:\s*([\w_]+)", t)
    if m and m.group(1).strip() in DIR_LABELS:
        direction = m.group(1).strip()

    # Fallback keyword scan
    if speed is None or direction is None:
        spd_fb, dir_fb = parse_text_to_label(text)
        if speed is None:
            speed = spd_fb
        if direction is None:
            direction = dir_fb

    return speed, direction

# ==================== GT EXTRACTION ====================
def get_behavior_qa_pairs(frame_data):
    """Return list of {Q, A} from behavior QA field."""
    pairs = []
    for item in frame_data.get("QA", {}).get("behavior", []):
        if isinstance(item, dict):
            q = item.get("Q", item.get("question", ""))
            a = item.get("A", item.get("answer", ""))
            pairs.append({"Q": q, "A": a})
    return pairs


def extract_gt(frame_data):
    """Extract (speed, direction) GT labels from behavior QA answers."""
    pairs = get_behavior_qa_pairs(frame_data)
    combined = " ".join(p["A"] for p in pairs if p["A"])
    speed, direction = parse_text_to_label(combined)
    return speed, direction, combined, pairs


def resolve_image(raw_path):
    fixed = raw_path.replace("../nuscenes/", "nuscenes/")
    return os.path.join(IMAGE_ROOT, fixed)

# ==================== MAIN ====================
def main():
    print("=" * 60)
    print("UC9a: Direct Decision Baseline")
    print("=" * 60)

    with open(DATA_PATH) as f:
        data = json.load(f)

    # ---- Phase 1: Behavior QA Exploration ----
    print("\n[Phase 1] Behavior QA Exploration")
    print("-" * 60)
    print("IMPORTANT: Check GT text below to verify parser correctness")
    print("If speed or direction shows None, inspect raw QA text.\n")

    for scene_id, frame_tokens in EVAL_FRAMES.items():
        kf = data.get(scene_id, {}).get("key_frames", {})
        print(f"Scene: {scene_id[:16]}...")
        for ft in frame_tokens:
            fd = kf.get(ft, {})
            if not fd:
                print(f"  Frame {ft[:8]}: NOT FOUND")
                continue
            speed, direction, combined, pairs = extract_gt(fd)
            print(f"  Frame {ft[:8]}...")
            for p in pairs:
                print(f"    Q: {p['Q'][:90]}")
                print(f"    A: {p['A'][:90]}")
            print(f"    -> GT: speed={speed}, direction={direction}")
            if combined and speed is None and direction is None:
                print(f"    WARNING: Parser returned None for: {combined[:80]}")
        print()

    # ---- Phase 2: Load Model ----
    print("[Phase 2] Loading Qwen2.5-VL-7B (4-bit)...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    print("Model ready.\n")

    # ---- Phase 3: Inference ----
    print("[Phase 3] Zero-shot Direct Decision Inference")
    print("-" * 60)

    results = []

    for scene_id, frame_tokens in EVAL_FRAMES.items():
        kf = data.get(scene_id, {}).get("key_frames", {})
        for ft in frame_tokens:
            fd = kf.get(ft, {})
            if not fd:
                continue

            gt_speed, gt_dir, gt_text, gt_pairs = extract_gt(fd)
            cam_path = resolve_image(fd.get("image_paths", {}).get("CAM_FRONT", ""))
            img_exists = os.path.exists(cam_path)

            print(f"\n[{ft[:12]}...]")
            print(f"  GT: speed={gt_speed}, dir={gt_dir}")
            print(f"  Image: {'OK' if img_exists else 'MISSING - ' + cam_path}")

            # Build message
            content = []
            if img_exists:
                content.append({"type": "image", "image": cam_path})
            content.append({"type": "text", "text": DECISION_PROMPT})
            messages = [{"role": "user", "content": content}]

            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                out_ids = model.generate(**inputs, max_new_tokens=60, do_sample=False)

            prompt_len = inputs["input_ids"].shape[1]
            response = processor.decode(
                out_ids[0][prompt_len:], skip_special_tokens=True
            ).strip()

            pred_speed, pred_dir = parse_model_output(response)
            parsable = (pred_speed is not None) and (pred_dir is not None)
            spd_ok = (pred_speed == gt_speed)
            dir_ok = (pred_dir == gt_dir)

            print(f"  Output: {response}")
            print(f"  Pred: speed={pred_speed}, dir={pred_dir}")
            print(f"  Match: spd={spd_ok}, dir={dir_ok}, parsable={parsable}")

            results.append({
                "scene_id": scene_id,
                "frame_token": ft,
                "gt_speed": gt_speed,
                "gt_direction": gt_dir,
                "gt_text": gt_text,
                "gt_qa_pairs": gt_pairs,
                "image_path": cam_path,
                "model_output": response,
                "pred_speed": pred_speed,
                "pred_direction": pred_dir,
                "parsable": parsable,
                "speed_correct": spd_ok,
                "direction_correct": dir_ok,
                "both_correct": spd_ok and dir_ok
            })

    # ---- Phase 4: Metrics ----
    print("\n" + "=" * 60)
    print("[Phase 4] Metrics")
    print("=" * 60)

    total = len(results)
    par = [r for r in results if r["parsable"]]
    n_p = len(par)

    parsability = n_p / total if total else 0
    speed_acc = sum(r["speed_correct"] for r in par) / n_p if n_p else 0
    dir_acc = sum(r["direction_correct"] for r in par) / n_p if n_p else 0
    both_acc = sum(r["both_correct"] for r in par) / n_p if n_p else 0

    # GT distribution (parsable GT only)
    gt_spds = [r["gt_speed"] for r in results if r["gt_speed"]]
    gt_dirs = [r["gt_direction"] for r in results if r["gt_direction"]]
    gt_none_spd = sum(1 for r in results if r["gt_speed"] is None)
    gt_none_dir = sum(1 for r in results if r["gt_direction"] is None)

    metrics = {
        "total_frames": total,
        "parsable_count": n_p,
        "parsability_rate": round(parsability, 4),
        "speed_accuracy": round(speed_acc, 4),
        "direction_accuracy": round(dir_acc, 4),
        "combined_accuracy": round(both_acc, 4),
        "gt_speed_none_count": gt_none_spd,
        "gt_direction_none_count": gt_none_dir,
        "gt_speed_distribution": {s: gt_spds.count(s) for s in SPEED_LABELS},
        "gt_direction_distribution": {d: gt_dirs.count(d) for d in DIR_LABELS},
        "pred_speed_distribution": {
            s: sum(1 for r in results if r["pred_speed"] == s) for s in SPEED_LABELS
        },
        "pred_direction_distribution": {
            d: sum(1 for r in results if r["pred_direction"] == d) for d in DIR_LABELS
        }
    }

    print(f"Total frames       : {total}")
    print(f"Parsable           : {n_p}/{total} ({parsability:.3f})")
    print(f"Speed accuracy     : {speed_acc:.3f}")
    print(f"Direction accuracy : {dir_acc:.3f}")
    print(f"Combined accuracy  : {both_acc:.3f}")
    print(f"GT speed=None      : {gt_none_spd} frames")
    print(f"GT direction=None  : {gt_none_dir} frames")
    print(f"GT speed dist      : {metrics['gt_speed_distribution']}")
    print(f"GT dir dist        : {metrics['gt_direction_distribution']}")
    print(f"Pred speed dist    : {metrics['pred_speed_distribution']}")
    print(f"Pred dir dist      : {metrics['pred_direction_distribution']}")

    output = {
        "experiment": "UC9a_direct_decision_baseline",
        "model": MODEL_ID,
        "camera": "CAM_FRONT",
        "prompt": DECISION_PROMPT,
        "metrics": metrics,
        "results": results
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
