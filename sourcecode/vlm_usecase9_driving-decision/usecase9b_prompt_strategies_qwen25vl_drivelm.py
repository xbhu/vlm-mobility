#!/usr/bin/env python3
"""
UC9b: Prompt Strategy Comparison for Direct Decision
3 strategies on the same 9 frames (CAM_FRONT only — controlled variable):
  S1 Imperative   : strict format constraint, minimal context
  S2 Role-based   : AV agent framing + urgency
  S3 Chain-of-Thought : explicit scene reasoning before final decision
Key output: parsability + accuracy per strategy; full CoT reasoning chains saved.
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
OUTPUT_PATH = "/home/xzh5180/Research/vlm-mobility/outputs/usecase9b_prompt_strategies_qwen25vl_drivelm.json"

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

# ==================== PROMPT STRATEGIES ====================
STRATEGIES = {
    "S1_imperative": {
        "name": "Imperative Classification",
        "hypothesis": "Strict format forces compliance; minimal context = no distraction",
        "prompt": (
            "Classify the driving behavior visible in this front camera image.\n\n"
            "Speed (choose one): fast | medium | slow | stop\n"
            "Direction (choose one): straight | turn_left | turn_right | "
            "lane_change_left | lane_change_right\n\n"
            "Output format only:\n"
            "Speed: [choice]\n"
            "Direction: [choice]"
        ),
        "max_new_tokens": 30
    },
    "S2_role_based": {
        "name": "Role-Based Agent",
        "hypothesis": "Agent identity + urgency framing improves decision compliance",
        "prompt": (
            "You are an autonomous vehicle making a real-time driving decision. "
            "Based solely on what you observe in this front camera image, "
            "output your decision immediately.\n\n"
            "Speed: [fast/medium/slow/stop]\n"
            "Direction: [straight/turn_left/turn_right/lane_change_left/lane_change_right]"
        ),
        "max_new_tokens": 40
    },
    "S3_chain_of_thought": {
        "name": "Chain-of-Thought",
        "hypothesis": "Explicit reasoning chain makes decision basis visible; tests reasoning-decision consistency",
        "prompt": (
            "You are an autonomous vehicle decision system. "
            "Analyze this front camera image step by step, then make a driving decision.\n\n"
            "Step 1 - Scene: What do you observe? (road, traffic, obstacles, lane markings)\n"
            "Step 2 - Situation: What is the current driving context?\n"
            "Step 3 - Rationale: Why does this situation require a specific behavior?\n\n"
            "Final Decision:\n"
            "Speed: [fast/medium/slow/stop]\n"
            "Direction: [straight/turn_left/turn_right/lane_change_left/lane_change_right]"
        ),
        "max_new_tokens": 220
    }
}

# ==================== PARSERS ====================
def parse_text_to_label(text):
    t = text.lower()

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
    t = text.lower()
    speed, direction = None, None

    # Prefer structured "Speed: X" / "Direction: Y"
    m = re.search(r"speed:\s*([\w_]+)", t)
    if m and m.group(1).strip() in SPEED_LABELS:
        speed = m.group(1).strip()

    m = re.search(r"direction:\s*([\w_]+)", t)
    if m and m.group(1).strip() in DIR_LABELS:
        direction = m.group(1).strip()

    # Keyword fallback
    if speed is None or direction is None:
        spd_fb, dir_fb = parse_text_to_label(text)
        if speed is None:
            speed = spd_fb
        if direction is None:
            direction = dir_fb

    return speed, direction


def extract_final_decision_from_cot(text):
    """For CoT outputs: extract only the final decision block."""
    t = text.lower()
    # Look for "final decision:" header
    idx = t.find("final decision:")
    if idx != -1:
        decision_section = text[idx:]
        return parse_model_output(decision_section)
    # Fallback: parse full text
    return parse_model_output(text)

# ==================== GT EXTRACTION ====================
def get_gt(frame_data):
    pairs = []
    for item in frame_data.get("QA", {}).get("behavior", []):
        if isinstance(item, dict):
            q = item.get("Q", item.get("question", ""))
            a = item.get("A", item.get("answer", ""))
            pairs.append({"Q": q, "A": a})
    combined = " ".join(p["A"] for p in pairs if p["A"])
    speed, direction = parse_text_to_label(combined)
    return speed, direction, combined


def resolve_image(raw_path):
    fixed = raw_path.replace("../nuscenes/", "nuscenes/")
    return os.path.join(IMAGE_ROOT, fixed)

# ==================== MAIN ====================
def main():
    print("=" * 65)
    print("UC9b: Prompt Strategy Comparison")
    print("=" * 65)

    with open(DATA_PATH) as f:
        data = json.load(f)

    # Pre-collect frames + GT
    frames = []
    for scene_id, frame_tokens in EVAL_FRAMES.items():
        kf = data.get(scene_id, {}).get("key_frames", {})
        for ft in frame_tokens:
            fd = kf.get(ft, {})
            if not fd:
                continue
            gt_spd, gt_dir, gt_text = get_gt(fd)
            img_path = resolve_image(fd.get("image_paths", {}).get("CAM_FRONT", ""))
            frames.append({
                "scene_id": scene_id,
                "frame_token": ft,
                "gt_speed": gt_spd,
                "gt_direction": gt_dir,
                "gt_text": gt_text,
                "image_path": img_path
            })

    print(f"Frames: {len(frames)} | Strategies: {len(STRATEGIES)}")
    print(f"Total inference calls: {len(frames) * len(STRATEGIES)}\n")

    print("Loading model...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    print("Model ready.\n")

    all_results = {}

    for sk, scfg in STRATEGIES.items():
        print("=" * 65)
        print(f"Strategy: {scfg['name']} ({sk})")
        print(f"Hypothesis: {scfg['hypothesis']}")
        print(f"Max tokens: {scfg['max_new_tokens']}")
        print("=" * 65)

        strategy_results = []

        for frame in frames:
            img_ok = os.path.exists(frame["image_path"])
            print(f"\n  [{frame['frame_token'][:12]}...] "
                  f"GT: spd={frame['gt_speed']} dir={frame['gt_direction']}")

            content = []
            if img_ok:
                content.append({"type": "image", "image": frame["image_path"]})
            content.append({"type": "text", "text": scfg["prompt"]})
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
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=scfg["max_new_tokens"],
                    do_sample=False
                )

            prompt_len = inputs["input_ids"].shape[1]
            response = processor.decode(
                out_ids[0][prompt_len:], skip_special_tokens=True
            ).strip()

            # Use CoT-aware parser for S3
            if sk == "S3_chain_of_thought":
                pred_spd, pred_dir = extract_final_decision_from_cot(response)
            else:
                pred_spd, pred_dir = parse_model_output(response)

            parsable = (pred_spd is not None) and (pred_dir is not None)
            spd_ok = (pred_spd == frame["gt_speed"])
            dir_ok = (pred_dir == frame["gt_direction"])

            # Print truncated output for non-CoT, full for CoT
            display_len = 300 if sk == "S3_chain_of_thought" else 100
            print(f"  Output: {response[:display_len]}")
            print(f"  Pred: spd={pred_spd}, dir={pred_dir} | "
                  f"Match: spd={spd_ok}, dir={dir_ok}, parsable={parsable}")

            # CoT consistency check: does reasoning text align with decision?
            cot_consistency = None
            if sk == "S3_chain_of_thought":
                reasoning_spd, reasoning_dir = parse_text_to_label(response)
                # Consistent if reasoning keywords agree with final decision
                cot_consistency = {
                    "reasoning_implies_speed": reasoning_spd,
                    "reasoning_implies_dir": reasoning_dir,
                    "speed_consistent": (reasoning_spd == pred_spd) if (reasoning_spd and pred_spd) else None,
                    "dir_consistent": (reasoning_dir == pred_dir) if (reasoning_dir and pred_dir) else None
                }

            strategy_results.append({
                "scene_id": frame["scene_id"],
                "frame_token": frame["frame_token"],
                "gt_speed": frame["gt_speed"],
                "gt_direction": frame["gt_direction"],
                "model_output": response,
                "pred_speed": pred_spd,
                "pred_direction": pred_dir,
                "parsable": parsable,
                "speed_correct": spd_ok,
                "direction_correct": dir_ok,
                "both_correct": spd_ok and dir_ok,
                "cot_consistency": cot_consistency
            })

        # Strategy metrics
        total = len(strategy_results)
        par = [r for r in strategy_results if r["parsable"]]
        n_p = len(par)

        metrics = {
            "parsability_rate": round(n_p / total, 4) if total else 0,
            "speed_accuracy": round(sum(r["speed_correct"] for r in par) / n_p, 4) if n_p else 0,
            "direction_accuracy": round(sum(r["direction_correct"] for r in par) / n_p, 4) if n_p else 0,
            "combined_accuracy": round(sum(r["both_correct"] for r in par) / n_p, 4) if n_p else 0,
            "n_parsable": n_p,
            "n_total": total
        }

        # CoT: consistency stats
        if sk == "S3_chain_of_thought":
            cot_list = [r["cot_consistency"] for r in strategy_results
                        if r["cot_consistency"] is not None]
            spd_consistent = sum(1 for c in cot_list if c["speed_consistent"] is True)
            dir_consistent = sum(1 for c in cot_list if c["dir_consistent"] is True)
            metrics["cot_speed_consistency"] = round(spd_consistent / len(cot_list), 4) if cot_list else 0
            metrics["cot_dir_consistency"] = round(dir_consistent / len(cot_list), 4) if cot_list else 0

        print(f"\n  >> parsability={metrics['parsability_rate']:.3f}  "
              f"speed_acc={metrics['speed_accuracy']:.3f}  "
              f"dir_acc={metrics['direction_accuracy']:.3f}  "
              f"combined={metrics['combined_accuracy']:.3f}")
        if "cot_speed_consistency" in metrics:
            print(f"     CoT reasoning-decision consistency: "
                  f"speed={metrics['cot_speed_consistency']:.3f}  "
                  f"dir={metrics['cot_dir_consistency']:.3f}")

        all_results[sk] = {
            "config": scfg,
            "metrics": metrics,
            "results": strategy_results
        }

    # ---- Cross-strategy comparison table ----
    print("\n\n" + "=" * 65)
    print("CROSS-STRATEGY COMPARISON")
    print("=" * 65)
    hdr = f"{'Strategy':<28} {'Parsable':>9} {'Spd Acc':>9} {'Dir Acc':>9} {'Combined':>9}"
    print(hdr)
    print("-" * 65)
    for sk, sv in all_results.items():
        m = sv["metrics"]
        name = sv["config"]["name"][:27]
        print(f"{name:<28} {m['parsability_rate']:>9.3f} "
              f"{m['speed_accuracy']:>9.3f} {m['direction_accuracy']:>9.3f} "
              f"{m['combined_accuracy']:>9.3f}")

    # ---- CoT reasoning chain display ----
    print("\n\n" + "=" * 65)
    print("CoT REASONING-DECISION CONSISTENCY (S3)")
    print("Tests: does the reasoning chain actually support the decision?")
    print("=" * 65)
    if "S3_chain_of_thought" in all_results:
        for r in all_results["S3_chain_of_thought"]["results"]:
            print(f"\nFrame: {r['frame_token'][:16]}...")
            print(f"GT    : speed={r['gt_speed']}, dir={r['gt_direction']}")
            print(f"Pred  : speed={r['pred_speed']}, dir={r['pred_direction']}  "
                  f"(match={r['both_correct']})")
            if r["cot_consistency"]:
                c = r["cot_consistency"]
                print(f"CoT implied: spd={c['reasoning_implies_speed']}, "
                      f"dir={c['reasoning_implies_dir']}  "
                      f"(spd_consistent={c['speed_consistent']}, "
                      f"dir_consistent={c['dir_consistent']})")
            print(f"Reasoning:\n{r['model_output'][:500]}")
            print("-" * 50)

    # Save
    output = {
        "experiment": "UC9b_prompt_strategy_comparison",
        "model": MODEL_ID,
        "camera": "CAM_FRONT",
        "strategies": list(STRATEGIES.keys()),
        "results_by_strategy": all_results
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
