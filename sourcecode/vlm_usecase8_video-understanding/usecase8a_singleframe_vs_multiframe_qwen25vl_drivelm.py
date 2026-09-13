"""
UC8a: Single-frame vs Multi-frame Temporal Understanding
Compare VLM responses between single-frame and multi-frame (pseudo-video) input.
"""

import json
import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import BitsAndBytesConfig
from PIL import Image

# ─────────────────────────── CONFIG ───────────────────────────
MODEL_NAME   = "Qwen/Qwen2.5-VL-7B-Instruct"
IMAGE_ROOT   = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm"
DRIVELM_JSON = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm/v1_1_train_nus.json"
NUSCENES_DIR = "/home/xzh5180/Research/vlm-mobility/datasets/nuscenes/v1.0-trainval"
OUTPUT_PATH  = "/home/xzh5180/Research/vlm-mobility/outputs/usecase8a_singleframe_vs_multiframe_qwen25vl_drivelm.json"
CAMERA       = "CAM_FRONT"

# Fixed evaluation set: scene_id -> [frame_tokens in intended order]
EVAL_SCENES = {
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
    ],
}

# Questions
QUESTIONS = {
    "Q1_description": "Describe what you see in this driving scene.",
    "Q2_action":      (
        "Based on the image sequence provided, what is the ego vehicle doing? "
        "Describe its behavior over time using temporal language such as initially, then, and finally."
    ),
    "Q3_change":      (
        "What are the key differences between the first frame and the last frame of this sequence? "
        "Focus on changes in the scene, road conditions, or surrounding vehicles."
    ),
    "Q4_prediction":  (
        "Based on what you observe, what do you predict the ego vehicle will do next? "
        "Explain your reasoning."
    ),
}
# Q3 is multi-frame only (needs at least 2 frames to compare)
MULTIFRAME_ONLY = {"Q3_change"}
# ──────────────────────────────────────────────────────────────


def resolve_image_path(raw_path: str) -> str:
    """Replace ../nuscenes/ prefix with absolute IMAGE_ROOT/nuscenes/ path."""
    cleaned = raw_path.replace("../nuscenes/", "nuscenes/")
    return os.path.join(IMAGE_ROOT, cleaned)


def load_timestamps(nuscenes_dir: str) -> dict:
    """Load sample timestamps from nuScenes annotation for ordering frames."""
    sample_json = os.path.join(nuscenes_dir, "sample.json")
    with open(sample_json, "r") as f:
        samples = json.load(f)
    return {s["token"]: s["timestamp"] for s in samples}


def get_ordered_frames(scene_data: dict, requested_tokens: list, timestamps: dict) -> list:
    """
    Return frame_tokens in chronological order.
    Only includes tokens present in both scene key_frames and requested list.
    """
    available = set(scene_data["key_frames"].keys())
    valid_tokens = [t for t in requested_tokens if t in available]
    # Sort by nuScenes timestamp; fall back to requested order if token missing
    def ts(token):
        return timestamps.get(token, 0)
    return sorted(valid_tokens, key=ts)


def build_messages(image_paths: list, question: str) -> list:
    """
    Build Qwen2.5-VL message list.
    image_paths: list of absolute paths (1 for single-frame, N for multi-frame)
    """
    content = []
    for idx, img_path in enumerate(image_paths):
        if len(image_paths) > 1:
            content.append({"type": "text", "text": f"Frame {idx + 1}:"})
        content.append({"type": "image", "image": img_path})
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def run_inference(model, processor, messages: list) -> str:
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated = output_ids[0][prompt_len:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def count_temporal_words(text: str) -> dict:
    """Count temporal and dynamic vocabulary in a response."""
    text_lower = text.lower()
    temporal_connectors = ["first", "then", "next", "finally", "initially",
                           "subsequently", "before", "after", "as the sequence"]
    dynamic_descriptors = ["accelerating", "decelerating", "slowing", "turning",
                           "stopping", "moving", "braking", "speeding"]
    change_descriptors  = ["changes", "shifts", "increases", "decreases",
                           "appears to", "seems to", "transitions"]
    frame_references    = ["frame 1", "frame 2", "frame 3", "frame 4", "frame 5",
                           "first frame", "last frame", "second frame", "third frame"]

    def count(keywords):
        return sum(1 for kw in keywords if kw in text_lower)

    return {
        "temporal_connectors": count(temporal_connectors),
        "dynamic_descriptors": count(dynamic_descriptors),
        "change_descriptors":  count(change_descriptors),
        "frame_references":    count(frame_references),
        "total_words":         len(text.split()),
    }


def main():
    print("Loading nuScenes timestamps...")
    timestamps = load_timestamps(NUSCENES_DIR)

    print("Loading DriveLM dataset...")
    with open(DRIVELM_JSON, "r") as f:
        drivelm = json.load(f)

    print("Loading model (4-bit quantization)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    results = {}

    for scene_id, requested_tokens in EVAL_SCENES.items():
        print(f"\n{'='*60}")
        print(f"Scene: {scene_id}")
        scene_data = drivelm[scene_id]

        # Get chronologically ordered frames for this scene
        ordered_tokens = get_ordered_frames(scene_data, requested_tokens, timestamps)
        print(f"  Ordered frames ({len(ordered_tokens)}): {ordered_tokens}")

        # Resolve image paths
        all_image_paths = []
        for token in ordered_tokens:
            frame = scene_data["key_frames"][token]
            raw_path = frame["image_paths"][CAMERA]
            abs_path = resolve_image_path(raw_path)
            if not os.path.exists(abs_path):
                print(f"  WARNING: image not found: {abs_path}")
            all_image_paths.append(abs_path)

        single_image_paths = [all_image_paths[0]]  # first frame only

        scene_result = {
            "scene_description": scene_data.get("scene_description", ""),
            "frame_tokens_used": ordered_tokens,
            "num_frames": len(ordered_tokens),
            "single_frame": {},
            "multi_frame": {},
        }

        # ── Single-frame inference ──
        print(f"  [Single-frame] using: {ordered_tokens[0]}")
        for qkey, qtext in QUESTIONS.items():
            if qkey in MULTIFRAME_ONLY:
                print(f"    Skipping {qkey} (multi-frame only)")
                continue
            print(f"    {qkey} ...", end=" ", flush=True)
            messages = build_messages(single_image_paths, qtext)
            response = run_inference(model, processor, messages)
            lexical  = count_temporal_words(response)
            scene_result["single_frame"][qkey] = {
                "question": qtext,
                "response": response,
                "lexical_analysis": lexical,
            }
            print(f"done ({lexical['total_words']} words)")

        # ── Multi-frame inference ──
        print(f"  [Multi-frame] using {len(all_image_paths)} frames")
        for qkey, qtext in QUESTIONS.items():
            print(f"    {qkey} ...", end=" ", flush=True)
            messages = build_messages(all_image_paths, qtext)
            response = run_inference(model, processor, messages)
            lexical  = count_temporal_words(response)
            scene_result["multi_frame"][qkey] = {
                "question": qtext,
                "response": response,
                "lexical_analysis": lexical,
            }
            print(f"done ({lexical['total_words']} words)")

        results[scene_id] = scene_result

    # ── Save output ──
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {OUTPUT_PATH}")

    # ── Quick summary ──
    print("\n" + "="*60)
    print("QUICK SUMMARY: Temporal Lexical Rate (multi vs single)")
    print("="*60)
    for scene_id, sr in results.items():
        print(f"\nScene {scene_id[:8]}...")
        for qkey in QUESTIONS:
            if qkey in MULTIFRAME_ONLY:
                continue
            sf = sr["single_frame"].get(qkey, {}).get("lexical_analysis", {})
            mf = sr["multi_frame"].get(qkey, {}).get("lexical_analysis", {})
            sf_tc = sf.get("temporal_connectors", 0)
            mf_tc = mf.get("temporal_connectors", 0)
            sf_dd = sf.get("dynamic_descriptors", 0)
            mf_dd = mf.get("dynamic_descriptors", 0)
            print(f"  {qkey}: temporal_connectors {sf_tc}→{mf_tc} | dynamic_descriptors {sf_dd}→{mf_dd}")
        # Q3 multi-frame only
        q3 = sr["multi_frame"].get("Q3_change", {}).get("lexical_analysis", {})
        print(f"  Q3_change (multi only): temporal={q3.get('temporal_connectors',0)} frame_refs={q3.get('frame_references',0)}")


if __name__ == "__main__":
    main()
