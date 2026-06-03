"""
UC8b: Temporal Question Type Analysis
Multi-frame only. Three question categories (Action, Change, Event),
each with 2-3 variants. Analyzes task completion, temporal lexical density,
and cross-variant consistency.
"""

import json
import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# ─────────────────────────── CONFIG ───────────────────────────
MODEL_NAME   = "Qwen/Qwen2.5-VL-7B-Instruct"
IMAGE_ROOT   = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm"
DRIVELM_JSON = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm/v1_1_train_nus.json"
NUSCENES_DIR = "/home/xzh5180/Research/vlm-mobility/datasets/nuscenes/v1.0-trainval"
OUTPUT_PATH  = "/home/xzh5180/Research/vlm-mobility/outputs/usecase8b_temporal_question_types_qwen25vl_drivelm.json"
CAMERA       = "CAM_FRONT"

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

# Three categories, each with variants
QUESTIONS = {
    "action": {
        "A1": "What is the ego vehicle doing throughout this sequence?",
        "A2": "Describe the driving behavior of the ego vehicle from the first frame to the last frame.",
        "A3": "Is the ego vehicle accelerating, decelerating, turning, or maintaining speed? Describe how this changes across frames.",
    },
    "change": {
        "C1": "What changes between the first frame and the last frame?",
        "C2": "What new objects or vehicles appear or disappear across the frames?",
        "C3": "Does the traffic density around the ego vehicle increase or decrease from the beginning to the end of the sequence?",
    },
    "event": {
        "E1": "In which frame does the scene change the most?",
        "E2": "At what point in the sequence does the ego vehicle appear to be closest to other vehicles or obstacles?",
        "E3": "Which frame best represents the most complex or challenging driving situation in this sequence?",
    },
}

# Minimum completion criteria per category
COMPLETION_KEYWORDS = {
    "action": ["accelerat", "deceler", "slow", "turn", "stop", "mov", "brake", "speed",
                "maintain", "continu", "approach", "veer", "chang"],
    "change":  ["first frame", "last frame", "beginning", "end of", "initial", "final",
                "earlier", "later", "compared to", "whereas", "while", "from", "to"],
    "event":   ["frame 1", "frame 2", "frame 3", "first frame", "second frame", "third frame",
                "midway", "at the", "in the", "point in", "moment"],
}

TEMPORAL_CONNECTORS = ["first", "then", "next", "finally", "initially", "subsequently",
                        "before", "after", "as the sequence", "over time", "throughout",
                        "at first", "by the end"]
DYNAMIC_DESCRIPTORS = ["accelerating", "decelerating", "slowing", "turning", "stopping",
                        "moving", "braking", "speeding", "approaching", "veering"]
CHANGE_DESCRIPTORS  = ["changes", "shifts", "increases", "decreases", "appears to",
                        "seems to", "transitions", "emerging", "disappear", "new"]
FRAME_REFERENCES    = ["frame 1", "frame 2", "frame 3", "frame 4", "frame 5",
                        "first frame", "last frame", "second frame", "third frame",
                        "midway", "at the beginning", "by the end"]
# ──────────────────────────────────────────────────────────────


def resolve_image_path(raw_path: str) -> str:
    cleaned = raw_path.replace("../nuscenes/", "nuscenes/")
    return os.path.join(IMAGE_ROOT, cleaned)


def load_timestamps(nuscenes_dir: str) -> dict:
    sample_json = os.path.join(nuscenes_dir, "sample.json")
    with open(sample_json, "r") as f:
        samples = json.load(f)
    return {s["token"]: s["timestamp"] for s in samples}


def get_ordered_frames(scene_data: dict, requested_tokens: list, timestamps: dict) -> list:
    available = set(scene_data["key_frames"].keys())
    valid_tokens = [t for t in requested_tokens if t in available]
    return sorted(valid_tokens, key=lambda t: timestamps.get(t, 0))


def build_messages(image_paths: list, question: str) -> list:
    content = []
    for idx, img_path in enumerate(image_paths):
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


def lexical_analysis(text: str) -> dict:
    t = text.lower()
    def count(kws):
        return sum(1 for kw in kws if kw in t)
    return {
        "temporal_connectors": count(TEMPORAL_CONNECTORS),
        "dynamic_descriptors": count(DYNAMIC_DESCRIPTORS),
        "change_descriptors":  count(CHANGE_DESCRIPTORS),
        "frame_references":    count(FRAME_REFERENCES),
        "total_words":         len(text.split()),
    }


def check_completion(text: str, category: str) -> dict:
    """Check whether the response meets the minimum completion standard."""
    t = text.lower()
    keywords = COMPLETION_KEYWORDS[category]
    matched = [kw for kw in keywords if kw in t]
    completed = len(matched) > 0
    return {
        "completed": completed,
        "matched_keywords": matched,
    }


def cross_variant_consistency(variant_responses: dict, category: str) -> dict:
    """
    Detect potential contradictions across variants within a category.
    Checks directional terms that should be consistent.
    """
    direction_pairs = [
        (["accelerat", "speed up", "faster"],   ["deceler", "slow", "brake"]),
        (["increas", "more", "heavier"],         ["decreas", "less", "lighter", "fewer"]),
        (["turn left"],                           ["turn right"]),
        (["stop", "halt", "stationary"],         ["moving", "driving", "continu"]),
    ]

    all_text = " ".join(r.lower() for r in variant_responses.values())
    contradictions = []
    for pos_kws, neg_kws in direction_pairs:
        has_pos = any(kw in all_text for kw in pos_kws)
        has_neg = any(kw in all_text for kw in neg_kws)
        if has_pos and has_neg:
            contradictions.append({
                "positive_signals": [kw for kw in pos_kws if kw in all_text],
                "negative_signals": [kw for kw in neg_kws if kw in all_text],
            })

    return {
        "contradiction_count": len(contradictions),
        "contradictions": contradictions,
        "consistent": len(contradictions) == 0,
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

        ordered_tokens = get_ordered_frames(scene_data, requested_tokens, timestamps)
        print(f"  Frames ({len(ordered_tokens)}): {ordered_tokens}")

        image_paths = []
        for token in ordered_tokens:
            frame = scene_data["key_frames"][token]
            raw_path = frame["image_paths"][CAMERA]
            abs_path = resolve_image_path(raw_path)
            if not os.path.exists(abs_path):
                print(f"  WARNING: image not found: {abs_path}")
            image_paths.append(abs_path)

        scene_result = {
            "scene_description": scene_data.get("scene_description", ""),
            "frame_tokens_used": ordered_tokens,
            "num_frames": len(ordered_tokens),
            "categories": {},
        }

        for category, variants in QUESTIONS.items():
            print(f"\n  Category: {category.upper()}")
            category_result = {
                "variants": {},
                "cross_variant_consistency": None,
            }
            variant_responses = {}

            for vkey, qtext in variants.items():
                print(f"    {vkey} ...", end=" ", flush=True)
                messages = build_messages(image_paths, qtext)
                response = run_inference(model, processor, messages)
                lex      = lexical_analysis(response)
                comp     = check_completion(response, category)

                category_result["variants"][vkey] = {
                    "question":         qtext,
                    "response":         response,
                    "lexical_analysis": lex,
                    "completion":       comp,
                }
                variant_responses[vkey] = response
                print(f"done | completed={comp['completed']} | words={lex['total_words']} | "
                      f"temporal={lex['temporal_connectors']} | frame_refs={lex['frame_references']}")

            # Cross-variant consistency check
            consistency = cross_variant_consistency(variant_responses, category)
            category_result["cross_variant_consistency"] = consistency
            if not consistency["consistent"]:
                print(f"    ⚠ Contradictions detected: {consistency['contradiction_count']}")

            scene_result["categories"][category] = category_result

        results[scene_id] = scene_result

    # ── Save output ──
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {OUTPUT_PATH}")

    # ── Aggregated summary across scenes ──
    print("\n" + "="*60)
    print("AGGREGATED SUMMARY BY CATEGORY")
    print("="*60)

    for category in QUESTIONS:
        print(f"\n[{category.upper()}]")
        for vkey in QUESTIONS[category]:
            completions   = []
            temporal_list = []
            frame_ref_list = []
            for sr in results.values():
                vdata = sr["categories"][category]["variants"].get(vkey, {})
                completions.append(vdata.get("completion", {}).get("completed", False))
                lex = vdata.get("lexical_analysis", {})
                temporal_list.append(lex.get("temporal_connectors", 0))
                frame_ref_list.append(lex.get("frame_references", 0))

            n = len(completions)
            comp_rate = sum(completions) / n if n > 0 else 0
            avg_temp  = sum(temporal_list) / n if n > 0 else 0
            avg_fref  = sum(frame_ref_list) / n if n > 0 else 0
            print(f"  {vkey}: completion={comp_rate:.2f} | "
                  f"avg_temporal={avg_temp:.1f} | avg_frame_refs={avg_fref:.1f}")

        # Consistency across scenes
        contradiction_counts = [
            results[sid]["categories"][category]["cross_variant_consistency"]["contradiction_count"]
            for sid in results
        ]
        print(f"  Contradictions per scene: {contradiction_counts} "
              f"(avg={sum(contradiction_counts)/len(contradiction_counts):.1f})")


if __name__ == "__main__":
    main()
