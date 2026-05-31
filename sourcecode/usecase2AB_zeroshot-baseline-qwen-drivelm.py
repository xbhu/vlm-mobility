# usecase2AB_zeroshot-baseline-qwen-drivelm.py
"""
Zero-shot baseline for both Prediction and Planning QA.
Same eval subset as 2A/2B few-shot experiments, for direct comparison.
"""

import json, random, re, torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

DATA_JSON   = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm/v1_1_train_nus.json"
IMG_PREFIX  = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm/nuscenes/"
OUTPUT_JSON = "/home/xzh5180/Research/vlm-mobility/outputs/usecase2AB_zeroshot-baseline_qwen_drivelm.json"
MODEL_ID    = "Qwen/Qwen2.5-VL-7B-Instruct"

RANDOM_SEED   = 42
TARGET_QTYPES = ["prediction", "planning"]
random.seed(RANDOM_SEED)

with open(DATA_JSON) as f:
    data = json.load(f)

scene_tokens      = list(data.keys())
eval_scene_tokens = scene_tokens[:3]

eval_sets = {qtype: [] for qtype in TARGET_QTYPES}
for sc_tok in eval_scene_tokens:
    frames       = data[sc_tok]["key_frames"]
    frame_tokens = list(frames.keys())[:3]
    for fr_tok in frame_tokens:
        frame    = frames[fr_tok]
        img_path = frame["image_paths"]["CAM_FRONT"].replace("../nuscenes/", IMG_PREFIX)
        for qtype in TARGET_QTYPES:
            qa_list = frame.get("QA", {}).get(qtype, [])
            for qa in qa_list[:5]:
                eval_sets[qtype].append({
                    "scene": sc_tok, "frame": fr_tok,
                    "image": img_path,
                    "question": qa["Q"], "gt_answer": qa["A"]
                })

for qtype in TARGET_QTYPES:
    print(f"Eval set [{qtype}]: {len(eval_sets[qtype])} items")

bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model   = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, quantization_config=bnb_cfg, device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model.eval()

def build_zeroshot_messages(item):
    return [
        {"role": "system", "content": "You are a driving assistant analyzing front-camera images. "
                                       "Answer concisely based on visual evidence."},
        {"role": "user", "content": [
            {"type": "image", "image": item["image"]},
            {"type": "text",  "text": f"Question: {item['question']}\nAnswer:"}
        ]}
    ]

def run_inference(messages):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

def compute_recall(pred, gt):
    pt = set(re.findall(r'\w+', pred.lower()))
    gt = set(re.findall(r'\w+', gt.lower()))
    return len(pt & gt) / len(gt) if gt else 1.0

def compute_halluc(pred, gt):
    pt = set(re.findall(r'\w+', pred.lower()))
    gt = set(re.findall(r'\w+', gt.lower()))
    return len(pt - gt) / len(pt) if pt else 0.0

all_results = {}
for qtype in TARGET_QTYPES:
    print(f"\n--- zero-shot | {qtype} ---")
    results = []
    for i, item in enumerate(eval_sets[qtype]):
        messages = build_zeroshot_messages(item)
        try:
            pred = run_inference(messages)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at item {i}, skipping")
            torch.cuda.empty_cache()
            pred = ""
        recall = compute_recall(pred, item["gt_answer"])
        halluc = compute_halluc(pred, item["gt_answer"])
        results.append({
            "index": i, "scene": item["scene"], "frame": item["frame"],
            "question": item["question"], "gt_answer": item["gt_answer"],
            "prediction": pred, "recall": recall, "halluc": halluc
        })
        if i % 5 == 0:
            print(f"  [{i+1}/{len(eval_sets[qtype])}] recall={recall:.3f} halluc={halluc:.3f}")

    valid      = [r for r in results if r["prediction"]]
    avg_recall = sum(r["recall"] for r in valid) / len(valid)
    avg_halluc = sum(r["halluc"] for r in valid) / len(valid)
    print(f"  → Recall={avg_recall:.3f}  Halluc={avg_halluc:.3f}  N={len(valid)}/{len(results)}")
    all_results[qtype] = {
        "avg_recall": avg_recall, "avg_halluc": avg_halluc,
        "n_valid": len(valid), "results": results
    }

with open(OUTPUT_JSON, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved → {OUTPUT_JSON}")

print("\n=== Zero-shot Baseline Summary ===")
print(f"{'QType':<12} {'Recall':<8} {'Halluc':<8}")
print("-" * 30)
for qtype in TARGET_QTYPES:
    r = all_results[qtype]
    print(f"{qtype:<12} {r['avg_recall']:.3f}    {r['avg_halluc']:.3f}")
