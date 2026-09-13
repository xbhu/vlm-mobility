# usecase2C_shotnum-ablation-qwen-drivelm.py
"""
Use Case 2C: Shot number ablation for Prediction + Planning QA
Tests 0/1/2/3 shots to find the inflection point.
3-shot may OOM on A2000; handled gracefully.
"""

import json, random, re, torch
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

DATA_JSON   = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm/v1_1_train_nus.json"
IMG_PREFIX  = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm/nuscenes/"
OUTPUT_JSON = "/home/xzh5180/Research/vlm-mobility/outputs/usecase2C_shotnum-ablation_qwen_drivelm.json"
MODEL_ID    = "Qwen/Qwen2.5-VL-7B-Instruct"

SHOT_COUNTS   = [1, 2, 3]
TARGET_QTYPES = ["prediction", "planning"]
RANDOM_SEED   = 42
random.seed(RANDOM_SEED)

with open(DATA_JSON) as f:
    data = json.load(f)

scene_tokens = list(data.keys())
eval_scene_tokens = scene_tokens[:3]

# ── build eval sets and example pools for both qtypes ─────────────────────
eval_sets     = {qtype: [] for qtype in TARGET_QTYPES}
example_pools = {qtype: [] for qtype in TARGET_QTYPES}

for sc_tok in scene_tokens:
    frames = data[sc_tok]["key_frames"]
    is_eval = sc_tok in eval_scene_tokens
    frame_items = list(frames.items())[:3] if is_eval else list(frames.items())
    for fr_tok, frame in frame_items:
        img_path = frame["image_paths"]["CAM_FRONT"].replace("../nuscenes/", IMG_PREFIX)
        for qtype in TARGET_QTYPES:
            qa_list = frame.get("QA", {}).get(qtype, [])
            for qa in (qa_list[:5] if is_eval else qa_list):
                item = {"scene": sc_tok, "frame": fr_tok,
                        "image": img_path, "question": qa["Q"], "gt_answer": qa["A"]}
                if is_eval:
                    eval_sets[qtype].append(item)
                elif Path(img_path).exists():
                    example_pools[qtype].append(item)

for qtype in TARGET_QTYPES:
    print(f"Eval set [{qtype}]: {len(eval_sets[qtype])} items")
    print(f"Example pool [{qtype}]: {len(example_pools[qtype])} items")

# ── load model ─────────────────────────────────────────────────────────────
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, quantization_config=bnb_cfg, device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model.eval()

# ── helpers ────────────────────────────────────────────────────────────────
def build_messages(target, examples):
    content = []
    for ex in examples:
        content.append({"type": "image", "image": ex["image"]})
        content.append({"type": "text",
                         "text": f"Question: {ex['question']}\nAnswer: {ex['gt_answer']}\n\n"})
    content.append({"type": "image", "image": target["image"]})
    content.append({"type": "text",
                     "text": f"Question: {target['question']}\nAnswer:"})
    return [
        {"role": "system", "content": "You are a driving assistant. Answer concisely based on visual evidence."},
        {"role": "user", "content": content}
    ]

def infer(messages):
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

# ── main loop ──────────────────────────────────────────────────────────────
all_results = {}
for qtype in TARGET_QTYPES:
    all_results[qtype] = {}
    pool = example_pools[qtype]
    for n_shot in SHOT_COUNTS:
        print(f"\n--- {qtype} | {n_shot}-shot ---")
        shot_results = []
        oom_count = 0
        for item in eval_sets[qtype]:
            examples = random.sample(pool, min(n_shot, len(pool)))
            try:
                pred = infer(build_messages(item, examples))
            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                torch.cuda.empty_cache()
                pred = ""
            shot_results.append({
                "question": item["question"], "gt": item["gt_answer"],
                "pred": pred,
                "recall": compute_recall(pred, item["gt_answer"]),
                "halluc": compute_halluc(pred, item["gt_answer"])
            })
        valid = [r for r in shot_results if r["pred"]]
        avg_r = sum(r["recall"] for r in valid) / len(valid) if valid else 0.0
        avg_h = sum(r["halluc"] for r in valid) / len(valid) if valid else 0.0
        print(f"  Recall={avg_r:.3f}  Halluc={avg_h:.3f}  OOM={oom_count}")
        all_results[qtype][str(n_shot)] = {
            "avg_recall": avg_r, "avg_halluc": avg_h,
            "n_valid": len(valid), "oom": oom_count,
            "detail": shot_results
        }

# ── save ───────────────────────────────────────────────────────────────────
with open(OUTPUT_JSON, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved → {OUTPUT_JSON}")

# ── summary table ──────────────────────────────────────────────────────────
print("\n=== Shot Ablation Summary ===")
print(f"{'QType':<12} {'Shots':<6} {'Recall':<8} {'Halluc':<8}")
print("-" * 36)
for qtype in TARGET_QTYPES:
    for n in SHOT_COUNTS:
        r = all_results[qtype][str(n)]
        print(f"{qtype:<12} {n:<6} {r['avg_recall']:.3f}    {r['avg_halluc']:.3f}")
