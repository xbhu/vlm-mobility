# usecase2B_planning-fewshot-qwen-drivelm.py
"""
Use Case 2B: Few-shot prompting for Planning QA
- 2-shot: 2 example (image, Q, A) pairs prepended before target question
- Examples drawn from training scenes NOT in eval subset
- Evaluates Planning QA only
"""

import json, random, re, torch
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# ── paths ──────────────────────────────────────────────────────────────────
DATA_JSON   = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm/v1_1_train_nus.json"
IMG_PREFIX  = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm/nuscenes/"
OUTPUT_JSON = "/home/xzh5180/Research/vlm-mobility/outputs/usecase2B_planning-fewshot_qwen_drivelm.json"
MODEL_ID    = "Qwen/Qwen2.5-VL-7B-Instruct"

NUM_SHOTS   = 2
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

TARGET_QTYPE = "planning"

# ── load data ──────────────────────────────────────────────────────────────
with open(DATA_JSON) as f:
    data = json.load(f)

scene_tokens = list(data.keys())
eval_scene_tokens = scene_tokens[:3]
eval_set = []

for sc_tok in eval_scene_tokens:
    frames = data[sc_tok]["key_frames"]
    frame_tokens = list(frames.keys())[:3]
    for fr_tok in frame_tokens:
        frame = frames[fr_tok]
        img_path = frame["image_paths"]["CAM_FRONT"].replace("../nuscenes/", IMG_PREFIX)
        qa_list = frame.get("QA", {}).get(TARGET_QTYPE, [])
        for qa in qa_list[:5]:
            eval_set.append({
                "scene": sc_tok, "frame": fr_tok,
                "image": img_path,
                "question": qa["Q"], "gt_answer": qa["A"]
            })

print(f"Eval set: {len(eval_set)} items (type={TARGET_QTYPE})")

# ── build few-shot example pool (exclude eval scenes) ─────────────────────
example_pool = []
for sc_tok in scene_tokens:
    if sc_tok in eval_scene_tokens:
        continue
    for fr_tok, frame in data[sc_tok]["key_frames"].items():
        img_path = frame["image_paths"]["CAM_FRONT"].replace("../nuscenes/", IMG_PREFIX)
        qa_list = frame.get("QA", {}).get(TARGET_QTYPE, [])
        for qa in qa_list:
            if Path(img_path).exists():
                example_pool.append({
                    "image": img_path,
                    "question": qa["Q"],
                    "answer": qa["A"]
                })

print(f"Example pool: {len(example_pool)} items")

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

# ── inference ──────────────────────────────────────────────────────────────
def build_fewshot_messages(target_item, examples):
    content = []
    for ex in examples:
        content.append({"type": "image", "image": ex["image"]})
        content.append({"type": "text",
                         "text": f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"})
    content.append({"type": "image", "image": target_item["image"]})
    content.append({"type": "text",
                     "text": f"Question: {target_item['question']}\nAnswer:"})
    return [
        {"role": "system", "content": "You are a driving assistant analyzing front-camera images. "
                                       "Answer concisely based on visual evidence."},
        {"role": "user", "content": content}
    ]

def run_inference(messages):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

# ── metrics ────────────────────────────────────────────────────────────────
def compute_recall(pred, gt):
    pred_tok = set(re.findall(r'\w+', pred.lower()))
    gt_tok   = set(re.findall(r'\w+', gt.lower()))
    if not gt_tok:
        return 1.0
    return len(pred_tok & gt_tok) / len(gt_tok)

def compute_halluc(pred, gt):
    pred_tok = set(re.findall(r'\w+', pred.lower()))
    gt_tok   = set(re.findall(r'\w+', gt.lower()))
    if not pred_tok:
        return 0.0
    return len(pred_tok - gt_tok) / len(pred_tok)

# ── main loop ──────────────────────────────────────────────────────────────
results = []
for i, item in enumerate(eval_set):
    examples = random.sample(example_pool, NUM_SHOTS)
    messages = build_fewshot_messages(item, examples)
    try:
        pred = run_inference(messages)
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM at item {i}, skipping")
        torch.cuda.empty_cache()
        pred = ""
    recall  = compute_recall(pred, item["gt_answer"])
    halluc  = compute_halluc(pred, item["gt_answer"])
    results.append({
        "index": i, "scene": item["scene"], "frame": item["frame"],
        "question": item["question"], "gt_answer": item["gt_answer"],
        "prediction": pred, "recall": recall, "halluc": halluc,
        "num_shots": NUM_SHOTS
    })
    if i % 5 == 0:
        print(f"[{i+1}/{len(eval_set)}] recall={recall:.3f} halluc={halluc:.3f}")

# ── summary ────────────────────────────────────────────────────────────────
valid = [r for r in results if r["prediction"]]
avg_recall = sum(r["recall"] for r in valid) / len(valid)
avg_halluc = sum(r["halluc"] for r in valid) / len(valid)
print(f"\n=== Use Case 2B Results ({NUM_SHOTS}-shot, {TARGET_QTYPE}) ===")
print(f"Recall:        {avg_recall:.3f}")
print(f"Hallucination: {avg_halluc:.3f}")
print(f"N={len(valid)}/{len(results)}")

summary = {
    "experiment": "usecase2B_planning_fewshot",
    "model": MODEL_ID, "num_shots": NUM_SHOTS,
    "qtype": TARGET_QTYPE, "n_eval": len(valid),
    "avg_recall": avg_recall, "avg_halluc": avg_halluc,
    "results": results
}
with open(OUTPUT_JSON, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved → {OUTPUT_JSON}")
