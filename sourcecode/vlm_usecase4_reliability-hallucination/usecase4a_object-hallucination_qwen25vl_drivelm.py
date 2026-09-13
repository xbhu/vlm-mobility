"""
UC4a: Object Hallucination Evaluation (POPE-style)
====================================================
Builds positive/negative question pairs:
  Positive: "Is there a <X> in this image?" where X exists in the frame (GT label = yes)
  Negative: same question where X is absent from this frame but exists elsewhere (GT label = no)

Metrics: yes-bias, precision, recall, F1, accuracy, confusion matrix
"""

import os, json, random, re
from pathlib import Path
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_JSON   = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm/v1_1_train_nus.json"
IMG_ROOT    = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm"
OUTPUT_JSON = "/home/xzh5180/Research/vlm-mobility/outputs/usecase4a_object-hallucination_qwen25vl_drivelm.json"
MODEL_ID    = "Qwen/Qwen2.5-VL-7B-Instruct"

# ── fixed evaluation subset (UC1–UC3 consistent) ──────────────────────────────
EVAL_SCENES            = 3
EVAL_FRAMES_PER_SCENE  = 3
MAX_POS_PER_FRAME      = 5   # positive questions per frame
MAX_NEG_PER_FRAME      = 5   # negative questions per frame
RANDOM_SEED            = 42
random.seed(RANDOM_SEED)

# ── model load ─────────────────────────────────────────────────────────────────
print("Loading model (4-bit)...")
bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, quantization_config=bnb_cfg, device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model.eval()
print("Model loaded.")


def fix_image_path(raw_path: str) -> str:
    return raw_path.replace("../nuscenes/", IMG_ROOT + "/nuscenes/")


def extract_yes_no(text: str) -> str:
    """Parse 'yes'/'no'/'unknown' from model output."""
    t = text.strip().lower()
    # Strong signals first
    if re.match(r"^\s*yes", t):
        return "yes"
    if re.match(r"^\s*no\b", t):
        return "no"
    # Fallback: search first 60 chars
    snippet = t[:60]
    if "yes" in snippet:
        return "yes"
    if "no" in snippet:
        return "no"
    return "unknown"


def ask_presence(img_path: str, object_desc: str) -> dict:
    """Ask existence question; return raw text + parsed yes/no."""
    question = (
        f"Is there a {object_desc} in this image? "
        "Answer with Yes or No only, with no additional explanation."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text",  "text": question},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=16)
    trimmed   = output_ids[:, inputs["input_ids"].shape[1]:]
    raw_text  = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return {"raw": raw_text.strip(), "parsed": extract_yes_no(raw_text)}


# ── data loading ───────────────────────────────────────────────────────────────
print("Loading dataset...")
with open(DATA_JSON, "r") as f:
    data = json.load(f)

scene_tokens = list(data.keys())[:EVAL_SCENES]

eval_frames = []  # list of (scene_token, frame_token, frame_data)
for st in scene_tokens:
    frames = list(data[st]["key_frames"].items())
    for ft, fd in frames[:EVAL_FRAMES_PER_SCENE]:
        eval_frames.append((st, ft, fd))

print(f"Eval frames: {len(eval_frames)}")

# ── build per-frame object description sets ───────────────────────────────────
# Use Visual_description as the object identifier; fall back to Category if absent
def get_descriptions(fd: dict) -> set:
    descs = set()
    for obj in fd.get("key_object_infos", {}).values():
        vd = obj.get("Visual_description", "").strip()
        if vd:
            descs.add(vd)
        elif obj.get("Category", "").strip():
            descs.add(obj["Category"].strip())
    return descs

frame_descs = {}   # ft → set of descriptions present in frame
for st, ft, fd in eval_frames:
    frame_descs[ft] = get_descriptions(fd)

global_pool = set()
for s in frame_descs.values():
    global_pool.update(s)

print(f"Global object pool: {len(global_pool)} unique descriptions")

# ── main evaluation loop ──────────────────────────────────────────────────────
frame_results = []

for idx, (st, ft, fd) in enumerate(eval_frames):
    print(f"\n[{idx+1}/{len(eval_frames)}] scene={st[:8]}  frame={ft[:8]}")

    img_raw  = fd.get("image_paths", {}).get("CAM_FRONT", "")
    img_path = fix_image_path(img_raw)
    if not os.path.exists(img_path):
        print(f"  WARNING: image missing → {img_path}")
        continue

    present = frame_descs[ft]
    absent  = global_pool - present

    pos_sample = random.sample(sorted(present), min(MAX_POS_PER_FRAME, len(present)))
    neg_sample = random.sample(sorted(absent),  min(MAX_NEG_PER_FRAME, len(absent)))

    rec = {
        "scene_token": st,
        "frame_token": ft,
        "image_path":  img_path,
        "n_gt_objects": len(present),
        "positive_questions": [],
        "negative_questions": [],
    }

    for desc in pos_sample:
        print(f"  [POS] {desc}")
        ans = ask_presence(img_path, desc)
        rec["positive_questions"].append({
            "object": desc, "gt_label": "yes",
            "model_raw": ans["raw"], "model_parsed": ans["parsed"],
            "correct": ans["parsed"] == "yes",
        })

    for desc in neg_sample:
        print(f"  [NEG] {desc}")
        ans = ask_presence(img_path, desc)
        rec["negative_questions"].append({
            "object": desc, "gt_label": "no",
            "model_raw": ans["raw"], "model_parsed": ans["parsed"],
            "correct": ans["parsed"] == "no",
        })

    frame_results.append(rec)

# ── aggregate metrics ─────────────────────────────────────────────────────────
pairs = []
for fr in frame_results:
    for q in fr["positive_questions"]:
        pairs.append(("yes", q["model_parsed"]))
    for q in fr["negative_questions"]:
        pairs.append(("no",  q["model_parsed"]))

total    = len(pairs)
yes_pred = sum(1 for _, p in pairs if p == "yes")
yes_bias = yes_pred / total if total else 0

TP = sum(1 for g, p in pairs if g == "yes" and p == "yes")
FP = sum(1 for g, p in pairs if g == "no"  and p == "yes")
FN = sum(1 for g, p in pairs if g == "yes" and p == "no")
TN = sum(1 for g, p in pairs if g == "no"  and p == "no")

precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
accuracy  = (TP + TN) / total if total else 0.0

# Per-frame breakdown
per_frame = []
for fr in frame_results:
    fp_pairs = (
        [("yes", q["model_parsed"]) for q in fr["positive_questions"]] +
        [("no",  q["model_parsed"]) for q in fr["negative_questions"]]
    )
    n  = len(fp_pairs)
    yp = sum(1 for _, p in fp_pairs if p == "yes")
    per_frame.append({
        "frame_token":    fr["frame_token"],
        "n_questions":    n,
        "yes_pred_count": yp,
        "yes_bias":       round(yp / n, 4) if n else 0,
        "pos_correct":    sum(1 for q in fr["positive_questions"] if q["correct"]),
        "pos_total":      len(fr["positive_questions"]),
        "neg_correct":    sum(1 for q in fr["negative_questions"] if q["correct"]),
        "neg_total":      len(fr["negative_questions"]),
    })

summary = {
    "total_questions": total,
    "positive_count":  TP + FN,
    "negative_count":  FP + TN,
    "yes_predictions": yes_pred,
    "yes_bias":        round(yes_bias, 4),
    "TP": TP, "FP": FP, "FN": FN, "TN": TN,
    "precision": round(precision, 4),
    "recall":    round(recall, 4),
    "f1":        round(f1, 4),
    "accuracy":  round(accuracy, 4),
    "per_frame": per_frame,
}

output = {
    "experiment":  "UC4a_object_hallucination_POPE_style",
    "model":       MODEL_ID,
    "eval_frames": len(eval_frames),
    "summary":     summary,
    "frame_results": frame_results,
}

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)

# ── print summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("UC4a Results Summary")
print("=" * 60)
print(f"Total questions : {total}  (pos={TP+FN}, neg={FP+TN})")
print(f"Yes-bias        : {yes_bias:.4f}  ({yes_pred}/{total} answered Yes)")
print(f"Accuracy        : {accuracy:.4f}")
print(f"Precision       : {precision:.4f}")
print(f"Recall          : {recall:.4f}")
print(f"F1              : {f1:.4f}")
print(f"Confusion       : TP={TP}  FP={FP}  FN={FN}  TN={TN}")
print(f"\nSaved: {OUTPUT_JSON}")
