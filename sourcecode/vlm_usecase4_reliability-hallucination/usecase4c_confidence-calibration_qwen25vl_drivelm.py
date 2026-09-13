"""
UC4c: Confidence Calibration Evaluation
========================================
Reuses the same positive/negative presence questions as UC4a,
but asks the model to provide an answer AND a confidence score [0.0–1.0].

Metrics:
  - Calibration curve: for each confidence bucket [0,0.2), [0.2,0.4), …, [0.8,1.0]
    plot mean predicted confidence vs actual accuracy
  - ECE (Expected Calibration Error)
  - Overconfidence rate: % of wrong answers with confidence > 0.7
  - Underconfidence rate: % of correct answers with confidence < 0.3
  - Confidence distribution by correctness
"""

import os, json, random, re
from pathlib import Path
from collections import defaultdict
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_JSON   = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm/v1_1_train_nus.json"
IMG_ROOT    = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm"
OUTPUT_JSON = "/home/xzh5180/Research/vlm-mobility/outputs/usecase4c_confidence-calibration_qwen25vl_drivelm.json"
MODEL_ID    = "Qwen/Qwen2.5-VL-7B-Instruct"

EVAL_SCENES           = 3
EVAL_FRAMES_PER_SCENE = 3
MAX_POS_PER_FRAME     = 5
MAX_NEG_PER_FRAME     = 5
RANDOM_SEED           = 42
random.seed(RANDOM_SEED)

N_BINS = 5  # calibration bins: [0,0.2), [0.2,0.4), ..., [0.8,1.0]

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


def parse_answer_and_confidence(text: str) -> dict:
    """
    Expected output format:
      Answer: Yes\nConfidence: 0.85

    Fallback: try to extract any yes/no and any float 0-1 from the text.
    Returns {"answer": "yes"/"no"/"unknown", "confidence": float or None}
    """
    t = text.strip().lower()

    # Try structured parse
    answer     = "unknown"
    confidence = None

    ans_match  = re.search(r"answer\s*[:\-]?\s*(yes|no)", t)
    conf_match = re.search(r"confidence\s*[:\-]?\s*([0-9]*\.?[0-9]+)", t)

    if ans_match:
        answer = ans_match.group(1)
    else:
        if re.match(r"^\s*yes", t):
            answer = "yes"
        elif re.match(r"^\s*no\b", t):
            answer = "no"
        elif "yes" in t[:80]:
            answer = "yes"
        elif "no" in t[:80]:
            answer = "no"

    if conf_match:
        val = float(conf_match.group(1))
        # Normalize if given as 0-100
        if val > 1.0:
            val = val / 100.0
        confidence = round(min(max(val, 0.0), 1.0), 4)

    # If confidence not found, search for any float 0-1 in the text
    if confidence is None:
        floats = re.findall(r"\b(0\.\d+|1\.0+)\b", t)
        if floats:
            confidence = round(float(floats[-1]), 4)

    return {"answer": answer, "confidence": confidence}


def ask_with_confidence(img_path: str, object_desc: str) -> dict:
    """Ask existence question; request structured Answer + Confidence response."""
    question = (
        f"Is there a {object_desc} in this image?\n"
        "Respond in exactly this format:\n"
        "Answer: Yes or No\n"
        "Confidence: a number between 0.0 and 1.0 indicating how confident you are"
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
        output_ids = model.generate(**inputs, max_new_tokens=64)
    trimmed  = output_ids[:, inputs["input_ids"].shape[1]:]
    raw_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    parsed   = parse_answer_and_confidence(raw_text)
    return {"raw": raw_text.strip(), **parsed}


# ── data loading ───────────────────────────────────────────────────────────────
print("Loading dataset...")
with open(DATA_JSON, "r") as f:
    data = json.load(f)

scene_tokens = list(data.keys())[:EVAL_SCENES]

eval_frames = []
for st in scene_tokens:
    for ft, fd in list(data[st]["key_frames"].items())[:EVAL_FRAMES_PER_SCENE]:
        eval_frames.append((st, ft, fd))

print(f"Eval frames: {len(eval_frames)}")


def get_descriptions(fd):
    descs = set()
    for obj in fd.get("key_object_infos", {}).values():
        vd = obj.get("Visual_description", "").strip()
        if vd:
            descs.add(vd)
        elif obj.get("Category", "").strip():
            descs.add(obj["Category"].strip())
    return descs


frame_descs = {ft: get_descriptions(fd) for st, ft, fd in eval_frames}
global_pool = set()
for s in frame_descs.values():
    global_pool.update(s)

# ── evaluation ────────────────────────────────────────────────────────────────
frame_results = []

for idx, (st, ft, fd) in enumerate(eval_frames):
    print(f"\n[{idx+1}/{len(eval_frames)}] scene={st[:8]}  frame={ft[:8]}")

    img_path = fix_image_path(fd.get("image_paths", {}).get("CAM_FRONT", ""))
    if not os.path.exists(img_path):
        print(f"  WARNING: image missing → {img_path}")
        continue

    present = frame_descs[ft]
    absent  = global_pool - present

    pos_sample = random.sample(sorted(present), min(MAX_POS_PER_FRAME, len(present)))
    neg_sample = random.sample(sorted(absent),  min(MAX_NEG_PER_FRAME, len(absent)))

    rec = {
        "scene_token": st, "frame_token": ft, "image_path": img_path,
        "positive_questions": [], "negative_questions": [],
    }

    for desc in pos_sample:
        print(f"  [POS] {desc}")
        ans = ask_with_confidence(img_path, desc)
        correct = ans["answer"] == "yes"
        rec["positive_questions"].append({
            "object": desc, "gt_label": "yes",
            "model_raw": ans["raw"], "model_answer": ans["answer"],
            "confidence": ans["confidence"], "correct": correct,
        })

    for desc in neg_sample:
        print(f"  [NEG] {desc}")
        ans = ask_with_confidence(img_path, desc)
        correct = ans["answer"] == "no"
        rec["negative_questions"].append({
            "object": desc, "gt_label": "no",
            "model_raw": ans["raw"], "model_answer": ans["answer"],
            "confidence": ans["confidence"], "correct": correct,
        })

    frame_results.append(rec)

# ── calibration computation ───────────────────────────────────────────────────
all_items = []  # (correct: bool, confidence: float)
for fr in frame_results:
    for q in fr["positive_questions"] + fr["negative_questions"]:
        if q["confidence"] is not None:
            all_items.append((q["correct"], q["confidence"]))

# Bin edges: 5 bins of width 0.2
bin_edges = [i / N_BINS for i in range(N_BINS + 1)]
bins = defaultdict(list)  # bin_idx → list of (correct, confidence)

for correct, conf in all_items:
    b = min(int(conf * N_BINS), N_BINS - 1)
    bins[b].append((correct, conf))

calibration_curve = []
ece_numerator = 0.0
total_with_conf = len(all_items)

for b in range(N_BINS):
    items = bins[b]
    if not items:
        calibration_curve.append({
            "bin": f"[{bin_edges[b]:.1f}, {bin_edges[b+1]:.1f})",
            "count": 0, "mean_confidence": None, "accuracy": None,
        })
        continue
    mean_conf = sum(c for _, c in items) / len(items)
    acc       = sum(1 for ok, _ in items if ok) / len(items)
    ece_numerator += len(items) * abs(acc - mean_conf)
    calibration_curve.append({
        "bin":             f"[{bin_edges[b]:.1f}, {bin_edges[b+1]:.1f})",
        "count":           len(items),
        "mean_confidence": round(mean_conf, 4),
        "accuracy":        round(acc, 4),
        "gap":             round(abs(acc - mean_conf), 4),
    })

ece = ece_numerator / total_with_conf if total_with_conf else 0.0

# Overconfidence / underconfidence rates
overconf_wrong  = sum(1 for ok, c in all_items if not ok and c > 0.7)
underconf_right = sum(1 for ok, c in all_items if ok  and c < 0.3)
n_wrong  = sum(1 for ok, _ in all_items if not ok)
n_right  = sum(1 for ok, _ in all_items if ok)

overconf_rate  = overconf_wrong  / n_wrong  if n_wrong  else 0.0
underconf_rate = underconf_right / n_right  if n_right  else 0.0

# Confidence parse rate (how often did model produce a parseable confidence?)
total_q = sum(
    len(fr["positive_questions"]) + len(fr["negative_questions"])
    for fr in frame_results
)
conf_parsed = sum(
    1 for fr in frame_results
    for q in fr["positive_questions"] + fr["negative_questions"]
    if q["confidence"] is not None
)
conf_parse_rate = conf_parsed / total_q if total_q else 0.0

# Accuracy (with-confidence subset vs all)
n_correct = sum(1 for ok, _ in all_items if ok)
accuracy  = n_correct / total_with_conf if total_with_conf else 0.0

summary = {
    "total_questions":      total_q,
    "confidence_parsed":    conf_parsed,
    "conf_parse_rate":      round(conf_parse_rate, 4),
    "accuracy":             round(accuracy, 4),
    "ece":                  round(ece, 4),
    "overconfidence_rate":  round(overconf_rate, 4),
    "underconfidence_rate": round(underconf_rate, 4),
    "n_wrong":  n_wrong,
    "n_right":  n_right,
    "calibration_curve":    calibration_curve,
}

output = {
    "experiment":  "UC4c_confidence_calibration",
    "model":       MODEL_ID,
    "eval_frames": len(eval_frames),
    "summary":     summary,
    "frame_results": frame_results,
}

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)

print("\n" + "=" * 60)
print("UC4c Results Summary")
print("=" * 60)
print(f"Total questions     : {total_q}")
print(f"Confidence parsed   : {conf_parsed}/{total_q}  (parse_rate={conf_parse_rate:.4f})")
print(f"Accuracy            : {accuracy:.4f}")
print(f"ECE                 : {ece:.4f}")
print(f"Overconfidence rate : {overconf_rate:.4f}  ({overconf_wrong}/{n_wrong} wrong answered with conf>0.7)")
print(f"Underconfidence rate: {underconf_rate:.4f}  ({underconf_right}/{n_right} correct answered with conf<0.3)")
print("\nCalibration curve:")
for row in calibration_curve:
    if row["count"] > 0:
        print(f"  {row['bin']}  n={row['count']:2d}  mean_conf={row['mean_confidence']:.3f}  acc={row['accuracy']:.3f}  gap={row['gap']:.3f}")
    else:
        print(f"  {row['bin']}  n=0  (empty)")
print(f"\nSaved: {OUTPUT_JSON}")
