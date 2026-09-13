"""
UC4b: Negation & Counting Evaluation
======================================
Tests two failure modes:

1. COUNTING: "How many <Category> are in this image?"
   GT = count from key_object_infos; evaluate exact match, off-by-one, MAE.

2. NEGATION (existence with zero-count GT):
   For categories absent from the frame: "Are there any <Category> in this image?"
   GT = No. Tests whether model correctly denies presence.

Categories evaluated: Vehicle, Pedestrian, Cyclist, Traffic Cone, Barrier
(subset that appears in DriveLM key_object_infos)
"""

import os, json, random, re
from collections import defaultdict
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_JSON   = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm/v1_1_train_nus.json"
IMG_ROOT    = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm"
OUTPUT_JSON = "/home/xzh5180/Research/vlm-mobility/outputs/usecase4b_negation-counting_qwen25vl_drivelm.json"
MODEL_ID    = "Qwen/Qwen2.5-VL-7B-Instruct"

EVAL_SCENES           = 3
EVAL_FRAMES_PER_SCENE = 3
RANDOM_SEED           = 42
random.seed(RANDOM_SEED)

# Categories to probe (must match Category field in key_object_infos)
PROBE_CATEGORIES = ["Vehicle", "Pedestrian", "Cyclist", "Traffic Cone", "Barrier"]

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


def extract_integer(text: str):
    """Extract the first integer from model output; return None if not found."""
    # Match spelled-out numbers first (zero through ten)
    spelled = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "none": 0, "no ": 0,
    }
    t_lower = text.strip().lower()
    for word, val in spelled.items():
        if word in t_lower:
            return val
    nums = re.findall(r"\b(\d+)\b", text)
    return int(nums[0]) if nums else None


def extract_yes_no(text: str) -> str:
    t = text.strip().lower()
    if re.match(r"^\s*yes", t): return "yes"
    if re.match(r"^\s*no\b",  t): return "no"
    if "yes" in t[:80]: return "yes"
    if "no"  in t[:80]: return "no"
    return "unknown"


def ask_count(img_path: str, category: str) -> dict:
    question = (
        f"How many {category.lower()}s are visible in this image? "
        "Respond with a single number only."
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
    trimmed  = output_ids[:, inputs["input_ids"].shape[1]:]
    raw_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    parsed   = extract_integer(raw_text)
    return {"raw": raw_text.strip(), "parsed_count": parsed}


def ask_existence(img_path: str, category: str) -> dict:
    question = (
        f"Are there any {category.lower()}s in this image? "
        "Answer with Yes or No only."
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
    trimmed  = output_ids[:, inputs["input_ids"].shape[1]:]
    raw_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return {"raw": raw_text.strip(), "parsed": extract_yes_no(raw_text)}


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


def count_by_category(fd: dict) -> dict:
    """Return {Category: count} for objects in this frame (CAM_FRONT only if possible)."""
    counts = defaultdict(int)
    for obj in fd.get("key_object_infos", {}).values():
        cat = obj.get("Category", "").strip()
        if cat:
            counts[cat] += 1
    return dict(counts)


# ── evaluation ────────────────────────────────────────────────────────────────
frame_results = []

for idx, (st, ft, fd) in enumerate(eval_frames):
    print(f"\n[{idx+1}/{len(eval_frames)}] scene={st[:8]}  frame={ft[:8]}")

    img_path = fix_image_path(fd.get("image_paths", {}).get("CAM_FRONT", ""))
    if not os.path.exists(img_path):
        print(f"  WARNING: image missing → {img_path}")
        continue

    gt_counts = count_by_category(fd)
    print(f"  GT counts: {gt_counts}")

    rec = {
        "scene_token": st, "frame_token": ft, "image_path": img_path,
        "gt_counts": gt_counts,
        "counting_questions": [],
        "negation_questions": [],
    }

    for cat in PROBE_CATEGORIES:
        gt_n = gt_counts.get(cat, 0)

        # ── Counting question (always asked) ──────────────────────────────────
        print(f"  [COUNT] {cat}  gt={gt_n}")
        ans_c = ask_count(img_path, cat)
        pred_n = ans_c["parsed_count"]
        exact  = (pred_n == gt_n) if pred_n is not None else None
        off1   = (abs(pred_n - gt_n) <= 1) if pred_n is not None else None
        err    = (pred_n - gt_n) if pred_n is not None else None
        rec["counting_questions"].append({
            "category":    cat,
            "gt_count":    gt_n,
            "model_raw":   ans_c["raw"],
            "pred_count":  pred_n,
            "exact_match": exact,
            "off_by_one":  off1,
            "error":       err,
        })

        # ── Negation question (only for absent categories, gt_n == 0) ─────────
        if gt_n == 0:
            print(f"  [NEG]   {cat}  (absent, gt=0)")
            ans_e = ask_existence(img_path, cat)
            rec["negation_questions"].append({
                "category":      cat,
                "gt_label":      "no",
                "model_raw":     ans_e["raw"],
                "model_parsed":  ans_e["parsed"],
                "correct":       ans_e["parsed"] == "no",
            })

    frame_results.append(rec)

# ── aggregate counting metrics ────────────────────────────────────────────────
count_items = []
for fr in frame_results:
    for q in fr["counting_questions"]:
        if q["pred_count"] is not None:
            count_items.append(q)

n_count   = len(count_items)
exact_acc = sum(1 for q in count_items if q["exact_match"]) / n_count if n_count else 0
off1_acc  = sum(1 for q in count_items if q["off_by_one"])  / n_count if n_count else 0
mae       = sum(abs(q["error"]) for q in count_items) / n_count if n_count else 0
parse_rate_count = n_count / sum(
    len(fr["counting_questions"]) for fr in frame_results
) if frame_results else 0

# Per-category counting
cat_stats = defaultdict(lambda: {"n": 0, "exact": 0, "off1": 0, "sum_err": 0, "sum_abs_err": 0})
for q in count_items:
    c = q["category"]
    cat_stats[c]["n"]          += 1
    cat_stats[c]["exact"]      += int(q["exact_match"])
    cat_stats[c]["off1"]       += int(q["off_by_one"])
    cat_stats[c]["sum_err"]    += q["error"]
    cat_stats[c]["sum_abs_err"] += abs(q["error"])

per_category_counting = {}
for cat, s in cat_stats.items():
    n = s["n"]
    per_category_counting[cat] = {
        "n":          n,
        "exact_acc":  round(s["exact"] / n, 4),
        "off1_acc":   round(s["off1"]  / n, 4),
        "mae":        round(s["sum_abs_err"] / n, 4),
        "mean_error": round(s["sum_err"]     / n, 4),  # bias: positive = overcount
    }

# ── aggregate negation metrics ────────────────────────────────────────────────
neg_items = []
for fr in frame_results:
    neg_items.extend(fr["negation_questions"])

n_neg         = len(neg_items)
neg_correct   = sum(1 for q in neg_items if q["correct"])
neg_accuracy  = neg_correct / n_neg if n_neg else 0
yes_responses = sum(1 for q in neg_items if q["model_parsed"] == "yes")
yes_bias_neg  = yes_responses / n_neg if n_neg else 0  # should be 0 if model denies correctly

per_category_negation = defaultdict(lambda: {"n": 0, "correct": 0})
for q in neg_items:
    cat = q["category"]
    per_category_negation[cat]["n"]       += 1
    per_category_negation[cat]["correct"] += int(q["correct"])
per_category_negation = {
    k: {"n": v["n"], "accuracy": round(v["correct"] / v["n"], 4)}
    for k, v in per_category_negation.items()
}

summary = {
    "counting": {
        "total_questions":   sum(len(fr["counting_questions"]) for fr in frame_results),
        "parseable":         n_count,
        "parse_rate":        round(parse_rate_count, 4),
        "exact_match_acc":   round(exact_acc, 4),
        "off_by_one_acc":    round(off1_acc, 4),
        "mae":               round(mae, 4),
        "per_category":      per_category_counting,
    },
    "negation": {
        "total_questions":   n_neg,
        "correct":           neg_correct,
        "accuracy":          round(neg_accuracy, 4),
        "yes_bias":          round(yes_bias_neg, 4),
        "per_category":      per_category_negation,
    },
}

output = {
    "experiment":    "UC4b_negation_counting",
    "model":         MODEL_ID,
    "eval_frames":   len(eval_frames),
    "probe_categories": PROBE_CATEGORIES,
    "summary":       summary,
    "frame_results": frame_results,
}

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)

print("\n" + "=" * 60)
print("UC4b Results Summary")
print("=" * 60)
print("\n[COUNTING]")
print(f"  Questions parseable : {n_count}/{summary['counting']['total_questions']}  (parse_rate={parse_rate_count:.4f})")
print(f"  Exact match acc     : {exact_acc:.4f}")
print(f"  Off-by-one acc      : {off1_acc:.4f}")
print(f"  MAE                 : {mae:.4f}")
print("  Per-category:")
for cat, s in per_category_counting.items():
    print(f"    {cat:15s}  exact={s['exact_acc']:.3f}  MAE={s['mae']:.2f}  bias={s['mean_error']:+.2f}")

print("\n[NEGATION]  (categories absent from frame, gt=0)")
print(f"  Total questions : {n_neg}")
print(f"  Accuracy (said No): {neg_accuracy:.4f}")
print(f"  Yes-bias (false positives): {yes_bias_neg:.4f}")
print("  Per-category:")
for cat, s in per_category_negation.items():
    print(f"    {cat:15s}  acc={s['accuracy']:.3f}  n={s['n']}")
print(f"\nSaved: {OUTPUT_JSON}")
