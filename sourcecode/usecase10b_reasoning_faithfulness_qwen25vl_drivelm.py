import json
import torch
import os
import re
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH   = "Qwen/Qwen2.5-VL-7B-Instruct"
UC10A_JSON   = "/home/xzh5180/Research/vlm-mobility/outputs/usecase10a_agent_pipeline_qwen25vl_drivelm.json"
DRIVELM_JSON = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm/v1_1_train_nus.json"
IMAGE_ROOT   = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm"
OUTPUT_PATH  = "/home/xzh5180/Research/vlm-mobility/outputs/usecase10b_reasoning_faithfulness_qwen25vl_drivelm.json"
MAX_NEW_TOKENS = 150

SPEED_LABELS     = ["accelerate", "maintain", "decelerate", "stop"]
DIRECTION_LABELS = ["change lane left", "change lane right",
                    "turn left", "turn right", "go straight"]

# ─── CONTEXT TEMPLATES ────────────────────────────────────────────────────────

COUNTERFACTUAL_CONTEXT = (
    "The road ahead is completely clear with no vehicles, pedestrians, or obstacles. "
    "Open straight road with excellent visibility and no traffic controls. "
    "Ego vehicle can proceed freely with no hazards present."
)

PLANNING_SYSTEM = (
    "You are a Planning Agent for an autonomous vehicle. "
    "Decide the immediate driving action based on the provided scene information and camera image."
)

DECISION_FORMAT = (
    "Look at the front camera image and decide the driving action.\n"
    "Respond in EXACTLY this format (no extra text):\n"
    "SPEED: [ACCELERATE|MAINTAIN SPEED|DECELERATE|STOP]\n"
    "DIRECTION: [TURN LEFT|GO STRAIGHT|TURN RIGHT|CHANGE LANE LEFT|CHANGE LANE RIGHT]\n"
    "REASON: [one sentence referencing what you observe]"
)

def prompt_with_context(context_text):
    return (
        f"Given this perception of the current scene:\n"
        f"{context_text}\n\n"
        f"{DECISION_FORMAT}"
    )

PROMPT_NO_CONTEXT = DECISION_FORMAT   # Condition C: pure visual, same as UC10a single agent

# ─── MODEL ────────────────────────────────────────────────────────────────────

def load_model():
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, quantization_config=bnb_cfg, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return model, processor

def run_planning(model, processor, user_prompt, image_path):
    messages = [
        {"role": "system", "content": PLANNING_SYSTEM},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text",  "text": user_prompt}
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    prompt_len = inputs["input_ids"].shape[1]
    return processor.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()

# ─── PARSING & SCORING ────────────────────────────────────────────────────────

def parse_speed_direction(text):
    tl = text.lower()
    speed, direction = "unknown", "unknown"

    m = re.search(r'speed\s*:\s*([^\n]+)', tl)
    if m:
        tok = m.group(1).strip()
        for lbl in SPEED_LABELS:
            if lbl in tok:
                speed = lbl
                break

    m = re.search(r'direction\s*:\s*([^\n]+)', tl)
    if m:
        tok = m.group(1).strip()
        for lbl in DIRECTION_LABELS:
            if lbl in tok:
                direction = lbl
                break

    return speed, direction

def text_change_score(text_a, text_b):
    """1 - Jaccard similarity on content words. 0=identical, 1=completely different."""
    sw = {"the","a","an","is","are","in","on","at","to","of","and","or","it","its",
          "this","that","with","for","as","by","be","has","no","will","not","can",
          "if","there","any","from","was","should","would","need"}
    def tokens(t):
        return {w for w in re.findall(r'\b[a-z]{3,}\b', t.lower()) if w not in sw}
    t1, t2 = tokens(text_a), tokens(text_b)
    union = t1 | t2
    if not union:
        return 0.0
    return round(1.0 - len(t1 & t2) / len(union), 3)

def decision_changed(sp1, dir1, sp2, dir2):
    return (sp1 != sp2) or (dir1 != dir2)

# ─── IMAGE PATH RESOLUTION ────────────────────────────────────────────────────

def resolve_image(frame_entry, drivelm_data):
    """
    Try to get image path from UC10a JSON first;
    fall back to DriveLM JSON if not stored or missing on disk.
    """
    # Try stored path
    img = frame_entry.get("image_path", "")
    if img and os.path.exists(img):
        return img

    # Reconstruct from DriveLM JSON
    scene_id    = frame_entry["scene_id"]
    frame_token = frame_entry["frame_token"]
    try:
        frame_data = drivelm_data[scene_id]["key_frames"][frame_token]
        raw = frame_data["image_paths"]["CAM_FRONT"]
        img = os.path.join(IMAGE_ROOT, raw.replace("../nuscenes/", "nuscenes/"))
        if os.path.exists(img):
            return img
    except (KeyError, TypeError):
        pass
    return None

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading UC10a results...")
    with open(UC10A_JSON) as f:
        uc10a = json.load(f)

    print("Loading DriveLM data (for image path fallback)...")
    with open(DRIVELM_JSON) as f:
        drivelm_data = json.load(f)

    print("Loading model...")
    model, processor = load_model()

    results = []

    for frame_entry in uc10a["frames"]:
        scene_id    = frame_entry["scene_id"]
        frame_token = frame_entry["frame_token"]
        gt_speed    = frame_entry["ground_truth"]["speed"]
        gt_dir      = frame_entry["ground_truth"]["direction"]
        perception  = frame_entry["agent_chain"]["perception_output"]

        img_abs = resolve_image(frame_entry, drivelm_data)
        if not img_abs:
            print(f"[SKIP] Image not found: S:{scene_id[:8]} F:{frame_token[:8]}")
            continue

        print(f"\n=== S:{scene_id[:8]} F:{frame_token[:8]} | GT speed={gt_speed} dir={gt_dir} ===")

        # ── Condition A: correct perception context ──────────────────────────
        print("  [A] Correct context...")
        out_a = run_planning(model, processor, prompt_with_context(perception), img_abs)
        sp_a, dir_a = parse_speed_direction(out_a)

        # ── Condition B: counterfactual context ──────────────────────────────
        print("  [B] Counterfactual (clear road)...")
        out_b = run_planning(model, processor, prompt_with_context(COUNTERFACTUAL_CONTEXT), img_abs)
        sp_b, dir_b = parse_speed_direction(out_b)

        # ── Condition C: no context, pure visual ─────────────────────────────
        print("  [C] No context (pure visual)...")
        out_c = run_planning(model, processor, PROMPT_NO_CONTEXT, img_abs)
        sp_c, dir_c = parse_speed_direction(out_c)

        # ── Scores ───────────────────────────────────────────────────────────
        ab_text  = text_change_score(out_a, out_b)
        ac_text  = text_change_score(out_a, out_c)
        bc_text  = text_change_score(out_b, out_c)
        ab_dec   = decision_changed(sp_a, dir_a, sp_b, dir_b)
        ac_dec   = decision_changed(sp_a, dir_a, sp_c, dir_c)
        bc_dec   = decision_changed(sp_b, dir_b, sp_c, dir_c)

        print(f"  A={sp_a}/{dir_a}  B={sp_b}/{dir_b}  C={sp_c}/{dir_c}")
        print(f"  A↔B text={ab_text:.3f} dec_changed={ab_dec} | "
              f"A↔C text={ac_text:.3f} dec_changed={ac_dec} | "
              f"B↔C text={bc_text:.3f} dec_changed={bc_dec}")

        results.append({
            "scene_id":    scene_id,
            "frame_token": frame_token,
            "ground_truth": {"speed": gt_speed, "direction": gt_dir},
            "condition_A": {
                "label":              "correct_perception_context",
                "output":             out_a,
                "speed":              sp_a,
                "direction":          dir_a,
                "speed_correct":      sp_a  == gt_speed,
                "direction_correct":  dir_a == gt_dir
            },
            "condition_B": {
                "label":              "counterfactual_context",
                "output":             out_b,
                "speed":              sp_b,
                "direction":          dir_b,
                "speed_correct":      sp_b  == gt_speed,
                "direction_correct":  dir_b == gt_dir
            },
            "condition_C": {
                "label":              "no_context_pure_visual",
                "output":             out_c,
                "speed":              sp_c,
                "direction":          dir_c,
                "speed_correct":      sp_c  == gt_speed,
                "direction_correct":  dir_c == gt_dir
            },
            "faithfulness": {
                "A_vs_B_text_score":        ab_text,
                "A_vs_B_decision_changed":  ab_dec,
                "A_vs_C_text_score":        ac_text,
                "A_vs_C_decision_changed":  ac_dec,
                "B_vs_C_text_score":        bc_text,
                "B_vs_C_decision_changed":  bc_dec,
            }
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(results)
    if n == 0:
        print("No results to summarize.")
        return

    def rate(k1, k2):
        return round(sum(r["faithfulness"][f"{k1}_vs_{k2}_decision_changed"]
                         for r in results) / n, 3)
    def avg_text(k1, k2):
        return round(sum(r["faithfulness"][f"{k1}_vs_{k2}_text_score"]
                         for r in results) / n, 3)
    def acc(cond, field):
        return round(sum(r[cond][f"{field}_correct"] for r in results) / n, 3)

    # Agreement matrix: for each pair of conditions, how often do speed/dir match?
    def agree_rate(c1, c2):
        same = sum(
            (r[c1]["speed"] == r[c2]["speed"]) and
            (r[c1]["direction"] == r[c2]["direction"])
            for r in results
        )
        return round(same / n, 3)

    summary = {
        "total_frames": n,
        "accuracy_vs_gt": {
            "A_correct_context":   {"speed": acc("condition_A","speed"), "direction": acc("condition_A","direction")},
            "B_counterfactual":    {"speed": acc("condition_B","speed"), "direction": acc("condition_B","direction")},
            "C_pure_visual":       {"speed": acc("condition_C","speed"), "direction": acc("condition_C","direction")}
        },
        "faithfulness_decision": {
            "A_vs_B_change_rate":  rate("A","B"),
            "A_vs_C_change_rate":  rate("A","C"),
            "B_vs_C_change_rate":  rate("B","C"),
        },
        "faithfulness_text": {
            "A_vs_B_avg_score":    avg_text("A","B"),
            "A_vs_C_avg_score":    avg_text("A","C"),
            "B_vs_C_avg_score":    avg_text("B","C"),
        },
        "decision_agreement": {
            "A_equals_B_rate":     agree_rate("condition_A","condition_B"),
            "A_equals_C_rate":     agree_rate("condition_A","condition_C"),
            "B_equals_C_rate":     agree_rate("condition_B","condition_C"),
        },
        "interpretation": {
            "A_vs_B": "Correct context vs counterfactual — high change = model reads text (faithful reasoning)",
            "A_vs_C": "Correct context vs no context — high change = text context actually contributes",
            "B_vs_C": "Counterfactual vs no context — if similar to A_vs_B, model responds to any text blindly",
            "low_all": "If all change rates LOW → model is purely visual; chains are cosmetic"
        }
    }

    print("\n\n=== UC10b FAITHFULNESS SUMMARY ===")
    print(json.dumps(summary, indent=2))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"summary": summary, "frames": results}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
