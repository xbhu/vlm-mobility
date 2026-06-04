import json
import torch
import os
import re
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH   = "Qwen/Qwen2.5-VL-7B-Instruct"
DRIVELM_JSON = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm/v1_1_train_nus.json"
IMAGE_ROOT   = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm"
OUTPUT_PATH  = "/home/xzh5180/Research/vlm-mobility/outputs/usecase10a_agent_pipeline_qwen25vl_drivelm.json"
MAX_NEW_TOKENS = 300

EVAL_SUBSET = {
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

SPEED_LABELS     = ["accelerate", "maintain", "decelerate", "stop"]
DIRECTION_LABELS = ["change lane left", "change lane right",
                    "turn left", "turn right", "go straight"]

# ─── AGENT PROMPTS ────────────────────────────────────────────────────────────

PERC_SYSTEM = (
    "You are a Perception Agent for an autonomous vehicle. "
    "Your only job: accurately describe what you see in the camera images."
)
PERC_PROMPT = (
    "Analyze the front camera image. Produce a structured report with exactly these fields:\n"
    "VEHICLES: [each vehicle with type, position relative to ego (front-center/front-left/front-right/"
    "rear-left/rear-right), estimated distance in meters]\n"
    "PEDESTRIANS: [any people or cyclists with position]\n"
    "TRAFFIC_CONTROLS: [signals, signs, road markings visible]\n"
    "ROAD_STRUCTURE: [number of lanes, junction type, special conditions]\n"
    "Be specific. Use concrete positions. Do not speculate beyond what is visible."
)

PRED_SYSTEM = (
    "You are a Prediction Agent for an autonomous vehicle. "
    "Use the perception report and the camera image to predict near-future scene evolution."
)
PRED_TEMPLATE = (
    "Perception report from the previous agent:\n"
    "{perception}\n\n"
    "Now look at the camera image. Predict what will happen in the next 3 seconds:\n"
    "CRITICAL_OBJECTS: [objects requiring ego vehicle response, referencing perception report]\n"
    "PREDICTED_MOVES: [what each critical object will likely do]\n"
    "RISK_LEVEL: [LOW / MEDIUM / HIGH — one-sentence reason]\n"
    "SCENARIO_TYPE: [e.g., car-following, intersection-approach, lane-merge, free-flow, pedestrian-crossing]"
)

PLAN_SYSTEM = (
    "You are a Planning Agent for an autonomous vehicle. "
    "Decide the immediate driving action based on the situation reports and camera image."
)
PLAN_TEMPLATE = (
    "Situation from upstream agents:\n"
    "PERCEPTION: {perception}\n"
    "PREDICTION: {prediction}\n\n"
    "Look at the front camera image and decide the driving action.\n"
    "Respond in EXACTLY this format (no extra text, no explanation outside the fields):\n"
    "SPEED: [ACCELERATE|MAINTAIN SPEED|DECELERATE|STOP]\n"
    "DIRECTION: [TURN LEFT|GO STRAIGHT|TURN RIGHT|CHANGE LANE LEFT|CHANGE LANE RIGHT]\n"
    "REASON: [one sentence referencing key information from the perception or prediction reports]"
)

SINGLE_SYSTEM = (
    "You are an autonomous vehicle decision system. "
    "Analyze the front camera image and decide the immediate driving action."
)
SINGLE_PROMPT = (
    "Look at the front camera image and decide the driving action.\n"
    "Respond in EXACTLY this format (no extra text):\n"
    "SPEED: [ACCELERATE|MAINTAIN SPEED|DECELERATE|STOP]\n"
    "DIRECTION: [TURN LEFT|GO STRAIGHT|TURN RIGHT|CHANGE LANE LEFT|CHANGE LANE RIGHT]\n"
    "REASON: [one sentence]"
)

# ─── MODEL ────────────────────────────────────────────────────────────────────

def load_model():
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, quantization_config=bnb_cfg, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return model, processor

def run_agent(model, processor, system_msg, user_msg, image_path):
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text",  "text": user_msg}
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

# ─── PARSING ──────────────────────────────────────────────────────────────────

def parse_speed_direction(text):
    text_lower = text.lower()
    speed, direction = "unknown", "unknown"

    m = re.search(r'speed\s*:\s*([^\n]+)', text_lower)
    if m:
        tok = m.group(1).strip()
        for lbl in SPEED_LABELS:
            if lbl in tok:
                speed = lbl
                break

    m = re.search(r'direction\s*:\s*([^\n]+)', text_lower)
    if m:
        tok = m.group(1).strip()
        for lbl in DIRECTION_LABELS:   # longer labels checked first
            if lbl in tok:
                direction = lbl
                break

    return speed, direction

def extract_gt_behavior(frame_data):
    gt_speed, gt_direction = "unknown", "unknown"
    for qa in frame_data.get("QA", {}).get("behavior", []):
        ans = qa.get("A", "").lower()          # fixed: key is A not answer
        if gt_speed == "unknown":
            for phrase, label in [
                ("not moving",    "stop"),
                ("not move",      "stop"),
                ("driving fast",  "maintain"),
                ("going fast",    "maintain"),
                ("driving slow",  "decelerate"),
                ("going slow",    "decelerate"),
                ("deceler",       "decelerate"),
                ("accelerat",     "accelerate"),
                ("same speed",    "maintain"),
                ("keep going",    "maintain"),
                ("stop",          "stop"),
                ("fast",          "maintain"),
                ("slow",          "decelerate"),
            ]:
                if phrase in ans:
                    gt_speed = label
                    break
        if gt_direction == "unknown":
            for phrase, label in [
                ("change lane left",  "change lane left"),
                ("change lane right", "change lane right"),
                ("turn left",         "turn left"),
                ("turn right",        "turn right"),
                ("going straight",    "go straight"),
                ("go straight",       "go straight"),
                ("straight",          "go straight"),
            ]:
                if phrase in ans:
                    gt_direction = label
                    break
    return gt_speed, gt_direction

def chain_coherence(perc_text, pred_text, plan_text):
    """
    Rough proxy: what fraction of content words in upstream text
    appears in downstream text?
    """
    def stopwords():
        return {"the","a","an","is","are","in","on","at","to","of","and","or",
                "with","for","this","that","it","its","be","as","by"}

    def content_words(t):
        return {w for w in re.findall(r'\b[a-z]{4,}\b', t.lower())
                if w not in stopwords()}

    perc_w = content_words(perc_text)
    pred_w = content_words(pred_text)
    plan_w = content_words(plan_text)

    def overlap(a, b):
        if not a:
            return 0.0
        return round(len(a & b) / len(a), 3)

    return {
        "perc_to_pred": overlap(perc_w, pred_w),
        "pred_to_plan": overlap(pred_w, plan_w)
    }

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading model...")
    model, processor = load_model()

    print("Loading DriveLM data...")
    with open(DRIVELM_JSON) as f:
        data = json.load(f)

    results = []
    for scene_id, frame_tokens in EVAL_SUBSET.items():
        key_frames = data.get(scene_id, {}).get("key_frames", {})

        for frame_token in frame_tokens:
            frame_data = key_frames.get(frame_token, {})
            if not frame_data:
                print(f"[SKIP] frame {frame_token[:8]} not found")
                continue

            front_raw = frame_data.get("image_paths", {}).get("CAM_FRONT", "")
            img_abs   = os.path.join(IMAGE_ROOT, front_raw.replace("../nuscenes/", "nuscenes/"))
            if not os.path.exists(img_abs):
                print(f"[SKIP] image missing: {img_abs}")
                continue

            gt_speed, gt_dir = extract_gt_behavior(frame_data)
            print(f"\n=== S:{scene_id[:8]} F:{frame_token[:8]} | GT speed={gt_speed} dir={gt_dir} ===")

            # ── Agent Chain ──────────────────────────────────────────────
            print("  [A1] Perception...")
            perc_out = run_agent(model, processor, PERC_SYSTEM, PERC_PROMPT, img_abs)

            print("  [A2] Prediction...")
            pred_out = run_agent(
                model, processor, PRED_SYSTEM,
                PRED_TEMPLATE.format(perception=perc_out[:600]),
                img_abs
            )

            print("  [A3] Planning...")
            plan_out = run_agent(
                model, processor, PLAN_SYSTEM,
                PLAN_TEMPLATE.format(
                    perception=perc_out[:350],
                    prediction=pred_out[:350]
                ),
                img_abs
            )
            chain_speed, chain_dir = parse_speed_direction(plan_out)

            # ── Single Agent Baseline ────────────────────────────────────
            print("  [Base] Single agent...")
            single_out   = run_agent(model, processor, SINGLE_SYSTEM, SINGLE_PROMPT, img_abs)
            single_speed, single_dir = parse_speed_direction(single_out)

            coherence = chain_coherence(perc_out, pred_out, plan_out)

            print(f"  Chain : speed={chain_speed} dir={chain_dir} "
                  f"| coherence perc→pred={coherence['perc_to_pred']} pred→plan={coherence['pred_to_plan']}")
            print(f"  Single: speed={single_speed} dir={single_dir}")

            results.append({
                "scene_id": scene_id,
                "frame_token": frame_token,
                "ground_truth": {"speed": gt_speed, "direction": gt_dir},
                "agent_chain": {
                    "perception_output":  perc_out,
                    "prediction_output":  pred_out,
                    "planning_output":    plan_out,
                    "parsed_speed":       chain_speed,
                    "parsed_direction":   chain_dir,
                    "speed_correct":      chain_speed == gt_speed,
                    "direction_correct":  chain_dir   == gt_dir,
                    "chain_coherence":    coherence
                },
                "single_agent": {
                    "output":             single_out,
                    "parsed_speed":       single_speed,
                    "parsed_direction":   single_dir,
                    "speed_correct":      single_speed == gt_speed,
                    "direction_correct":  single_dir   == gt_dir
                }
            })

    # ── Summary ───────────────────────────────────────────────────────────
    valid = [r for r in results if r["ground_truth"]["speed"] != "unknown"]
    n = len(valid)

    def acc(key_chain, key_field):
        return round(sum(r[key_chain][key_field] for r in valid) / n, 3) if n else 0.0

    def avg_coherence(coh_key):
        vals = [r["agent_chain"]["chain_coherence"][coh_key] for r in valid]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    summary = {
        "total_frames": len(results),
        "frames_with_gt": n,
        "agent_chain": {
            "speed_accuracy":     acc("agent_chain", "speed_correct"),
            "direction_accuracy": acc("agent_chain", "direction_correct"),
            "avg_perc_to_pred_coherence": avg_coherence("perc_to_pred"),
            "avg_pred_to_plan_coherence": avg_coherence("pred_to_plan")
        },
        "single_agent": {
            "speed_accuracy":     acc("single_agent", "speed_correct"),
            "direction_accuracy": acc("single_agent", "direction_correct")
        }
    }
    print("\n\n=== UC10a SUMMARY ===")
    print(json.dumps(summary, indent=2))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"summary": summary, "frames": results}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
