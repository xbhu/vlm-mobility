"""
UC6a: Automatic Annotation Generation
VLM generates scene descriptions, object labels, and safety QA pairs.
Evaluated against DriveLM GT using ROUGE and object label F1.
Prompt strategy: zero-shot structured template.
"""

import json
import re
import os
import torch
import numpy as np
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from rouge_score import rouge_scorer

# ── CONFIG ───────────────────────────────────────────────────────────────────
DRIVELM_JSON   = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm/v1_1_train_nus.json"
IMAGE_ROOT     = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm"
OUTPUT_PATH    = "/home/xzh5180/Research/vlm-mobility/outputs/usecase6a_annotation_generation_qwen25vl_drivelm.json"
MODEL_NAME     = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_NEW_TOKENS = 512
DEVICE         = "cuda"

FIXED_SCENES = {
    "f0f120e4": ["4a0798f8", "ffd1bdf0", "d9075c2a"],
    "54cdaaae": ["542eaf1f", "1b45a97a", "d5e16062"],
    "1977a1c9": ["bd8a5e32", "7903e674", "b6bf5a2b"],
}

CANONICAL_OBJECTS = [
    "car", "truck", "bus", "motorcycle", "bicycle", "pedestrian",
    "traffic cone", "barrier", "traffic light", "stop sign"
]
# ─────────────────────────────────────────────────────────────────────────────


def load_model():
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
    print("Model loaded.")
    return model, processor


def load_drivelm(path):
    with open(path) as f:
        data = json.load(f)
    print(f"DriveLM loaded: {len(data)} scenes")
    return data


def resolve_image_path(raw_path):
    """
    ../nuscenes/samples/CAM_FRONT/xxx.jpg
    -> /home/.../datasets/drivelm/nuscenes/samples/CAM_FRONT/xxx.jpg
    """
    p = str(raw_path).replace("../nuscenes/", "nuscenes/", 1)
    return os.path.join(IMAGE_ROOT, p)


def get_image_paths(frame_data):
    img_dict = frame_data.get("image_paths", {})
    if not img_dict:
        print("  [WARN] No image_paths field in frame")
        return []

    resolved = []
    for cam, raw_p in img_dict.items():
        p = resolve_image_path(raw_p)
        if os.path.exists(p):
            resolved.append(p)
        else:
            print(f"  [WARN] Missing image: {p}")
    return resolved


def build_annotation_prompt(image_paths):
    content = []
    for p in image_paths[:6]:
        content.append({"type": "image", "image": f"file://{p}"})
    content.append({
        "type": "text",
        "text": (
            "You are an autonomous driving annotation assistant. "
            "Analyze the camera image(s) and generate structured annotations.\n\n"
            "Respond using EXACTLY this format — no extra text outside the sections:\n\n"
            "DESCRIPTION: <one sentence describing the overall driving scene>\n\n"
            "OBJECTS: <comma-separated list of object categories present, chosen only from: "
            "car, truck, bus, motorcycle, bicycle, pedestrian, traffic cone, barrier, "
            "traffic light, stop sign>\n\n"
            "SAFETY_QUESTION: <one safety-relevant question about this scene>\n"
            "SAFETY_ANSWER: <answer to the question above>"
        )
    })
    return content


def run_inference(model, processor, image_paths):
    content  = build_annotation_prompt(image_paths)
    messages = [{"role": "user", "content": content}]
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def parse_vlm_output(raw_text):
    result = {
        "description":    "",
        "objects":        [],
        "safety_question": "",
        "safety_answer":  "",
        "raw_output":     raw_text,
        "parse_success":  {"description": False, "objects": False, "safety_qa": False},
    }

    m = re.search(r"DESCRIPTION:\s*(.+?)(?=\n\nOBJECTS:|\nOBJECTS:|\Z)", raw_text, re.DOTALL)
    if m:
        result["description"] = m.group(1).strip()
        result["parse_success"]["description"] = bool(result["description"])

    m = re.search(r"OBJECTS:\s*(.+?)(?=\n\nSAFETY_QUESTION:|\nSAFETY_QUESTION:|\Z)", raw_text, re.DOTALL)
    if m:
        candidates = [o.strip().lower() for o in m.group(1).strip().split(",")]
        valid = [o for o in candidates if o in CANONICAL_OBJECTS]
        result["objects"]     = valid
        result["objects_raw"] = candidates
        result["parse_success"]["objects"] = len(valid) > 0

    mq = re.search(r"SAFETY_QUESTION:\s*(.+?)(?=\nSAFETY_ANSWER:|\Z)", raw_text, re.DOTALL)
    ma = re.search(r"SAFETY_ANSWER:\s*(.+?)$", raw_text, re.DOTALL)
    if mq:
        result["safety_question"] = mq.group(1).strip()
    if ma:
        result["safety_answer"] = ma.group(1).strip()
    result["parse_success"]["safety_qa"] = bool(
        result["safety_question"] and result["safety_answer"]
    )
    return result


def extract_gt(frame_data):
    gt = {"description": "", "objects": [], "safety_question": "", "safety_answer": ""}

    qa_sections = frame_data.get("QA", {})
    all_qa = []
    if isinstance(qa_sections, dict):
        for section, qa_list in qa_sections.items():
            if isinstance(qa_list, list):
                for item in qa_list:
                    if isinstance(item, dict):
                        all_qa.append({
                            "Q": item.get("Q", item.get("question", "")),
                            "A": item.get("A", item.get("answer", "")),
                            "section": section,
                        })

    # Description: first perception answer as proxy
    if not gt["description"] and all_qa:
        gt["description"] = all_qa[0]["A"]

    # Safety QA: keyword search, then planning, then first QA
    safety_kw = ["safe", "risk", "danger", "caution", "warning", "hazard", "collision"]
    safety_qa = None
    for item in all_qa:
        if any(kw in item["Q"].lower() for kw in safety_kw):
            safety_qa = item
            break
    if not safety_qa:
        for item in all_qa:
            if item["section"] == "planning":
                safety_qa = item
                break
    if not safety_qa and all_qa:
        safety_qa = all_qa[0]
    if safety_qa:
        gt["safety_question"] = safety_qa["Q"]
        gt["safety_answer"]   = safety_qa["A"]

    # Objects: scan perception QA text
    for item in all_qa:
        if item["section"] == "perception":
            text = (item["Q"] + " " + item["A"]).lower()
            for obj in CANONICAL_OBJECTS:
                if obj in text and obj not in gt["objects"]:
                    gt["objects"].append(obj)

    gt["all_qa_count"] = len(all_qa)
    return gt


def compute_rouge(pred, ref):
    if not pred or not ref:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s  = sc.score(ref, pred)
    return {k: round(s[k].fmeasure, 4) for k in ["rouge1", "rouge2", "rougeL"]}


def compute_object_metrics(pred, gt):
    ps, gs = set(pred), set(gt)
    if not gs and not ps:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": [], "fp": [], "fn": []}
    if not ps:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "tp": [], "fp": [], "fn": sorted(gs)}
    if not gs:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0,
                "tp": [], "fp": sorted(ps), "fn": []}
    tp = len(ps & gs)
    p  = tp / len(ps)
    r  = tp / len(gs)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {
        "precision": round(p,  4), "recall": round(r,  4), "f1": round(f1, 4),
        "pred_set": sorted(ps), "gt_set": sorted(gs),
        "tp": sorted(ps & gs), "fp": sorted(ps - gs), "fn": sorted(gs - ps),
    }


def aggregate(results):
    def mean(lst): return round(float(np.mean(lst)), 4) if lst else 0.0
    return {
        "n_frames": len(results),
        "parsability": {
            k: mean([int(r["parsed"]["parse_success"][k]) for r in results])
            for k in ["description", "objects", "safety_qa"]
        },
        "description_rouge": {
            k: mean([r["metrics"]["description_rouge"][k] for r in results])
            for k in ["rouge1", "rougeL"]
        },
        "safety_qa_rouge": {
            k: mean([r["metrics"]["safety_qa_rouge"][k] for r in results])
            for k in ["rouge1", "rougeL"]
        },
        "object_labels": {
            k: mean([r["metrics"]["object_labels"][k] for r in results])
            for k in ["precision", "recall", "f1"]
        },
    }


def main():
    print("=== UC6a: Automatic Annotation Generation ===\n")
    model, processor = load_model()
    drivelm = load_drivelm(DRIVELM_JSON)
    results = []

    for scene_token, frame_tokens in FIXED_SCENES.items():
        # Find scene
        scene_data = next((v for k, v in drivelm.items() if scene_token in k), None)
        if scene_data is None:
            print(f"[WARN] Scene {scene_token} not found"); continue

        key_frames = scene_data.get("key_frames", scene_data.get("frames", {}))
        if not isinstance(key_frames, dict):
            print(f"[WARN] Unexpected key_frames type: {type(key_frames)}"); continue

        for frame_token in frame_tokens:
            frame_data = next((v for k, v in key_frames.items() if frame_token in k), None)
            if frame_data is None:
                print(f"[WARN] Frame {frame_token} not found"); continue

            print(f"\nProcessing scene={scene_token[:8]} frame={frame_token[:8]}")
            image_paths = get_image_paths(frame_data)
            if not image_paths:
                print("  [SKIP] No images"); continue
            print(f"  Images: {len(image_paths)} cameras")

            raw_output = run_inference(model, processor, image_paths)
            print(f"  Output preview: {raw_output[:100].replace(chr(10),' ')}")

            parsed      = parse_vlm_output(raw_output)
            gt          = extract_gt(frame_data)
            desc_rouge  = compute_rouge(parsed["description"], gt["description"])
            qa_rouge    = compute_rouge(parsed["safety_answer"], gt["safety_answer"])
            obj_metrics = compute_object_metrics(parsed["objects"], gt["objects"])

            print(f"  Parse: {parsed['parse_success']}")
            print(f"  Objs pred={parsed['objects']}  gt={gt['objects']}")
            print(f"  Desc RL={desc_rouge['rougeL']}  QA RL={qa_rouge['rougeL']}  Obj F1={obj_metrics['f1']}")

            results.append({
                "scene_token": scene_token, "frame_token": frame_token,
                "image_paths": image_paths, "parsed": parsed, "gt": gt,
                "metrics": {
                    "description_rouge": desc_rouge,
                    "safety_qa_rouge":   qa_rouge,
                    "object_labels":     obj_metrics,
                },
            })

    agg = aggregate(results)
    output = {
        "experiment": "UC6a", "prompt_strategy": "zero_shot_structured",
        "n_frames": len(results), "aggregate_metrics": agg,
        "per_frame_results": results,
    }
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*55)
    print("UC6a RESULTS SUMMARY")
    print("="*55)
    print(f"\nParsability:")
    for k, v in agg["parsability"].items():
        print(f"  {k:<20}: {v:.3f}")
    print(f"\nDescription ROUGE:  R1={agg['description_rouge']['rouge1']:.4f}  RL={agg['description_rouge']['rougeL']:.4f}")
    print(f"Safety QA ROUGE:    R1={agg['safety_qa_rouge']['rouge1']:.4f}  RL={agg['safety_qa_rouge']['rougeL']:.4f}")
    print(f"Object labels:      P={agg['object_labels']['precision']:.4f}  R={agg['object_labels']['recall']:.4f}  F1={agg['object_labels']['f1']:.4f}")
    print(f"\nOutput: {OUTPUT_PATH}")

    print("\n" + "="*55)
    print("SAMPLE-LEVEL INSPECTION (first 3 frames)")
    print("="*55)
    for r in results[:3]:
        print(f"\n[{r['scene_token'][:8]} / {r['frame_token'][:8]}]")
        print(f"  VLM desc : {r['parsed']['description'][:120]}")
        print(f"  GT  desc : {r['gt']['description'][:120]}")
        print(f"  VLM objs : {r['parsed']['objects']}")
        print(f"  GT  objs : {r['gt']['objects']}")
        print(f"  VLM Q    : {r['parsed']['safety_question'][:100]}")
        print(f"  VLM A    : {r['parsed']['safety_answer'][:100]}")
        print(f"  GT  A    : {r['gt']['safety_answer'][:100]}")
        m = r["metrics"]["object_labels"]
        print(f"  TP/FP/FN : {m.get('tp',[])} / {m.get('fp',[])} / {m.get('fn',[])}")


if __name__ == "__main__":
    main()
