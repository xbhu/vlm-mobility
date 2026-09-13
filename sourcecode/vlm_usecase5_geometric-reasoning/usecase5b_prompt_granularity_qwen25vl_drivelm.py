# ============================================================
# UC5b: Prompt Granularity Effect on Geometric Reasoning
# Coarse-grained vs Fine-grained prompts for distance estimation
# and relative order tasks
# Model: Qwen2.5-VL-7B-Instruct (4-bit quantized)
# Dataset: DriveLM-nuScenes, fixed 9-frame evaluation subset
# ============================================================

import json
import os
import math
import re
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch

# ── CONFIG ──────────────────────────────────────────────────
DRIVELM_JSON   = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm/v1_1_train_nus.json"
NUSCENES_ROOT  = "/home/xzh5180/Research/vlm-mobility/datasets/nuscenes"
IMG_ROOT       = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm/nuscenes"
OUTPUT_PATH    = "/home/xzh5180/Research/vlm-mobility/outputs/usecase5b_prompt_granularity_qwen25vl_drivelm.json"
MODEL_NAME     = "Qwen/Qwen2.5-VL-7B-Instruct"

FRAME_TOKENS = [
    "4a0798f849ca477ab18009c3a20b7df2",
    "ffd1bdf020d145759224c629b501d2b2",
    "d9075c2a5f864a2b8abf41e703f4cf1c",
    "542eaf1fc9b34895a9e55fab57cb4cf4",
    "1b45a97a0e5e49fe9cd345dd4bd729c3",
    "d5e16062410f4e329d31a881b28e5c1c",
    "bd8a5e326b804b069d497d29dbf19c2b",
    "7903e67446c64958b0a660f10bdadf19",
    "b6bf5a2bcb094969ace1023f8fe0b9e2",
]

PIXEL_MATCH_THRESHOLD = 100

# ── NUSCENES HELPERS (same as UC5a) ──────────────────────────
def get_ego_position(nusc, sample_token):
    sample = nusc.get("sample", sample_token)
    cam_token = sample["data"]["CAM_FRONT"]
    cam_data = nusc.get("sample_data", cam_token)
    ep = nusc.get("ego_pose", cam_data["ego_pose_token"])
    return np.array(ep["translation"][:2])

def get_ego_heading(nusc, sample_token):
    sample = nusc.get("sample", sample_token)
    cam_token = sample["data"]["CAM_FRONT"]
    cam_data = nusc.get("sample_data", cam_token)
    ep = nusc.get("ego_pose", cam_data["ego_pose_token"])
    q = Quaternion(ep["rotation"])
    return q.yaw_pitch_roll[0]

def project_annotation_to_camera(nusc, ann_token, sample_token, cam_name):
    sample = nusc.get("sample", sample_token)
    cam_token = sample["data"][cam_name]
    cam_data = nusc.get("sample_data", cam_token)
    cs = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
    ep = nusc.get("ego_pose", cam_data["ego_pose_token"])
    ann = nusc.get("sample_annotation", ann_token)
    center_global = np.array(ann["translation"])
    ego_rot = Quaternion(ep["rotation"]).rotation_matrix
    ego_trans = np.array(ep["translation"])
    center_ego = ego_rot.T @ (center_global - ego_trans)
    cam_rot = Quaternion(cs["rotation"]).rotation_matrix
    cam_trans = np.array(cs["translation"])
    center_cam = cam_rot.T @ (center_ego - cam_trans)
    if center_cam[2] <= 0:
        return None
    K = np.array(cs["camera_intrinsic"])
    uv = K @ center_cam
    u = uv[0] / uv[2]
    v = uv[1] / uv[2]
    return (u, v)

def match_object_to_annotation(nusc, sample_token, cam_name, pixel_x, pixel_y):
    sample = nusc.get("sample", sample_token)
    best = None
    best_dist = float("inf")
    for ann_token in sample["anns"]:
        proj = project_annotation_to_camera(nusc, ann_token, sample_token, cam_name)
        if proj is None:
            continue
        u, v = proj
        d = math.sqrt((u - pixel_x) ** 2 + (v - pixel_y) ** 2)
        if d < best_dist:
            best_dist = d
            best = ann_token
    if best is None or best_dist > PIXEL_MATCH_THRESHOLD:
        return None
    ann = nusc.get("sample_annotation", best)
    return best, best_dist, ann["category_name"]

def compute_gt_distance(nusc, ann_token, sample_token):
    ego_xy = get_ego_position(nusc, sample_token)
    ann = nusc.get("sample_annotation", ann_token)
    obj_xy = np.array(ann["translation"][:2])
    return float(np.linalg.norm(obj_xy - ego_xy))

# ── MODEL LOADING ────────────────────────────────────────────
def load_model():
    print("Loading Qwen2.5-VL-7B-Instruct (4-bit)...")
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

# ── INFERENCE ────────────────────────────────────────────────
def run_inference(model, processor, image_path, question):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text",  "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256)
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()

# ── PARSE HELPERS ────────────────────────────────────────────
def parse_first_number(text):
    matches = re.findall(r"\d+\.?\d*", text)
    if matches:
        return float(matches[0])
    return None

def parse_all_numbers(text):
    return [float(x) for x in re.findall(r"\d+\.?\d*", text)]

def parse_closer_from_coarse(response, desc_a, desc_b):
    """
    Try to determine which vehicle the model thinks is closer
    from a free-form coarse response.
    Looks for the description strings or A/B labels.
    """
    resp = response.lower()
    has_a = desc_a.lower()[:10] in resp
    has_b = desc_b.lower()[:10] in resp

    # explicit A/B
    if "closer" in resp:
        after = resp[resp.index("closer"):]
        if desc_a.lower()[:8] in after:
            return "A"
        if desc_b.lower()[:8] in after:
            return "B"

    # fallback: first number mentioned — assign to whichever desc appears first
    nums = parse_all_numbers(response)
    if len(nums) >= 2:
        if nums[0] <= nums[1]:
            return "A" if has_a else None
        else:
            return "B" if has_b else None
    return None

# ── MAIN ─────────────────────────────────────────────────────
def main():
    print("Loading nuScenes...")
    nusc = NuScenes(version="v1.0-trainval", dataroot=NUSCENES_ROOT, verbose=False)

    print("Loading DriveLM...")
    with open(DRIVELM_JSON) as f:
        drivelm = json.load(f)

    frame_index = {}
    for scene_token, scene in drivelm.items():
        for ft, fd in scene["key_frames"].items():
            if ft in FRAME_TOKENS:
                frame_index[ft] = fd

    model, processor = load_model()

    results = {
        "distance_estimation": [],   # coarse vs fine per object
        "relative_order": [],        # coarse vs fine per pair
    }

    for frame_token in FRAME_TOKENS:
        frame_data = frame_index[frame_token]
        key_objects = frame_data.get("key_object_infos", {})
        image_paths = frame_data["image_paths"]

        front_img = image_paths.get("CAM_FRONT", "").replace("../nuscenes", IMG_ROOT)

        print(f"\n=== Frame {frame_token[:8]} | objects={len(key_objects)} ===")

        # build matched objects (same logic as UC5a)
        matched_objects = []
        for obj_key, obj_info in key_objects.items():
            if obj_info["Category"] != "Vehicle":
                continue
            parts = obj_key.strip("<>").split(",")
            if len(parts) != 4:
                continue
            cam_name = parts[1]
            px, py = float(parts[2]), float(parts[3])
            match = match_object_to_annotation(nusc, frame_token, cam_name, px, py)
            if match is None:
                continue
            ann_token, pixel_dist, cat_name = match
            gt_distance = compute_gt_distance(nusc, ann_token, frame_token)
            vis_desc = obj_info.get("Visual_description", "a vehicle")
            cam_img = image_paths.get(cam_name, "").replace("../nuscenes", IMG_ROOT)

            matched_objects.append({
                "obj_key":     obj_key,
                "cam_name":    cam_name,
                "image_path":  cam_img,
                "vis_desc":    vis_desc,
                "ann_token":   ann_token,
                "gt_distance": gt_distance,
            })
            print(f"  matched {obj_key} → {cat_name} | gt_dist={gt_distance:.1f}m")

        # ── TASK 1: Distance Estimation — Coarse vs Fine ─────
        for obj in matched_objects:
            gt_dist  = obj["gt_distance"]
            img_path = obj["image_path"]
            vis_desc = obj["vis_desc"]

            # COARSE prompt
            q_coarse = ("You are a driver. Looking at this image, "
                        "what vehicles can you see and approximately how far away are they? "
                        "Give distances in meters.")
            resp_coarse = run_inference(model, processor, img_path, q_coarse)
            pred_coarse = parse_first_number(resp_coarse)
            mae_coarse  = abs(pred_coarse - gt_dist) if pred_coarse is not None else None
            pct_coarse  = (mae_coarse / gt_dist * 100) if (mae_coarse is not None and gt_dist > 0) else None

            # FINE prompt
            q_fine = (f"You are a driver. Looking at this image, "
                      f"approximately how far away (in meters) is the {vis_desc} "
                      f"from your vehicle? Please give a single number.")
            resp_fine = run_inference(model, processor, img_path, q_fine)
            pred_fine = parse_first_number(resp_fine)
            mae_fine  = abs(pred_fine - gt_dist) if pred_fine is not None else None
            pct_fine  = (mae_fine / gt_dist * 100) if (mae_fine is not None and gt_dist > 0) else None

            mae_c_str = f"{mae_coarse:.1f}" if mae_coarse is not None else "N/A"
            mae_f_str = f"{mae_fine:.1f}"   if mae_fine   is not None else "N/A"
            print(f"  [DIST] {vis_desc[:25]} gt={gt_dist:.1f}m | "
                  f"coarse_mae={mae_c_str} fine_mae={mae_f_str}")

            results["distance_estimation"].append({
                "frame_token":    frame_token,
                "obj_key":        obj["obj_key"],
                "vis_desc":       vis_desc,
                "gt_distance":    gt_dist,
                "coarse": {
                    "prompt":     q_coarse,
                    "response":   resp_coarse,
                    "pred":       pred_coarse,
                    "mae":        mae_coarse,
                    "pct_error":  pct_coarse,
                },
                "fine": {
                    "prompt":     q_fine,
                    "response":   resp_fine,
                    "pred":       pred_fine,
                    "mae":        mae_fine,
                    "pct_error":  pct_fine,
                },
            })

        # ── TASK 2: Relative Order — Coarse vs Fine ──────────
        if len(matched_objects) >= 2:
            for i in range(len(matched_objects) - 1):
                for j in range(i + 1, len(matched_objects)):
                    obj_a = matched_objects[i]
                    obj_b = matched_objects[j]
                    gt_closer = "A" if obj_a["gt_distance"] <= obj_b["gt_distance"] else "B"
                    gt_diff   = abs(obj_a["gt_distance"] - obj_b["gt_distance"])

                    # COARSE prompt
                    q_coarse = ("You are a driver. Looking at this image, "
                                "which vehicle in the scene appears closest to you? "
                                "Describe which one and roughly how far away it is.")
                    resp_coarse = run_inference(model, processor, front_img, q_coarse)
                    pred_coarse = parse_closer_from_coarse(
                        resp_coarse, obj_a["vis_desc"], obj_b["vis_desc"]
                    )
                    correct_coarse = (pred_coarse == gt_closer) if pred_coarse else None

                    # FINE prompt
                    q_fine = (f"You are a driver. Looking at this image, "
                              f"which vehicle is closer to you: "
                              f"(A) the {obj_a['vis_desc']} or (B) the {obj_b['vis_desc']}? "
                              f"Answer with just A or B.")
                    resp_fine = run_inference(model, processor, front_img, q_fine)
                    pred_fine = "A" if "A" in resp_fine.upper() and "B" not in resp_fine.upper() else \
                                "B" if "B" in resp_fine.upper() else None
                    correct_fine = (pred_fine == gt_closer) if pred_fine else None

                    print(f"  [REL] A={obj_a['vis_desc'][:18]}({obj_a['gt_distance']:.1f}m) "
                          f"B={obj_b['vis_desc'][:18]}({obj_b['gt_distance']:.1f}m) "
                          f"gt={gt_closer} | coarse={pred_coarse}({correct_coarse}) fine={pred_fine}({correct_fine})")

                    results["relative_order"].append({
                        "frame_token": frame_token,
                        "obj_a":       obj_a["vis_desc"],
                        "obj_b":       obj_b["vis_desc"],
                        "dist_a":      obj_a["gt_distance"],
                        "dist_b":      obj_b["gt_distance"],
                        "gt_diff_m":   gt_diff,
                        "gt_closer":   gt_closer,
                        "coarse": {
                            "prompt":   q_coarse,
                            "response": resp_coarse,
                            "pred":     pred_coarse,
                            "correct":  correct_coarse,
                        },
                        "fine": {
                            "prompt":   q_fine,
                            "response": resp_fine,
                            "pred":     pred_fine,
                            "correct":  correct_fine,
                        },
                    })

    # ── AGGREGATE METRICS ────────────────────────────────────
    print("\n" + "="*60)
    print("AGGREGATE RESULTS — Coarse vs Fine Prompt")
    print("="*60)

    de = results["distance_estimation"]
    if de:
        c_maes = [r["coarse"]["mae"] for r in de if r["coarse"]["mae"] is not None]
        f_maes = [r["fine"]["mae"]   for r in de if r["fine"]["mae"]   is not None]
        c_pcts = [r["coarse"]["pct_error"] for r in de if r["coarse"]["pct_error"] is not None]
        f_pcts = [r["fine"]["pct_error"]   for r in de if r["fine"]["pct_error"]   is not None]
        print(f"\nDistance Estimation (n={len(de)})")
        print(f"  Coarse — MAE: {np.mean(c_maes):.2f}m  |  Mean Pct Err: {np.mean(c_pcts):.1f}%")
        print(f"  Fine   — MAE: {np.mean(f_maes):.2f}m  |  Mean Pct Err: {np.mean(f_pcts):.1f}%")
        delta_mae = np.mean(f_maes) - np.mean(c_maes)
        print(f"  Delta (fine - coarse): {delta_mae:+.2f}m  "
              f"({'fine better' if delta_mae < 0 else 'coarse better or no gain'})")

    ro = results["relative_order"]
    if ro:
        c_valid = [r for r in ro if r["coarse"]["correct"] is not None]
        f_valid = [r for r in ro if r["fine"]["correct"]   is not None]
        c_acc = sum(r["coarse"]["correct"] for r in c_valid) / len(c_valid) * 100 if c_valid else 0
        f_acc = sum(r["fine"]["correct"]   for r in f_valid) / len(f_valid) * 100 if f_valid else 0
        print(f"\nRelative Order (n={len(ro)})")
        print(f"  Coarse — Accuracy: {c_acc:.1f}% ({len(c_valid)} parseable)")
        print(f"  Fine   — Accuracy: {f_acc:.1f}% ({len(f_valid)} parseable)")
        print(f"  Delta (fine - coarse): {f_acc - c_acc:+.1f}%")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
