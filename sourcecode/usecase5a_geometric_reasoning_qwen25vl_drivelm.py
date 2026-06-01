# ============================================================
# UC5a: Geometric Reasoning — Distance, Bearing, Relative Order
# Model: Qwen2.5-VL-7B-Instruct (4-bit quantized)
# Dataset: DriveLM-nuScenes, fixed 9-frame evaluation subset
# ============================================================

import json
import os
import math
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
OUTPUT_PATH    = "/home/xzh5180/Research/vlm-mobility/outputs/usecase5a_geometric_reasoning_qwen25vl_drivelm.json"
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

PIXEL_MATCH_THRESHOLD = 100  # pixels, for 2D→3D annotation matching

# ── NUSCENES HELPERS ─────────────────────────────────────────
def get_ego_position(nusc, sample_token):
    """Get ego vehicle (x, y) from CAM_FRONT ego_pose."""
    sample = nusc.get("sample", sample_token)
    cam_token = sample["data"]["CAM_FRONT"]
    cam_data = nusc.get("sample_data", cam_token)
    ep = nusc.get("ego_pose", cam_data["ego_pose_token"])
    return np.array(ep["translation"][:2])  # (x, y)

def get_ego_heading(nusc, sample_token):
    """Get ego heading (yaw) in radians from CAM_FRONT ego_pose."""
    sample = nusc.get("sample", sample_token)
    cam_token = sample["data"]["CAM_FRONT"]
    cam_data = nusc.get("sample_data", cam_token)
    ep = nusc.get("ego_pose", cam_data["ego_pose_token"])
    q = Quaternion(ep["rotation"])
    # yaw from quaternion
    yaw = q.yaw_pitch_roll[0]
    return yaw

def project_annotation_to_camera(nusc, ann_token, sample_token, cam_name):
    """
    Project a 3D annotation center into camera image coordinates.
    Returns (u, v) pixel or None if behind camera / out of frame.
    """
    sample = nusc.get("sample", sample_token)
    cam_token = sample["data"][cam_name]
    cam_data = nusc.get("sample_data", cam_token)
    cs = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
    ep = nusc.get("ego_pose", cam_data["ego_pose_token"])

    ann = nusc.get("sample_annotation", ann_token)
    # 3D center in global frame
    center_global = np.array(ann["translation"])

    # global → ego frame
    ego_rot = Quaternion(ep["rotation"]).rotation_matrix
    ego_trans = np.array(ep["translation"])
    center_ego = ego_rot.T @ (center_global - ego_trans)

    # ego → camera frame
    cam_rot = Quaternion(cs["rotation"]).rotation_matrix
    cam_trans = np.array(cs["translation"])
    center_cam = cam_rot.T @ (center_ego - cam_trans)

    # behind camera
    if center_cam[2] <= 0:
        return None

    # project with intrinsic
    K = np.array(cs["camera_intrinsic"])
    uv = K @ center_cam
    u = uv[0] / uv[2]
    v = uv[1] / uv[2]
    return (u, v)

def match_object_to_annotation(nusc, sample_token, cam_name, pixel_x, pixel_y):
    """
    Find the nuScenes annotation closest to (pixel_x, pixel_y) in cam_name.
    Returns (ann_token, distance_2d, category_name) or None.
    """
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
    """Euclidean distance (horizontal) from ego to annotation center."""
    ego_xy = get_ego_position(nusc, sample_token)
    ann = nusc.get("sample_annotation", ann_token)
    obj_xy = np.array(ann["translation"][:2])
    return float(np.linalg.norm(obj_xy - ego_xy))

def compute_gt_bearing(nusc, ann_token, sample_token):
    """
    Bearing of annotation relative to ego heading.
    Returns one of: front, back, left, right,
                    front-left, front-right, back-left, back-right
    """
    ego_xy  = get_ego_position(nusc, sample_token)
    ego_yaw = get_ego_heading(nusc, sample_token)
    ann = nusc.get("sample_annotation", ann_token)
    obj_xy = np.array(ann["translation"][:2])

    dx, dy = obj_xy - ego_xy
    # angle in global frame
    angle_global = math.atan2(dy, dx)
    # relative to ego heading
    rel_angle = angle_global - ego_yaw
    # normalise to [-pi, pi]
    rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi
    deg = math.degrees(rel_angle)

    # 8-sector classification
    if   -22.5 <= deg <  22.5:  return "front"
    elif  22.5 <= deg <  67.5:  return "front-left"
    elif  67.5 <= deg < 112.5:  return "left"
    elif 112.5 <= deg < 157.5:  return "back-left"
    elif deg >=  157.5 or deg < -157.5: return "back"
    elif -157.5 <= deg < -112.5: return "back-right"
    elif -112.5 <= deg <  -67.5: return "right"
    else:                        return "front-right"

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
        output_ids = model.generate(**inputs, max_new_tokens=128)
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()

# ── EVALUATION HELPERS ───────────────────────────────────────
def parse_distance_from_response(response):
    """Extract first number (float) from model response."""
    import re
    matches = re.findall(r"\d+\.?\d*", response)
    if matches:
        return float(matches[0])
    return None

def parse_bearing_from_response(response):
    """Map model response to one of the 8 bearing labels."""
    resp = response.lower()
    if   "front-left"  in resp or "front left"  in resp: return "front-left"
    elif "front-right" in resp or "front right" in resp: return "front-right"
    elif "back-left"   in resp or "back left"   in resp or "rear-left"  in resp: return "back-left"
    elif "back-right"  in resp or "back right"  in resp or "rear-right" in resp: return "back-right"
    elif "front"       in resp or "ahead"        in resp or "forward"   in resp: return "front"
    elif "back"        in resp or "behind"       in resp or "rear"      in resp: return "back"
    elif "left"        in resp:                                                   return "left"
    elif "right"       in resp:                                                   return "right"
    return None

# ── MAIN ─────────────────────────────────────────────────────
def main():
    print("Loading nuScenes...")
    nusc = NuScenes(version="v1.0-trainval", dataroot=NUSCENES_ROOT, verbose=False)

    print("Loading DriveLM...")
    with open(DRIVELM_JSON) as f:
        drivelm = json.load(f)

    # build frame_token → frame_data index
    frame_index = {}
    for scene_token, scene in drivelm.items():
        for ft, fd in scene["key_frames"].items():
            if ft in FRAME_TOKENS:
                frame_index[ft] = fd

    model, processor = load_model()

    results = {
        "distance_estimation": [],
        "bearing_classification": [],
        "relative_order": [],
    }

    for frame_token in FRAME_TOKENS:
        frame_data = frame_index[frame_token]
        key_objects = frame_data.get("key_object_infos", {})
        image_paths = frame_data["image_paths"]

        print(f"\n=== Frame {frame_token[:8]} | objects={len(key_objects)} ===")

        # resolve CAM_FRONT image path
        front_img_rel = image_paths.get("CAM_FRONT", "")
        front_img = front_img_rel.replace("../nuscenes", IMG_ROOT)

        # collect matched objects with 3D GT
        matched_objects = []
        for obj_key, obj_info in key_objects.items():
            if obj_info["Category"] != "Vehicle":
                continue
            # parse key: <cN,CAM_XX,px,py>
            parts = obj_key.strip("<>").split(",")
            if len(parts) != 4:
                continue
            cam_name = parts[1]
            px, py = float(parts[2]), float(parts[3])

            match = match_object_to_annotation(nusc, frame_token, cam_name, px, py)
            if match is None:
                print(f"  [{obj_key}] no annotation match within threshold")
                continue
            ann_token, pixel_dist, cat_name = match

            gt_distance = compute_gt_distance(nusc, ann_token, frame_token)
            gt_bearing  = compute_gt_bearing(nusc, ann_token, frame_token)
            vis_desc    = obj_info.get("Visual_description", "a vehicle")

            cam_img_rel = image_paths.get(cam_name, front_img_rel)
            cam_img = cam_img_rel.replace("../nuscenes", IMG_ROOT)

            matched_objects.append({
                "obj_key":     obj_key,
                "cam_name":    cam_name,
                "image_path":  cam_img,
                "vis_desc":    vis_desc,
                "ann_token":   ann_token,
                "gt_distance": gt_distance,
                "gt_bearing":  gt_bearing,
                "pixel_dist":  pixel_dist,
            })
            print(f"  matched {obj_key} → {cat_name} | gt_dist={gt_distance:.1f}m | gt_bearing={gt_bearing}")

        # ── TASK 1: Distance Estimation ──────────────────────
        for obj in matched_objects:
            q = (f"You are a driver. Looking at this image, "
                 f"approximately how far away (in meters) is the {obj['vis_desc']} "
                 f"from your vehicle? Please give a single number.")
            response = run_inference(model, processor, obj["image_path"], q)
            pred_dist = parse_distance_from_response(response)
            gt_dist   = obj["gt_distance"]
            mae = abs(pred_dist - gt_dist) if pred_dist is not None else None
            pct_err = (mae / gt_dist * 100) if (mae is not None and gt_dist > 0) else None
            mae_str = f"{mae:.1f}" if mae is not None else "N/A"
            print(f"  [DIST] {obj['vis_desc'][:30]} | gt={gt_dist:.1f}m pred={pred_dist} mae={mae_str}")
            results["distance_estimation"].append({
                "frame_token":  frame_token,
                "obj_key":      obj["obj_key"],
                "vis_desc":     obj["vis_desc"],
                "gt_distance":  gt_dist,
                "pred_distance": pred_dist,
                "mae":          mae,
                "pct_error":    pct_err,
                "response":     response,
            })

        # ── TASK 2: Bearing Classification ──────────────────
        for obj in matched_objects:
            q = (f"You are a driver. Looking at this image, "
                 f"in which direction is the {obj['vis_desc']} relative to your vehicle? "
                 f"Choose from: front, back, left, right, front-left, front-right, back-left, back-right.")
            response = run_inference(model, processor, obj["image_path"], q)
            pred_bearing = parse_bearing_from_response(response)
            gt_bearing   = obj["gt_bearing"]
            correct = (pred_bearing == gt_bearing)

            print(f"  [BEAR] {obj['vis_desc'][:30]} | gt={gt_bearing} pred={pred_bearing} correct={correct}")
            results["bearing_classification"].append({
                "frame_token":   frame_token,
                "obj_key":       obj["obj_key"],
                "vis_desc":      obj["vis_desc"],
                "gt_bearing":    gt_bearing,
                "pred_bearing":  pred_bearing,
                "correct":       correct,
                "response":      response,
            })

        # ── TASK 3: Relative Order ───────────────────────────
        if len(matched_objects) >= 2:
            for i in range(len(matched_objects) - 1):
                for j in range(i + 1, len(matched_objects)):
                    obj_a = matched_objects[i]
                    obj_b = matched_objects[j]
                    gt_closer = "A" if obj_a["gt_distance"] <= obj_b["gt_distance"] else "B"
                    gt_diff   = abs(obj_a["gt_distance"] - obj_b["gt_distance"])

                    # use CAM_FRONT image for both (both visible in front view ideally)
                    q = (f"You are a driver. Looking at this image, "
                         f"which vehicle is closer to you: "
                         f"(A) the {obj_a['vis_desc']} or (B) the {obj_b['vis_desc']}? "
                         f"Answer with just A or B.")
                    response = run_inference(model, processor, front_img, q)
                    pred_closer = "A" if "A" in response.upper() and "B" not in response.upper() else \
                                  "B" if "B" in response.upper() else None
                    correct = (pred_closer == gt_closer)

                    print(f"  [REL] A={obj_a['vis_desc'][:20]}({obj_a['gt_distance']:.1f}m) "
                          f"B={obj_b['vis_desc'][:20]}({obj_b['gt_distance']:.1f}m) "
                          f"gt={gt_closer} pred={pred_closer} correct={correct}")
                    results["relative_order"].append({
                        "frame_token":  frame_token,
                        "obj_a":        obj_a["vis_desc"],
                        "obj_b":        obj_b["vis_desc"],
                        "dist_a":       obj_a["gt_distance"],
                        "dist_b":       obj_b["gt_distance"],
                        "gt_diff_m":    gt_diff,
                        "gt_closer":    gt_closer,
                        "pred_closer":  pred_closer,
                        "correct":      correct,
                        "response":     response,
                    })

    # ── AGGREGATE METRICS ────────────────────────────────────
    print("\n" + "="*60)
    print("AGGREGATE RESULTS")
    print("="*60)

    # Distance
    de = results["distance_estimation"]
    if de:
        maes = [r["mae"] for r in de if r["mae"] is not None]
        pcts = [r["pct_error"] for r in de if r["pct_error"] is not None]
        print(f"\nDistance Estimation (n={len(de)}, valid={len(maes)})")
        print(f"  MAE:           {np.mean(maes):.2f} m")
        print(f"  Median AE:     {np.median(maes):.2f} m")
        print(f"  Mean Pct Err:  {np.mean(pcts):.1f} %")

    # Bearing
    bc = results["bearing_classification"]
    if bc:
        valid = [r for r in bc if r["pred_bearing"] is not None]
        correct = [r for r in valid if r["correct"]]
        print(f"\nBearing Classification (n={len(bc)}, parseable={len(valid)})")
        print(f"  Accuracy: {len(correct)/len(valid)*100:.1f}% ({len(correct)}/{len(valid)})")

    # Relative Order
    ro = results["relative_order"]
    if ro:
        valid = [r for r in ro if r["pred_closer"] is not None]
        correct = [r for r in valid if r["correct"]]
        print(f"\nRelative Order (n={len(ro)}, parseable={len(valid)})")
        print(f"  Accuracy: {len(correct)/len(valid)*100:.1f}% ({len(correct)}/{len(valid)})")

    # save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
