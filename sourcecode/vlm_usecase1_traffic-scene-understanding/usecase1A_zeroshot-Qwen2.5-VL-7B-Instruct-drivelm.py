import json, os, torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

DATA_PATH  = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm/v1_1_train_nus.json"
IMG_BASE   = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm"
OUT_PATH   = "/home/xzh5180/Research/vlm-mobility/outputs/usecase1A_zeroshot_qwen25_drivelm.json"
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# ── Fixed evaluation subset: 3 scenes × 3 frames × 5 questions = 45 samples ──
MAX_SCENES     = 3
MAX_FRAMES     = 3
MAX_QUESTIONS  = 5

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

print("Loading model (4bit)...")
bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME, quantization_config=bnb_cfg, device_map="auto"
)
model.eval()
print("Model loaded.")

with open(DATA_PATH) as f:
    data = json.load(f)

scene_keys = list(data.keys())[:MAX_SCENES]
results = []

for scene_key in scene_keys:
    scene = data[scene_key]
    for frame_id, frame in list(scene["key_frames"].items())[:MAX_FRAMES]:
        img_rel = frame["image_paths"].get("CAM_FRONT")
        if not img_rel:
            continue
        img_path = os.path.join(IMG_BASE, img_rel.replace("../nuscenes/", "./nuscenes/"))
        if not os.path.exists(img_path):
            continue

        for qa in frame["QA"]["perception"][:MAX_QUESTIONS]:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img_path},
                {"type": "text",  "text": qa["Q"]}
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                               return_tensors="pt").to(model.device)
            with torch.no_grad():
                out_ids = model.generate(**inputs, max_new_tokens=128)
            pred = processor.batch_decode(
                out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0].strip()

            results.append({
                "scene": scene_key, "frame": frame_id,
                "question": qa["Q"], "gt": qa["A"], "pred": pred
            })
            print(f"[{len(results):02d}] Q: {qa['Q']}")
            print(f"      GT: {qa['A']}")
            print(f"      PR: {pred}\n")

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Done. {len(results)} pairs → {OUT_PATH}")
