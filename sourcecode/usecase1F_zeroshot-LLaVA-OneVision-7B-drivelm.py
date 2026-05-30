import json, os, torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig
from PIL import Image

DATA_PATH  = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm/v1_1_train_nus.json"
IMG_BASE   = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm"
OUT_PATH   = "/home/xzh5180/Research/vlm-mobility/outputs/usecase1C_zeroshot_llava_ov_drivelm.json"
MODEL_NAME = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

# ── 固定评估子集：3 scenes × 3 frames × 5 questions = 45条 ──
MAX_SCENES    = 3
MAX_FRAMES    = 3
MAX_QUESTIONS = 5

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

print("Loading model (4bit)...")
bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
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

        image = Image.open(img_path).convert("RGB")

        for qa in frame["QA"]["perception"][:MAX_QUESTIONS]:
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": qa["Q"]}
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)

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
