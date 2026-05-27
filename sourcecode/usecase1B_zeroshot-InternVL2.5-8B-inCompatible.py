"""
Use Case 1B: Zero-shot Traffic Scene Understanding
Script:      usecase1B_zeroshot-InternVL2.5-8B.py
Model:       OpenGVLab/InternVL2_5-8B (4-bit bitsandbytes)

Compatibility fixes for transformers 5.x:
  1. Monkey-patch quantizer + accelerate to handle InternVL's missing
     all_tied_weights_keys attribute (expects dict, not list)
  2. Load tokenizer with use_fast=False to bypass protobuf/TikToken conflict
"""

# ============================================================
# COMPATIBILITY PATCHES  —— 必须在任何 transformers import 之后、
# 模型加载之前执行
# ============================================================
import transformers.quantizers.base as _qbase
import transformers.integrations.accelerate as _acc_int

_orig_get_keys = _qbase.get_keys_to_not_convert
def _patched_get_keys(model):
    val = getattr(model, 'all_tied_weights_keys', None)
    if val is None or isinstance(val, list):
        model.__dict__['all_tied_weights_keys'] = {}
    return _orig_get_keys(model)
_qbase.get_keys_to_not_convert = _patched_get_keys

_orig_compute = _acc_int.compute_module_sizes
def _patched_compute(model, *args, **kwargs):
    val = getattr(model, 'all_tied_weights_keys', None)
    if val is None or isinstance(val, list):
        model.__dict__['all_tied_weights_keys'] = {}
    return _orig_compute(model, *args, **kwargs)
_acc_int.compute_module_sizes = _patched_compute

# ============================================================
# IMPORTS
# ============================================================
import torch
import json
import re
import os
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

# ============================================================
# CONFIG
# ============================================================
MODEL_ID     = "OpenGVLab/InternVL2_5-8B"
DATA_DIR     = "/home/xzh5180/Research/vlm-mobility/datasets/bdd100k_hf/data/"
SAMPLES_JSON = "/home/xzh5180/Research/vlm-mobility/datasets/bdd100k_hf/samples.json"
OUTPUT_DIR   = "/home/xzh5180/Research/vlm-mobility/outputs/usecase1_zeroshot/"
N_IMAGES     = 50
MAX_NEW_TOKENS = 512
IMAGE_SIZE   = 448

# ============================================================
# QUANTIZATION CONFIG
# ============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# ============================================================
# IMAGE PREPROCESSING
# ============================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def load_image(image_path: str) -> torch.Tensor:
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(torch.bfloat16).cuda()

# ============================================================
# PROMPTS
# ============================================================
SYSTEM_A = (
    "You are a traffic scene analysis assistant. "
    "Analyze the given image and output ONLY a valid JSON object with no extra text.\n"
    "Output format:\n"
    '{"weather": "<one of: clear, overcast, rainy, snowy, foggy, partly cloudy>", '
    '"timeofday": "<one of: daytime, dawn/dusk, night>", '
    '"scene": "<one of: city street, residential, highway, parking lot, tunnel, gas stations, undefined>"}'
)

SYSTEM_B = (
    "You are a traffic scene analysis assistant. "
    "Analyze the given image and output ONLY a valid JSON object with no extra text.\n"
    "Output format:\n"
    '{"weather": "<describe the weather condition>", '
    '"timeofday": "<describe the time of day>", '
    '"scene": "<describe the scene type>"}'
)

USER_PROMPT = "Analyze this image and output the JSON only."

# ============================================================
# HELPERS
# ============================================================
def parse_json_output(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def load_annotations(samples_json: str) -> dict:
    with open(samples_json) as f:
        data = json.load(f)
    lookup = {}
    for s in data["samples"]:
        fname = Path(s["filepath"]).name
        lookup[fname] = {
            "weather":   s.get("weather",   {}).get("label", None),
            "timeofday": s.get("timeofday", {}).get("label", None),
            "scene":     s.get("scene",     {}).get("label", None),
        }
    return lookup


def run_inference(model, tokenizer, image_path: str, system_prompt: str) -> tuple[str, dict | None]:
    pixel_values = load_image(image_path)
    question = f"<image>\n{system_prompt}\n\n{USER_PROMPT}"
    generation_config = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    raw = model.chat(tokenizer, pixel_values, question, generation_config)
    return raw, parse_json_output(raw)


# ============================================================
# MAIN
# ============================================================
def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    all_images = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".jpg")])
    selected = all_images[:N_IMAGES]
    print(f"Selected {len(selected)} images")
    print(f"First: {selected[0]}  Last: {selected[-1]}\n")

    annotations = load_annotations(SAMPLES_JSON)

    print("Loading model...")
    model = AutoModel.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    # use_fast=False：绕过 transformers 5.x 的 TikToken/protobuf 冲突
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_fast=False,
    )

    allocated = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded  |  VRAM used: {allocated:.1f} GB\n")

    results_a, results_b = [], []

    for i, fname in enumerate(selected):
        img_path = os.path.join(DATA_DIR, fname)
        gt = annotations.get(fname, {"weather": None, "timeofday": None, "scene": None})

        print(f"[{i+1:02d}/{N_IMAGES}] {fname}")

        raw_a, parsed_a = run_inference(model, tokenizer, img_path, SYSTEM_A)
        results_a.append({
            "image": fname,
            "ground_truth": gt,
            "raw_output": raw_a,
            "parsed": parsed_a,
            "parse_success": parsed_a is not None,
        })

        raw_b, parsed_b = run_inference(model, tokenizer, img_path, SYSTEM_B)
        results_b.append({
            "image": fname,
            "ground_truth": gt,
            "raw_output": raw_b,
            "parsed": parsed_b,
            "parse_success": parsed_b is not None,
        })

        print(f"  A: {parsed_a}  |  B: {parsed_b}  |  GT: {gt}")

    model_tag = "InternVL2.5-8B"
    for tag, results in [("expA", results_a), ("expB", results_b)]:
        out_path = Path(OUTPUT_DIR) / f"{tag}_{model_tag}_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved → {out_path}")

    for tag, results in [("Exp-A", results_a), ("Exp-B", results_b)]:
        success = sum(r["parse_success"] for r in results)
        print(f"{tag} parse success: {success}/{N_IMAGES} ({100*success/N_IMAGES:.0f}%)")


if __name__ == "__main__":
    main()
