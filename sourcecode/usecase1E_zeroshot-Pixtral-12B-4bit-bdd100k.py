"""
Use Case 1E: Zero-shot Traffic Scene Understanding
Script:      usecase1E_zeroshot-Pixtral-12B-4bit.py
Model:       mistralai/Pixtral-12B-2409 (4-bit bitsandbytes)
Note:        Gated model — requires HF login and Mistral license acceptance.
             huggingface-cli login  before running.
             pip install mistral_common --break-system-packages
"""

import torch
import json
import re
import os
from pathlib import Path
from PIL import Image
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

# ============================================================
# CONFIG
# ============================================================
MODEL_ID     = "mistralai/Pixtral-12B-2409"
DATA_DIR     = "/home/xzh5180/Research/vlm-mobility/datasets/bdd100k_hf/data/"
SAMPLES_JSON = "/home/xzh5180/Research/vlm-mobility/datasets/bdd100k_hf/samples.json"
OUTPUT_DIR   = "/home/xzh5180/Research/vlm-mobility/outputs/usecase1_zeroshot/"
N_IMAGES     = 50
MAX_NEW_TOKENS = 512

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
# PROMPTS  —— 与其他模型完全一致
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


def run_inference(model, processor, image_path: str, system_prompt: str) -> tuple[str, dict | None]:
    """
    Pixtral 推理接口：
    - 使用 LLaVA 风格的消息格式
    - system prompt 单独传入
    - 图像以 PIL Image 形式传给 processor
    """
    image = Image.open(image_path).convert("RGB")

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]

    text_input = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    inputs = processor(
        images=image,
        text=text_input,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    trimmed = generated_ids[:, inputs["input_ids"].shape[-1]:]
    raw = processor.decode(trimmed[0], skip_special_tokens=True).strip()
    return raw, parse_json_output(raw)


# ============================================================
# MAIN
# ============================================================
def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ── 1. 选图（相同排序）────────────────────────────────────
    all_images = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".jpg")])
    selected = all_images[:N_IMAGES]
    print(f"Selected {len(selected)} images")
    print(f"First: {selected[0]}  Last: {selected[-1]}\n")

    # ── 2. 加载标注 ───────────────────────────────────────────
    annotations = load_annotations(SAMPLES_JSON)

    # ── 3. 加载模型 ───────────────────────────────────────────
    print("Loading model (first run downloads ~24GB)...")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    allocated = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded  |  VRAM used: {allocated:.1f} GB\n")

    # ── 4. 推理 ───────────────────────────────────────────────
    results_a, results_b = [], []

    for i, fname in enumerate(selected):
        img_path = os.path.join(DATA_DIR, fname)
        gt = annotations.get(fname, {"weather": None, "timeofday": None, "scene": None})

        print(f"[{i+1:02d}/{N_IMAGES}] {fname}")

        raw_a, parsed_a = run_inference(model, processor, img_path, SYSTEM_A)
        results_a.append({
            "image": fname,
            "ground_truth": gt,
            "raw_output": raw_a,
            "parsed": parsed_a,
            "parse_success": parsed_a is not None,
        })

        raw_b, parsed_b = run_inference(model, processor, img_path, SYSTEM_B)
        results_b.append({
            "image": fname,
            "ground_truth": gt,
            "raw_output": raw_b,
            "parsed": parsed_b,
            "parse_success": parsed_b is not None,
        })

        print(f"  A: {parsed_a}  |  B: {parsed_b}  |  GT: {gt}")

    # ── 5. 保存结果 ───────────────────────────────────────────
    model_tag = "Pixtral-12B-4bit"
    for tag, results in [("expA", results_a), ("expB", results_b)]:
        out_path = Path(OUTPUT_DIR) / f"{tag}_{model_tag}_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved → {out_path}")

    # ── 6. 统计 ───────────────────────────────────────────────
    for tag, results in [("Exp-A", results_a), ("Exp-B", results_b)]:
        success = sum(r["parse_success"] for r in results)
        print(f"{tag} parse success: {success}/{N_IMAGES} ({100*success/N_IMAGES:.0f}%)")


if __name__ == "__main__":
    main()
