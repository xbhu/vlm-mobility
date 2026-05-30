"""
Use Case 1: Zero-shot Traffic Scene Understanding
Script:      usecase1_zeroshot-PaliGemma2-3B.py
Model:       google/paligemma2-3b-mix-448 (4-bit bitsandbytes)
Note:        Gated model — requires HF login and Google license acceptance.
             huggingface-cli login  before running.

Architecture difference vs Qwen/InternVL/LLaMA:
  PaliGemma 2 has NO system/user chat separation.
  Prompt = image + single text string. No chat template.
"""

import torch
import json
import re
import os
from pathlib import Path
from PIL import Image
from transformers import (
    PaliGemmaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

# ============================================================
# CONFIG
# ============================================================
MODEL_ID     = "google/paligemma2-3b-mix-448"
DATA_DIR     = "/home/xzh5180/Research/vlm-mobility/datasets/bdd100k_hf/data/"
SAMPLES_JSON = "/home/xzh5180/Research/vlm-mobility/datasets/bdd100k_hf/samples.json"
OUTPUT_DIR   = "/home/xzh5180/Research/vlm-mobility/outputs/usecase1_zeroshot/"
N_IMAGES     = 50
MAX_NEW_TOKENS = 512

# ============================================================
# QUANTIZATION CONFIG
# 注：PaliGemma2-3B 在 BF16 下只需 ~6GB，12GB 显存放得下
# 但这里保持 4-bit 与其他模型一致，便于对比
# ============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# ============================================================
# PROMPTS
# PaliGemma 2 没有 system/user 分离，把所有指令合并成一段文字
# Exp-A 和 Exp-B 的区别仍然保留：有无固定选项
# ============================================================
PROMPT_A = "<image>\n" + (
    "You are a traffic scene analysis assistant. "
    "Analyze the image and output ONLY a valid JSON object with no extra text.\n"
    "Output format:\n"
    '{"weather": "<one of: clear, overcast, rainy, snowy, foggy, partly cloudy>", '
    '"timeofday": "<one of: daytime, dawn/dusk, night>", '
    '"scene": "<one of: city street, residential, highway, parking lot, tunnel, gas stations, undefined>"}\n'
    "Analyze this image and output the JSON only."
)

PROMPT_B = "<image>\n" + (
    "You are a traffic scene analysis assistant. "
    "Analyze the image and output ONLY a valid JSON object with no extra text.\n"
    "Output format:\n"
    '{"weather": "<describe the weather condition>", '
    '"timeofday": "<describe the time of day>", '
    '"scene": "<describe the scene type>"}\n'
    "Analyze this image and output the JSON only."
)

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


def run_inference(model, processor, image_path: str, prompt: str) -> tuple[str, dict | None]:
    """
    PaliGemma 2 的推理接口：
    - 直接传 PIL Image + 文本 prompt，无 chat template
    - 输出截取：去掉 input tokens，只保留新生成部分
    """
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    ).to("cuda")

    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # 只解码新生成的 token
    new_tokens = generated_ids[0][input_len:]
    raw = processor.decode(new_tokens, skip_special_tokens=True).strip()
    return raw, parse_json_output(raw)


# ============================================================
# MAIN
# ============================================================
def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ── 1. 选图 ───────────────────────────────────────────────
    all_images = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".jpg")])
    selected = all_images[:N_IMAGES]
    print(f"Selected {len(selected)} images")
    print(f"First: {selected[0]}  Last: {selected[-1]}\n")

    # ── 2. 加载标注 ───────────────────────────────────────────
    annotations = load_annotations(SAMPLES_JSON)

    # ── 3. 加载模型 ───────────────────────────────────────────
    print("Loading model (first run downloads ~6GB)...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
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

        raw_a, parsed_a = run_inference(model, processor, img_path, PROMPT_A)
        results_a.append({
            "image": fname,
            "ground_truth": gt,
            "raw_output": raw_a,
            "parsed": parsed_a,
            "parse_success": parsed_a is not None,
        })

        raw_b, parsed_b = run_inference(model, processor, img_path, PROMPT_B)
        results_b.append({
            "image": fname,
            "ground_truth": gt,
            "raw_output": raw_b,
            "parsed": parsed_b,
            "parse_success": parsed_b is not None,
        })

        print(f"  A: {parsed_a}  |  B: {parsed_b}  |  GT: {gt}")

    # ── 5. 保存结果 ───────────────────────────────────────────
    model_tag = "PaliGemma2-3B"
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
