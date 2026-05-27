"""
Use Case 1: Zero-shot Traffic Scene Understanding
Script:      usecase1_zeroshot.py
Purpose:     对比两种 prompt 策略（受限 vs 开放）在 BDD100K 上的 zero-shot 表现
Experiments:
  Exp-A: 给出合法选项，要求 VLM 从中选择
  Exp-B: 不给选项，VLM 自由描述
"""

import torch
import json
import re
import os
from pathlib import Path
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info

# ============================================================
# CONFIG
# ============================================================
MODEL_ID      = "Qwen/Qwen2.5-VL-7B-Instruct"
DATA_DIR      = "/home/xzh5180/Research/vlm-mobility/datasets/bdd100k_hf/data/"
SAMPLES_JSON  = "/home/xzh5180/Research/vlm-mobility/datasets/bdd100k_hf/samples.json"
OUTPUT_DIR    = "/home/xzh5180/Research/vlm-mobility/outputs/usecase1_zeroshot/"
N_IMAGES      = 50          # 前50张（按文件名字母排序，保证跨模型一致）
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


def run_inference(model, processor, image_path: str, system_prompt: str) -> tuple[str, dict | None]:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text",  "text": USER_PROMPT},
            ],
        },
    ]
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
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
    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    raw = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return raw, parse_json_output(raw)


def load_annotations(samples_json: str) -> dict:
    """返回 {filename: {weather, timeofday, scene}} 的查找表"""
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


# ============================================================
# MAIN
# ============================================================
def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ── 1. 选图：按文件名字母排序，取前 N_IMAGES 张 ──────────
    all_images = sorted([
        f for f in os.listdir(DATA_DIR) if f.endswith(".jpg")
    ])
    selected = all_images[:N_IMAGES]
    print(f"Selected {len(selected)} images (alphabetical order)")
    print(f"First: {selected[0]}  Last: {selected[-1]}\n")

    # ── 2. 加载标注查找表 ─────────────────────────────────────
    annotations = load_annotations(SAMPLES_JSON)

    # ── 3. 加载模型 ───────────────────────────────────────────
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    allocated = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded  |  VRAM used: {allocated:.1f} GB\n")

    # ── 4. 推理：两个实验 ─────────────────────────────────────
    results_a, results_b = [], []

    for i, fname in enumerate(selected):
        img_path = os.path.join(DATA_DIR, fname)
        gt = annotations.get(fname, {"weather": None, "timeofday": None, "scene": None})

        print(f"[{i+1:02d}/{N_IMAGES}] {fname}")

        # Exp-A
        raw_a, parsed_a = run_inference(model, processor, img_path, SYSTEM_A)
        results_a.append({
            "image": fname,
            "ground_truth": gt,
            "raw_output": raw_a,
            "parsed": parsed_a,
            "parse_success": parsed_a is not None,
        })

        # Exp-B
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
    for tag, results in [("expA", results_a), ("expB", results_b)]:
        out_path = Path(OUTPUT_DIR) / f"{tag}_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved → {out_path}")

    # ── 6. 简单统计：parse 成功率 ─────────────────────────────
    for tag, results in [("Exp-A", results_a), ("Exp-B", results_b)]:
        success = sum(r["parse_success"] for r in results)
        print(f"{tag} parse success: {success}/{N_IMAGES} ({100*success/N_IMAGES:.0f}%)")


if __name__ == "__main__":
    main()
