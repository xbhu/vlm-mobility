"""
Use Case 1C: Fine-tuning LLaMA 3.2 Vision 11B on BDD100K
Script: usecase1C_finetune-Llama3.2-Vision-11B.py

Training set: identical to 1A (SEED=42, N_TRAIN=250, N_VAL=25, N_TEST=50)
OOM fixes:
  1. expandable_segments reduces VRAM fragmentation
  2. Resize images to 560x560 (single tile) to reduce vision encoder memory
  3. LoRA applied only to language model layers, skipping the vision encoder
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch, json, re, random
from pathlib import Path
from PIL import Image
from transformers import (
    MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType

# ============================================================
# CONFIG
# ============================================================
MODEL_ID     = "meta-llama/Llama-3.2-11B-Vision-Instruct"
DATA_DIR     = "/home/xzh5180/Research/vlm-mobility/datasets/bdd100k_hf/data/"
SAMPLES_JSON = "/home/xzh5180/Research/vlm-mobility/datasets/bdd100k_hf/samples.json"
OUTPUT_DIR   = "/home/xzh5180/Research/vlm-mobility/outputs/usecase1_finetune/Llama3.2-Vision-11B/"
ADAPTER_DIR  = OUTPUT_DIR + "adapter/"

N_TEST       = 50
N_TRAIN      = 250
N_VAL        = 25
SEED         = 42

EPOCHS       = 2
GRAD_ACCUM   = 8
LR           = 2e-4

LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05

# LLaMA 3.2 Vision processes images using 560x560 tiles
# resizing to this size guarantees only 1 tile, greatly reducing vision encoder memory
IMAGE_SIZE   = 560

# ============================================================
# PROMPT
# ============================================================
SYSTEM_PROMPT = (
    "You are a traffic scene analysis assistant. "
    "Analyze the given image and output ONLY a valid JSON object with no extra text.\n"
    "Output format:\n"
    '{"weather": "<one of: clear, overcast, rainy, snowy, foggy, partly cloudy>", '
    '"timeofday": "<one of: daytime, dawn/dusk, night>", '
    '"scene": "<one of: city street, residential, highway, parking lot, tunnel, gas stations, undefined>"}'
)
USER_PROMPT = "Analyze this image and output the JSON only."

# ============================================================
# Data preparation
# ============================================================
def load_annotations(samples_json):
    with open(samples_json) as f:
        data = json.load(f)
    lookup = {}
    for s in data["samples"]:
        fname = Path(s["filepath"]).name
        w  = (s.get("weather")   or {}).get("label")
        t  = (s.get("timeofday") or {}).get("label")
        sc = (s.get("scene")     or {}).get("label")
        if w and t and sc and w != "undefined":
            lookup[fname] = {"weather": w, "timeofday": t, "scene": sc}
    return lookup

def prepare_splits(data_dir, annotations):
    all_images = sorted([f for f in os.listdir(data_dir) if f.endswith(".jpg")])
    pool = [f for f in all_images[N_TEST:] if f in annotations]
    random.seed(SEED)
    random.shuffle(pool)
    val_list   = pool[:N_VAL]
    train_list = pool[N_VAL : N_VAL + N_TRAIN]
    print(f"Split → train:{len(train_list)}  val:{len(val_list)}  test:{N_TEST}")
    print(f"Train[0]: {train_list[0]}  Train[-1]: {train_list[-1]}")
    return train_list, val_list

# ============================================================
# Label masking
# ============================================================
def build_inputs_with_labels(processor, image_path, gt_json, device):
    # resize to single tile size to reduce vision encoder memory
    image = Image.open(image_path).convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS
    )

    full_msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": USER_PROMPT},
        ]},
        {"role": "assistant", "content": gt_json},
    ]
    prompt_msgs = full_msgs[:2]

    full_text   = processor.apply_chat_template(full_msgs,   add_generation_prompt=False)
    prompt_text = processor.apply_chat_template(prompt_msgs, add_generation_prompt=True)

    full_enc   = processor(images=image, text=full_text,   return_tensors="pt")
    prompt_enc = processor(images=image, text=prompt_text, return_tensors="pt")
    prompt_len = prompt_enc["input_ids"].shape[1]

    labels = full_enc["input_ids"].clone()
    labels[0, :prompt_len] = -100

    result = {
        "input_ids":      full_enc["input_ids"].to(device),
        "attention_mask": full_enc["attention_mask"].to(device),
        "labels":         labels.to(device),
    }
    for key in ["pixel_values", "aspect_ratio_ids", "aspect_ratio_mask",
                "cross_attention_mask"]:
        if key in full_enc:
            result[key] = full_enc[key].to(device)
    return result

# ============================================================
# Train / validate one epoch
# ============================================================
def run_epoch(model, processor, samples, optimizer, device, is_train):
    model.train() if is_train else model.eval()
    total_loss, steps = 0.0, 0

    for i, (fname, gt) in enumerate(samples):
        img_path = os.path.join(DATA_DIR, fname)
        gt_json  = json.dumps(gt)

        try:
            inputs = build_inputs_with_labels(processor, img_path, gt_json, device)
        except Exception as e:
            print(f"  skip {fname}: {e}")
            continue

        ctx = torch.no_grad() if not is_train else torch.enable_grad()
        with ctx:
            outputs = model(**inputs)
            loss = outputs.loss / GRAD_ACCUM

        if is_train:
            loss.backward()
            if (i + 1) % GRAD_ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad()
            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()

        total_loss += loss.item() * GRAD_ACCUM
        steps += 1

        if is_train and (i + 1) % 50 == 0:
            print(f"  step {i+1}/{len(samples)}  loss={total_loss/steps:.4f}")

    return total_loss / steps if steps > 0 else 0.0

# ============================================================
# Evaluation
# ============================================================
def parse_json(text):
    try:
        return json.loads(text.strip())
    except:
        m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        try:
            return json.loads(m.group()) if m else None
        except:
            return None

def evaluate(model, processor, annotations, device):
    model.eval()
    all_images = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".jpg")])
    test_imgs  = all_images[:N_TEST]
    results = []

    for fname in test_imgs:
        img_path = os.path.join(DATA_DIR, fname)
        gt    = annotations.get(fname, {})
        image = Image.open(img_path).convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS
        )
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": USER_PROMPT},
            ]},
        ]
        text_in = processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs  = processor(images=image, text=text_in, return_tensors="pt").to(device)

        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=128,
                                 do_sample=False, temperature=None, top_p=None)
        raw = processor.decode(
            gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        ).strip()
        parsed = parse_json(raw)
        results.append({"image": fname, "gt": gt, "parsed": parsed,
                        "parse_success": parsed is not None})

    parsed_r = [r for r in results if r["parse_success"]]
    n = len(parsed_r)
    print(f"\n=== Post-finetune Eval (parsed {n}/{N_TEST}) ===")
    for field in ["weather", "timeofday", "scene"]:
        correct = sum(1 for r in parsed_r if r["parsed"].get(field) == r["gt"].get(field))
        print(f"  {field}: {correct}/{n} = {100*correct/n:.0f}%" if n else f"  {field}: N/A")

    out = Path(OUTPUT_DIR) / "eval_post_finetune.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {out}")

# ============================================================
# MAIN
# ============================================================
def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(ADAPTER_DIR).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    annotations = load_annotations(SAMPLES_JSON)
    train_list, val_list = prepare_splits(DATA_DIR, annotations)
    train_samples = [(f, annotations[f]) for f in train_list]
    val_samples   = [(f, annotations[f]) for f in val_list]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    print("Loading model...")
    model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # LoRA applied only to language model self-attention and cross-attention layers
    # Skip the vision encoder (vision_model) to reduce VRAM usage
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=r"language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",   # skip vision encoder
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*50}\nEpoch {epoch}/{EPOCHS}\n{'='*50}")
        tr = run_epoch(model, processor, train_samples, optimizer, device, True)
        vl = run_epoch(model, processor, val_samples,   optimizer, device, False)
        history["train_loss"].append(tr)
        history["val_loss"].append(vl)
        print(f"Epoch {epoch}: train={tr:.4f}  val={vl:.4f}")
        if vl < best_val:
            best_val = vl
            model.save_pretrained(ADAPTER_DIR)
            print(f"  ✓ Best adapter saved")

    with open(Path(OUTPUT_DIR) / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\nEvaluating on test set...")
    evaluate(model, processor, annotations, device)

if __name__ == "__main__":
    main()
