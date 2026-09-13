# Use Case 1: Traffic Scene Understanding — Zero-Shot Evaluation + Fine-Tuning

## Purpose
Establish the baseline use case of the VLM-Mobility series: evaluate whether general-purpose vision-language models can perceive traffic scenes accurately (zero-shot), then test whether lightweight fine-tuning improves that capability, given a VLM's two-component structure (frozen vision encoder + LoRA-tuned language model).

## Data
Two dataset versions were used across this use case's lifecycle:
- **Initial version — BDD100K** (UC Berkeley): single-camera front-view driving frames. This version was run first but was superseded — encountered significant model-compatibility issues (see below) and was replaced by a richer, more research-relevant dataset.
- **Final/official version — DriveLM-nuScenes**: `v1_1_train_nus.json`, 696 scenes, 4,072 frames, structured graph-based QA across perception/prediction/planning/behavior categories, 6-camera surround view (only `CAM_FRONT` used). 162,480 total perception QA entries. A fixed evaluation subset of 3 scenes × 3 frames × 5 questions (45 QA pairs) was established here and reused consistently across all later use cases in the series.

## Experiment Design
- **Zero-shot evaluation** across three VLMs: Qwen2.5-VL-7B-Instruct, LLaMA-3.2-11B-Vision (`MllamaForConditionalGeneration`), and LLaVA-OneVision-7B (`LlavaOnevisionForConditionalGeneration`), all loaded 4-bit quantized.
- A custom evaluation metric was developed — **Object Recall** (GT objects appearing in the model's prediction) and **Hallucination Rate** (predicted objects absent from GT) — chosen over BLEU/METEOR because free-form model output doesn't match DriveLM's templated GT answers well.
- **Fine-tuning**: only Qwen2.5-VL-7B was fine-tuned (GPU constraints ruled out the others). LoRA applied only to `q_proj`/`v_proj` in the language model, vision encoder kept **frozen** — the standard/recommended approach for VLM fine-tuning under limited VRAM.

## Results
- **Zero-shot**: Qwen (Recall 0.702, Halluc 0.616) — over-generating. LLaMA (Recall 0.676, Halluc 0.394) — more hallucination-controlled but prone to repetition loops. LLaVA (Recall 0.133, Halluc 0.300) — severely under-generating, single-word answers.
- **Fine-tuned Qwen**: Recall improved to 0.780–0.788, and **Hallucination Rate dropped sharply to 0.157–0.207**.
- **Key finding**: fine-tuning primarily *suppresses hallucination* rather than improving underlying perceptual capability — expected, since the frozen vision encoder means the model's actual "seeing" ability is unchanged; only how it reports what it sees is being tuned.

## Notable Technical Lessons (kept — recurring/general)
- On the BDD100K attempt: transformers 5.x broke InternVL2.5-8B compatibility; Pixtral-12B uses a proprietary Mistral format incompatible with `transformers`; PaliGemma2 requires an explicit `<image>` token (no chat template); LLaVA-OneVision doesn't support system messages; Qwen2.5-VL needs `min_pixels`/`max_pixels` constraints to avoid OOM.
- On DriveLM fine-tuning: a hand-written training loop stalled silently for 20 hours (GPU at 98% utilization, no loss output) — root cause was `process_vision_info` blocking during training. Fixed by switching to HuggingFace `Trainer` with a custom collator using `PIL.Image` loading directly, plus `gradient_checkpointing_enable()` and reduced LoRA rank (`r=8, lora_alpha=16`) to fit in 12GB VRAM.
