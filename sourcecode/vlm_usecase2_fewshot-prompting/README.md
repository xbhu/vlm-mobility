# Use Case 2: Few-Shot Image-Text Prompting

## Purpose
Test whether in-context learning (few-shot prompting) — giving the VLM a small number of image+answer examples before the real question — improves performance on tasks where zero-shot performance is weak, specifically Prediction and Planning QA (which depend on reasoning about what happens next, unlike the more direct Perception task tested in UC1). Conceptually, this use case is "work on the prompt, not the model" — no weights change.

## Data
DriveLM-nuScenes (same dataset/infrastructure as UC1): fixed evaluation subset of 3 scenes × 3 frames × 5 questions = 45 items per question type, for both Prediction and Planning QA categories.

## Experiment Design
Three sub-experiments using Qwen2.5-VL-7B-Instruct (4-bit quantized):
- **2A** — Prediction QA, fixed 2-shot prompting.
- **2B** — Planning QA, fixed 2-shot prompting.
- **2C** — Shot-count ablation (1-shot / 2-shot / 3-shot) for both question types.
- A **zero-shot baseline** for both Prediction and Planning was added mid-session (initially only Prediction was planned) to allow proper before/after comparison.

## Results
| Condition | Recall | Hallucination Rate |
|---|---|---|
| Prediction — zero-shot | 0.122 | 0.889 |
| Prediction — 2-shot | 0.256 | 0.755 |
| Planning — zero-shot | 0.329 | 0.789 |
| Planning — 2-shot | 0.432 | 0.700 |

- Few-shot prompting roughly **doubled Prediction recall** and gave a meaningful boost to Planning recall, while reducing hallucination in both.
- Shot-count ablation (2C): **2-shot was the inflection point for Prediction** (little further gain from 3-shot); for Planning, hallucination rate kept declining through 3-shot, suggesting Planning benefits from more examples than Prediction does.

## Key Takeaway
Few-shot prompting is a genuine (if partial) fix for zero-shot weaknesses on reasoning-dependent tasks, but the effect is task-dependent — the optimal number of examples differs by question type, and gains plateau rather than continuing indefinitely.
