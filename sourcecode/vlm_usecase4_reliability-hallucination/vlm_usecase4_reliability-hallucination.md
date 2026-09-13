# Use Case 4: Reliability and Hallucination Evaluation

## Purpose
Stress-test whether the VLM is genuinely perceiving images or relying on statistical shortcuts, through three angles: object-presence hallucination, counting/negation failures, and confidence calibration (does the model know when it doesn't know?).

## Data
DriveLM-nuScenes, same fixed 9-frame evaluation subset (3 scenes × 3 frames), using `key_object_infos` as GT for object presence/absence and counts.

## Experiment Design
Three sub-experiments, all with Qwen2.5-VL-7B-Instruct on the same 9-frame subset:
- **UC4a — Object Hallucination (POPE-style)**: positive questions ("Is there a `<X>` in this image?" where X exists) paired with negative questions (X borrowed from another frame, absent here). Measures yes-bias, precision, recall, F1.
- **UC4b — Negation and Counting**: category-level counting ("How many vehicles?") and zero-count negation ("Are there any pedestrians?") questions, compared to GT counts.
- **UC4c — Confidence Calibration**: model asked to attach a confidence score (0–1) to each answer; actual accuracy is measured within each confidence bin to compute Expected Calibration Error (ECE).

## Results
- **UC4a**: Found a **conservative No-bias** (yes-bias = 0.358, Recall = 0.611) — the opposite of the anticipated yes-bias — driven by fine-grained `Visual_description`-based prompts making the model overly cautious.
- **UC4b**: Vehicle counting was the weakest sub-task (exact-match accuracy = 0.222, negative bias). Barrier objects were a **systematic hallucination target** (counting MAE = 2.0) yet negation accuracy for Barrier was 0.667 — an internally inconsistent pattern (the model both over-counts and sometimes correctly denies presence). Coarse-category negation was generally reliable overall (accuracy = 0.909).
- **UC4c**: Revealed **severe overconfidence** — 78 of 81 questions received confidence scores above 0.8, the middle three calibration bins were essentially empty, overconfidence rate = 0.952, and **ECE = 0.1981**. Confidence scores cannot be used to filter likely-wrong answers — the model does not know when it doesn't know.

## Key Cross-Use-Case Pattern Established
**Description granularity is the key reliability variable**: coarse-category queries perform stably, while fine-grained visual-description-based queries trigger conservative failure modes. This pattern recurs and is explicitly tested/extended in later use cases (UC5, UC6, UC8).

## Note
UC4b's counting sub-task was recognized to overlap somewhat with UC3's object-count evaluation; this was acknowledged but not redesigned, since the overlap didn't undermine UC4's core reliability/calibration framing.
