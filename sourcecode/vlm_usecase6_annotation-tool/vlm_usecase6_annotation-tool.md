# Use Case 6: VLM as an Annotation Tool

## Purpose
Reframe the VLM from an *evaluated system* (its role in UC1–UC5) to a *practical annotation production tool*: even if the VLM's perception/reasoning is imperfect, can it still generate a useful first-pass annotation that a human only needs to review and correct, rather than replacing expert judgment entirely? This maps directly onto a real lab need — unlabeled work-zone photos — with the goal of cutting manual annotation cost substantially.

## Data
DriveLM-nuScenes fixed 9-frame evaluation subset (same as UC1–UC5), using DriveLM's existing QA/`key_object_infos` as a proxy for "ground-truth annotation" to evaluate VLM-generated annotations against.

## Experiment Design
Two sub-experiments with Qwen2.5-VL-7B-Instruct, both producing three annotation types per frame: a scene **description** (free text), **object category labels** (structured list), and a **safety-related QA pair**.
- **UC6a** — zero-shot **structured** prompt template only (fixed output format, all three annotation types), establishing a baseline.
- **UC6b** — comparison of three prompt strategies for the same annotation task: **free-form** (no format constraint), **structured** (fixed template, same as UC6a), and **few-shot** (1–2 GT-style examples provided as demonstrations).

Evaluation: Parsability (can the output be automatically parsed at all), ROUGE-1/ROUGE-L for description and safety-QA text, and set-level Precision/Recall/F1 for object labels.

## Results
- **UC6a**: Parsability = **1.000** (structured prompting guarantees parseable output), but Object label F1 = 0.684, with the same systematic **Barrier/Pedestrian miss-detection** pattern seen in UC4 persisting here.
- **UC6b — no universally best strategy**, with clear task-dependent trade-offs:
  - **Structured** prompting was best for object recall and for generating well-formed safety QA pairs.
  - **Few-shot** was best for matching GT's *description style* (wording/format alignment).
  - **Free-form** completely failed the safety-QA task (parsability = **0.000**) — without a format constraint, the model doesn't reliably produce a well-formed question-answer pair at all.

## Key Findings
- A **prompt-strategy × task-type interaction effect**: which prompting strategy is "best" depends on which annotation sub-task you're evaluating — there's no single winning strategy across description, labeling, and QA generation simultaneously.
- **Format constraints can trigger default-value anchoring** in the safety-QA sub-task — forcing a fixed output shape sometimes pushes the model toward generic, low-information answers rather than genuinely engaging with the specific frame.
- **Few-shot content is skewed by example semantics** — the specific examples chosen bias the *content* of what the model generates, not just its format (consistent with UC2's finding that few-shot examples matter a lot).

## Key Takeaway
Even with imperfect underlying perception (established across UC1–UC5), a VLM can serve as a practically useful annotation-drafting tool — but only with carefully chosen, task-specific prompt strategies; there is no one-size-fits-all prompting approach across annotation sub-tasks.
