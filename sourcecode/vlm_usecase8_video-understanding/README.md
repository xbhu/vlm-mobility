# Use Case 8: VLM Video Understanding via Image Sequences

## Purpose
Move from single-frame static understanding (UC1–UC7) to a temporal question: when a VLM is shown multiple frames of the same scene, does it genuinely reason about *time* (what changed, what happened when), or does it just produce independent per-frame descriptions stitched together? This matters directly for AV deployment, where driving decisions are inherently based on continuous perception, not single snapshots.

## Data
DriveLM-nuScenes. Important caveat established up front: DriveLM provides ~6 **key frames** per scene at uneven intervals — not continuous video. This was treated as a deliberate, acknowledged limitation (using key-frame sequences as a "pseudo-video" proxy) rather than a blocker; true continuous-video understanding would require the full 400GB nuScenes video release, which was out of scope.

## Experiment Design
Two sub-experiments with Qwen2.5-VL-7B-Instruct (multi-image input in a single prompt, with no dedicated temporal positional encoding — the model can only infer order from prompt image sequence position, explicit frame-number labels in the text, or visible content differences between frames):
- **UC8a — Single-frame vs. multi-frame comparison**: for each scene, the same set of temporal questions (covering action, change, and trajectory) was asked once using only the first frame, and once using the full frame sequence, to see whether multi-frame input produces genuinely different (cross-frame-inferred) answers vs. just more verbose single-frame-style description.
- **UC8b — Temporal question-type analysis**: three question categories of increasing expected difficulty — **action recognition** ("what is the ego vehicle doing?" — answerable from cross-frame state summary), **change detection** ("what changes between the first and last frame?" — requires comparing two specific frames), and **event localization** ("in which frame does X happen?" — requires frame-level precise tracking).

## Results
- **Temporal language in model outputs was driven by question phrasing, not visual reasoning**: single-frame and multi-frame inputs produced structurally identical "Initially → Then → Finally" narrative responses to the same question — the model generates temporal-sounding language whenever the question asks for it, regardless of whether it actually had multiple frames to reason over.
- **Frame references were triggered by the literal word "frame" appearing in the question**, not by genuine cross-frame inference — a surface pattern-matching behavior rather than real tracking.
- **The anticipated difficulty gradient (Action > Change > Event) did not materialize as expected.** Event localization showed a completion rate of 1.00 but a temporal-accuracy score of 0.0 — indicating the model was **avoiding the task** (giving confident-sounding but temporally unfounded answers) rather than genuinely attempting and failing at fine-grained localization.
- **Contradiction detection** revealed cross-variant factual inconsistencies — e.g., one prompt variant hallucinating a vehicle's presence while another variant (same frames) correctly reported none, indicating unstable rather than consistently wrong perception.
- Keyword-based completion metrics proved **unreliable** — they conflated honest, correct negative answers ("no such event occurs") with genuine task avoidance, requiring more careful manual/qualitative interpretation of results.

## Key Takeaway
Apparent temporal reasoning in VLM output is often a *linguistic artifact* of question phrasing rather than evidence of genuine cross-frame inference — a caution for any evaluation that uses the presence of temporal language ("then," "after," "frame 3") as a proxy for temporal understanding.
