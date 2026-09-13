# Use Case 9: Image → Driving Decision (Direct Decision Paradigm)

## Purpose
Shift from the QA format used throughout UC1–UC8 ((image + question) → answer) to a direct decision paradigm (image → decision, with no question as an intermediate input). This tests whether the VLM's visual capability is genuinely "decision-ready" rather than merely "question-answering-ready" — removing the question removes an implicit scaffold that constrains output format, focuses visual attention, and activates trained answer patterns.

## Data
DriveLM-nuScenes fixed 9-frame subset (same 3 scenes × 3 frames used throughout the series). Behavior QA answers (following the pattern "The ego vehicle is going straight. The ego vehicle is [driving fast/slowly/not moving].") were used as ground-truth labels for a discrete decision space: speed (accelerate / maintain / decelerate / stop) × direction (change lane left / change lane right / turn left / turn right / go straight). Note: DriveLM v1.1 lacks a continuous Motion/trajectory field, so only discrete behavior classification was possible — not trajectory-coordinate output.

## Experiment Design
Two sub-experiments with Qwen2.5-VL-7B-Instruct:
- **UC9a — Zero-shot direct-decision baseline**: given only the front-camera image (no question), the model must output a decision directly. Evaluated along a dependency chain: (1) GT label reliability, (2) output parsability, (3) decision accuracy — each gating the validity of the next.
- **UC9b — Three-way prompt strategy comparison**: S1 (imperative/instruction-style, strict format constraint), S2 (role-based, framing the model as an "AV agent"), S3 (chain-of-thought, forcing explicit reasoning before the final decision) — testing how prompt structure trades off format compliance against decision quality, and whether CoT reasoning is actually consistent with (faithful to) its own stated conclusion.

(A build bug was found and fixed: both scripts initially used the wrong model class, `Qwen2VLForConditionalGeneration` instead of `Qwen2_5_VLForConditionalGeneration` — an architecture mismatch in the vision-encoder MLP layers that caused an assertion failure during quantized inference.)

## Results
- **UC9a**: Parsability = 1.0 (the model always produced *a* decision), but Speed accuracy = 0.222, Direction accuracy = 0.444, Combined accuracy = 0.111. GT direction was uniformly "straight" (9/9) in this subset, while the model showed a systematic **turn-left bias** in two of the three scenes.
- **UC9b**: A clear **parsability–accuracy tradeoff** across strategies — S1 (strict imperative) had parsability 1.0 but combined accuracy only 0.111; S3 (CoT) had lower parsability (0.667) but the best combined accuracy (0.500). Direction accuracy reached 1.0 for both S2 (role-based) and S3 (CoT). CoT reasoning-to-decision consistency was 0.778 (i.e., roughly 78% of the time the final decision matched what the reasoning chain implied).

## Key Findings
- **Strict format constraints suppress decision accuracy** — forcing rigid output structure (S1) trades away correctness for parseability.
- **The turn-left bias seen under strict formatting disappears with more expressive prompting** (S2/S3) — suggesting it was partly a format-compliance artifact, not a pure visual/decision bug.
- **Slow/stop confusion persists even with CoT**, though partially improved — a genuine capability limitation, not just a prompting issue.
- **CoT reasoning-decision consistency does not guarantee correctness** — the model can reason coherently to an internally consistent but factually wrong decision 22% of the time, meaning "the reasoning matches the answer" is not sufficient evidence the answer is right.
