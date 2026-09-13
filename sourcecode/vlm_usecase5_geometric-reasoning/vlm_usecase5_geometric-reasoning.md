# Use Case 5: Geometric Reasoning Inference

## Purpose
Characterize where a VLM's geometric/metric reasoning ability breaks down in traffic scenes — testing a capability gradient from semantic (bearing/direction), through comparative (relative distance), to metric (absolute distance) reasoning — and separately test whether prompt granularity (coarse vs. object-specific) helps or hurts geometric reasoning, in contrast to UC4's finding that fine-grained prompts hurt perception tasks.

## Data
DriveLM-nuScenes fixed 9-frame subset, cross-referenced with full **nuScenes 3D annotations** (`sample_annotation.json`, `ego_pose.json`, etc. — 850 scenes, 34,149 samples, 1,166,187 annotations, downloaded separately via a public CloudFront mirror since these weren't on the server). Ground truth was constructed by projecting nuScenes 3D annotations onto the camera image plane and matching to DriveLM's `key_object_infos` pixel coordinates (100px threshold), then computing true Euclidean distance and bearing from `ego_pose`.

## Experiment Design
A preliminary conceptual discussion first established that VLM geometric reasoning is **language-based statistical mapping**, not true metric computation — unsuitable as a substitute for 3D object detection due to monocular scale ambiguity and no geometric inductive bias. This reframed the use case's goal from "can VLM replace 3D detection" to "characterize the capability boundary."

Two sub-experiments with Qwen2.5-VL-7B-Instruct:
- **UC5a — Geometric reasoning ceiling**: three task types — bearing classification (8-sector, semantic-level, GT from DriveLM QA), relative-order judgment ("which of A/B is closer to ego?", comparative-level, GT from nuScenes 3D distances), and absolute distance estimation (metric-level, GT = Euclidean ego-to-object distance).
- **UC5b — Prompt granularity effect**: same geometric questions asked with a coarse prompt (no specific object named, e.g., "is there a car ahead, how far?") vs. a fine prompt (specific object referenced by its `Visual_description`, e.g., "how far is that Brown SUV?").

## Results
- **UC5a**: Distance estimation MAE = **61.2m (326% relative error)**, with systematic anchoring to a default output of ~100m when uncertain. Bearing classification accuracy = **22.7%**, with a systematic bias toward predicting "front" even for `CAM_BACK` objects. Relative-order accuracy = **47.8%** (near chance level). The expected semantic > comparative > metric capability gradient **did not clearly materialize** — the real boundary observed was more about presence/absence of salient visual cues than task type per se.
- **UC5b**: Coarse-prompt distance MAE = 24.94m vs. fine-prompt MAE = 63.58m — coarse *appeared* better, but investigation showed this was because coarse prompts anchored to a default of ~1m rather than ~100m — **both are systematic biases, not genuine inference**, so the apparent "coarse wins" result is not a real capability finding. Coarse-prompt relative-order questions produced **0 parseable responses**, vs. 47.8% for fine prompts — confirming that structured/specific prompts are a *necessary condition* for evaluable output at all, even before considering accuracy.

## Key Takeaway
VLM geometric reasoning fails not gracefully along a capability gradient but through **anchoring to default values when uncertain** — a distinct failure mode from simple inaccuracy, and one that can make naive metric comparisons (like "coarse MAE beat fine MAE") misleading unless the underlying response distribution is inspected.

## Related Discussion (kept — informs future research framing)
The person and Claude discussed why VLM-estimated distances cannot be composed with known ego-position to reconstruct full 3D bounding boxes (as a cheaper substitute for 3D detection): VLM distance outputs are language pattern-matches, not triangulated measurements, and 3D boxes additionally require object size/heading, which VLMs cannot reliably extract from an image description. It was also confirmed that pavement-marking detection for AV applications should use 3D/BEV segmentation models (BEVFusion, HDMapNet, LiDAR) rather than VLMs, since that task requires pixel/centimeter-level precision incompatible with language-based estimation.
