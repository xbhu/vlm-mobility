# Use Case 3: Spatial Grounding and Referring Localization

## Purpose
Test the VLM's ability to localize objects spatially in an image — not just describe what's present (UC1) but output where it is, via bounding-box prediction. Two prompt styles were compared: category-level detection vs. descriptive single-object reference.

## Data
DriveLM-nuScenes, same fixed 9-frame evaluation subset (3 scenes × 3 frames) as prior use cases, using `key_object_infos` (Category, Visual_description, 2D bbox) as ground truth.

## Experiment Design
Two sub-experiments with Qwen2.5-VL-7B-Instruct:
- **UC3a** — category-level prompts ("detect all vehicles"), zero-shot grounding, evaluated against all GT objects of that category in the frame.
- **UC3b** — descriptive single-object prompts, built from each object's `Visual_description` and its spatial position (derived from bbox center coordinates) — e.g., asking the model to locate one specific described object rather than a whole category.

Evaluation metrics: Recall@0.5 (IoU threshold), Mean IoU, and normalized Center Distance. Output parsing switched from Qwen's native grounding-token format to structured JSON `bbox_2d` output for reliability. A separate visualization script overlaid GT (green) vs. predicted (blue/red) boxes on images for qualitative inspection.

## Results
| Experiment | Recall@0.5 | Mean IoU | Center Distance |
|---|---|---|---|
| UC3a (category-level) | 0.600 | 0.507 | 0.148 |
| UC3b (descriptive, single-object) | 0.400 | 0.370 | **0.053** (−64%) |

- UC3a showed **severe over-detection**: the model frequently predicted 8–15 boxes when GT had only 1 object of that category.
- UC3b showed **worse IoU and Recall** than UC3a, but a **64% improvement in center-point accuracy** — meaning descriptive prompts help the model point in roughly the right direction/location, but the predicted box *boundaries* (size/shape) remain unreliable.

## Important Data-Quality Finding
DriveLM's ground-truth annotations were found to be **severely incomplete**: frames visibly containing 5–6 pedestrians and 2+ vehicles had GT=1 for each category. This established that DriveLM's `key_object_infos` annotates only *driving-decision-relevant* objects, not all visible targets — an important caveat for interpreting recall/precision numbers computed against this GT throughout the whole series.

## Key Takeaway
The model has a reasonably good sense of *where* an object roughly is (direction/location) but a much weaker sense of its *extent* (precise boundary) — "strong direction sense, weak boundary sense." This finding directly informed later use cases' framing of granularity effects.
