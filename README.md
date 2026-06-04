# VLM-Mobility

A hands-on learning project that explores **ten distinct roles** Vision-Language Models (VLMs) can play in autonomous driving and transportation research. Each use case uses the same real-world dataset — DriveLM-nuScenes — and progressively introduces a new VLM technique, from zero-shot inference to agentic perception-decision loops.

**Core dataset:** [DriveLM-nuScenes](https://huggingface.co/datasets/OpenDriveLab/DriveLM) — 696 driving scenes, 4,072 keyframes, 24,432 images across 6 cameras, ~378K structured QA pairs covering perception, prediction, planning, and behavior.

**Application domain:** Autonomous vehicle perception, scene understanding, driving decision support, and multi-agent coordination.

---

## Repository Structure

```
vlm-mobility/
├── datasets/
│   └── drivelm/               # DriveLM-nuScenes (primary dataset for all use cases)
│       ├── v1_1_train_nus.json
│       ├── nuscenes/samples/  # 24,432 camera images (6 cameras × 4,072 frames)
│       └── sample/            # Single-scene sample for quick testing
├── sourcecode/                # Python scripts — one per use case
│   ├── usecase1_*.py
│   ├── usecase2_*.py
│   └── ...
└── outputs/                   # Predictions, evaluation results, and figures
    ├── usecase1/
    ├── usecase2/
    └── ...
```

Script naming convention: `usecase{N}_{descriptor}.py` — each script is self-contained and maps to a subfolder under `outputs/`.

---

## Setup

**Environment:** Python 3.11, conda env `vlm-mobility`

**Core dependencies:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate peft trl
pip install datasets pillow opencv-python
pip install pandas numpy matplotlib scikit-learn
```

**Use-case-specific dependencies:**

| Use Case | Additional packages |
|---|---|
| UC7 — Multi-modal RAG | `faiss-cpu`, `sentence-transformers` |

**GPU:** A single GPU with 12 GB VRAM is sufficient for all scripts. Models ≤4B run in BF16; 7B+ models use 4-bit QLoRA. The vision encoder is frozen during fine-tuning to stay within memory limits.

---

## Ten Use Cases

### UC1 — Traffic Scene Understanding: Zero-Shot and Fine-Tuning

**Role:** Baseline evaluation of VLM perception capability

Zero-shot inference on DriveLM perception QA across multiple models (Qwen2.5-VL-7B, LLaMA-3.2-11B, LLaVA-OV-7B), followed by LoRA fine-tuning on the full DriveLM training set. Establishes the performance gap between off-the-shelf and domain-adapted VLMs, and introduces the core fine-tuning workflow.

---

### UC2 — Few-Shot Image-Text Prompting

**Role:** In-context learning with image examples

Constructs prompts that include 2–3 image-answer examples before the query image. Extends zero-shot prompting to harder QA types (prediction, planning) where output format and reasoning style benefit from demonstration. No model weights are updated — this is prompt engineering applied to multimodal inputs.

---

### UC3 — Spatial Grounding and Referring Expression

**Role:** Linking language to image locations

Evaluates whether VLMs can localize specific objects in a scene when referred to by natural language descriptions, and conversely, whether they can generate spatial references (camera, pixel coordinates) for named objects. Uses DriveLM's object reference format `<cN, CAM_XXX, x, y>` as ground truth.

---

### UC4 — Reliability and Hallucination Evaluation

**Role:** Systematic testing of VLM failure modes

Evaluates VLM outputs for factual consistency, object hallucination, and overconfidence. Tests whether models assert the presence of objects not visible in the image, contradict themselves across rephrased questions, or assign high confidence to wrong answers. Covers POPE-style binary probing, negation queries, and confidence calibration.

---

### UC5 — Geometric Reasoning

**Role:** Spatial and metric understanding from camera images

Probes whether VLMs can reason about geometric relationships — object distance estimation, relative depth ordering, lane occupancy, and bearing from the ego vehicle. Uses DriveLM's 3D annotations as ground truth to measure how well language-based reasoning approximates metric spatial understanding.

---

### UC6 — VLM as an Annotation Tool

**Role:** Automated label generation for unlabeled image data

Uses a VLM to generate structured annotations (object labels, spatial descriptions, scene summaries) for images without ground truth, then evaluates annotation quality against DriveLM labels. Demonstrates how VLMs can accelerate dataset construction — a practical workflow for research labs with large volumes of unannotated field imagery.

---

### UC7 — Multi-Modal RAG (Image + Text Retrieval)

**Role:** Retrieval-augmented visual question answering

Builds a searchable index over DriveLM frames using CLIP image embeddings (FAISS). At query time, retrieves the most visually similar historical frames and their QA pairs, then passes retrieved context plus the new image to a VLM for a grounded answer. Extends text-only RAG to include visual similarity search.

**Pipeline:** CLIP embedding → FAISS retrieval → Qwen2.5-VL generation

---

### UC8 — Video Understanding (Image Sequences)

**Role:** Temporal reasoning across consecutive frames

Treats the keyframe sequence within a DriveLM scene as a pseudo-video and asks cross-frame questions about object motion, scene change, and event progression. Explores how VLMs handle temporal context when multiple images are provided in a single prompt, and where single-frame reasoning breaks down.

---

### UC9 — VLM for Driving Decision-Making

**Role:** Image-to-action classification

Trains a VLM to map directly from camera images to discrete driving behavior labels (5 speed levels × 5 direction levels = 25 classes) using DriveLM's behavior annotations. Removes the question prompt — the model sees the image and outputs an action, treating driving as a vision-to-decision task rather than a QA task.

---

### UC10 — Agentic VLM: Perception–Decision System

**Role:** VLM embedded in a multi-agent observe–reason–act loop

Three agents — Perception, Prediction, and Planning — each receive a camera image and produce structured outputs that are passed sequentially to the next agent. The pipeline mirrors DriveLM's QA graph structure. Extends the text-based multi-agent framework (LLM series UC7) by introducing visual inputs at each agent step.

```
[Image] → Perception Agent  → "Brown SUV at back-left, moving"
                ↓
          Prediction Agent  → "SUV likely to change lane right"
                ↓
          Planning Agent    → "Maintain speed, monitor mirror"
```

---

## Project Navigation

```
Phase 1 — Core VLM mechanics
  UC1: Zero-shot inference and LoRA fine-tuning
  UC2: Few-shot image-text prompting

Phase 2 — Reliability and spatial understanding
  UC3: Spatial grounding and referring expressions
  UC4: Hallucination and reliability evaluation
  UC5: Geometric reasoning

Phase 3 — Advanced paradigms
  UC6: VLM as annotation tool
  UC7: Multi-modal RAG
  UC8: Video / sequence understanding
  UC9: Image-to-action decision-making
  UC10: Agentic perception–decision loop
```

All use cases share the same dataset and environment. Each is self-contained and can be run independently.

---

## Related Resources

- 🏠 **Smart Mobility Lab:** [sites.psu.edu/xbhu](https://sites.psu.edu/xbhu/)
- 📖 **Research Atlas:** [atlas.mobilitypsu.com](https://atlas.mobilitypsu.com)
- 🔗 **Companion project (LLM):** [LLM-EVPrediction](https://github.com/xbhu/llm-evprediction)

---

## License

Code: [MIT License](LICENSE)
