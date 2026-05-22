# VLM-Mobility

A hands-on learning project that explores **six distinct roles** Vision-Language Models (VLMs) can play in mobility and transportation-related studies. Each use case uses a real public dataset, working Python code, and output examples — making this a self-contained reference for researchers and practitioners who want to understand how VLMs fit into visual transportation problems.

**Core capability explored:** Given an image — traffic camera frame, work zone photo, satellite image — can a VLM understand the scene, answer structured questions about it, and produce actionable outputs for transportation operations?

**Application domains:** Traffic scene understanding, work zone safety, infrastructure classification, remote sensing, and multi-agent traffic management.

---

## Repository Structure

```
vlm-transportation/
├── datasets/          # Image lists, annotation CSVs, and prompt files for all 6 use cases
├── sourcecode/        # Python scripts — one per model per use case
│   ├── usecase1_qwen_vl.py
│   ├── usecase1_llava.py
│   └── ...
└── outputs/           # Generated predictions, evaluation results, and figures
    ├── usecase1_qwen_vl/
    ├── usecase2_internvl/
    └── ...
```

Script naming convention: `usecase{N}_{model}.py` — each script is self-contained and maps directly to a subfolder under `outputs/`.

---

## Setup

**Python:** 3.10+

**Core dependencies:**

```bash
pip install torch transformers datasets peft accelerate
pip install pandas numpy matplotlib scikit-learn
pip install Pillow opencv-python
```

**Use-case-specific dependencies:**

| Use Case | Additional packages |
|---|---|
| UC4 — Multi-modal RAG | `faiss-cpu`, `sentence-transformers` |
| UC5 — Remote Sensing | `rasterio`, `torchvision` |

**GPU:** A single GPU with 12 GB VRAM is sufficient for all scripts using 4-bit quantization. Models up to 4B parameters run in BF16 without quantization. The vision encoder is frozen during fine-tuning (UC3) to reduce memory pressure.

---

## Six Use Cases

### Use Case 1 — VLM as a Zero-Shot Traffic Scene Analyst

**Role:** Direct visual question answering on traffic camera images

**Core idea:** A VLM processes an image the same way it processes text — as a sequence of tokens. Just as a language model answers questions about a document, a VLM answers structured questions about a scene, with no task-specific training required.

```
Input:  Traffic camera frame
           ↓
[VLM: "How many vehicles? What is the congestion level?
       Any visible incidents?"]
           ↓
Output: {"vehicle_count": 12, "congestion": "high",
         "incident": "none", "visibility": "clear"}
```

**Research angle:** *Zero-Shot Visual Scene Understanding for Traffic Operations* — evaluate how much domain knowledge about road conditions and congestion is encoded in general-purpose VLMs; identify failure modes before fine-tuning.

**Dataset:** [BDD100K](https://bdd-data.berkeley.edu/) — 100,000 driving video frames with annotations
- Annotations include weather, scene type, time of day, and object bounding boxes
- Subset used: 1,000 images sampled across congestion levels and weather conditions
- Labels provide ground truth for zero-shot evaluation

**Models explored:**

| Model | Developer | Parameters | Architecture | Key Characteristics |
|---|---|---|---|---|
| LLaVA-1.5-7B | Haotian Liu / CMU | 7B | CLIP ViT-L + Vicuna | Most widely cited open-source VLM baseline; architecture is maximally transparent for learning |
| Qwen2.5-VL-3B-Instruct | Alibaba | 3B | Qwen ViT + Qwen2.5 | Lightweight; strong instruction following; fits in BF16 on 12 GB GPU |
| Qwen2.5-VL-7B-Instruct | Alibaba | 7B | Qwen ViT + Qwen2.5 | Dynamic resolution tiling for high-detail images; requires 4-bit quantization for training |
| LLaMA-3.2-11B-Vision-Instruct | Meta | 11B | Vision encoder + LLaMA 3.2 | Mature ecosystem; consistent API with text-only LLaMA; 4-bit quantization required |

> **Note:** UC1 is the calibration step for the entire project. Running multiple models here builds intuition for the gap between zero-shot capability and domain-specific performance — which directly motivates the fine-tuning in UC3.

---

### Use Case 2 — VLM as a Work Zone Scene Descriptor

**Role:** Structured safety assessment from work zone photographs

**Core idea:** Feed a VLM an image of a highway work zone. It identifies safety-relevant elements — attenuator truck position, cone placement, worker proximity to traffic — and outputs a structured safety report.

```
Input:  Work zone photograph
           ↓
[VLM: "Is a truck-mounted attenuator present?
       Are cones correctly spaced?
       Are workers within the protected zone?"]
           ↓
Output: "ATMA: present. Cone spacing: irregular (gap near chainage 420).
         Workers: 2 detected, both within protected zone.
         Risk level: medium."
```

**Research angle:** *VLM-Assisted Work Zone Safety Inspection* — evaluate whether off-the-shelf VLMs can detect safety compliance violations; directly applicable to ATMA deployment verification and automated field audit tools.

**Dataset:** [MUTCD Work Zone Image Collection](https://mutcd.fhwa.dot.gov/) + [SHRP2 Naturalistic Driving Study](https://highways.dot.gov/research/research-programs/safety/naturalistic-driving-study) work zone subsets — publicly available through FHWA
- Images include lane closures, rolling slowdowns, and stationary work operations
- Annotations: equipment type, cone layout category, presence of workers

**Models explored:**

| Model | Developer | Parameters | Approach | Key Characteristics |
|---|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | Alibaba | 7B | Zero-shot structured prompting | Dynamic resolution handles the wide aspect ratios common in roadside camera feeds |
| InternVL2-8B | Shanghai AI Lab | 8B | Zero-shot + few-shot (2-shot) | High-resolution tiling; strong at counting and spatial localization within images |
| LLaVA-1.5-7B | CMU | 7B | Zero-shot | Baseline comparison; simpler architecture, clearer failure modes |

> **Note:** This use case directly connects to the ATMA research line. A key finding to look for is whether VLMs can reliably distinguish *present-but-non-compliant* setups from *absent* equipment — a distinction that matters for safety audits.

---

### Use Case 3 — VLM Fine-Tuning for Transportation Image Classification

**Role:** Domain-adapted visual classifier via parameter-efficient fine-tuning

**Core idea:** General-purpose VLMs perform well on common visual tasks but struggle with domain-specific distinctions — differentiating a longitudinal buffer zone from a lateral protection setup, or classifying pavement distress types. Fine-tuning with LoRA on a small labeled set closes this gap.

```
Zero-shot VLM:   "pavement damage"          (correct category, low precision)
Fine-tuned VLM:  "transverse cracking, severity level 2"  (domain-precise)
```

**Research angle:** *Parameter-Efficient VLM Adaptation for Transportation Infrastructure Inspection* — quantify how much labeled data is needed to achieve acceptable classification accuracy; compare LoRA-only vs. full fine-tuning of the language head.

**Dataset:** [RoadDamage Dataset 2022](https://github.com/sekilab/RoadDamageDetector) — 26,000+ road damage images from Japan, India, and the Czech Republic
- 8 damage categories: longitudinal cracking, transverse cracking, alligator cracking, pothole, etc.
- Labels include damage type and severity
- Used here as an image classification task (damage type prediction)

**Models explored:**

| Model | Developer | Parameters | Fine-tuning Method | Key Characteristics |
|---|---|---|---|---|
| Qwen2.5-VL-3B-Instruct | Alibaba | 3B | LoRA (vision encoder frozen) | Fits in BF16; fast fine-tuning cycle; good for iterating on data size experiments |
| InternVL2-4B | Shanghai AI Lab | 4B | LoRA (vision encoder frozen) | Strong spatial understanding baseline before fine-tuning |

> **Note:** The vision encoder is frozen throughout — only the language model's LoRA adapters are trained. This mirrors the standard approach from LLaVA's original training protocol and keeps GPU memory within 12 GB. Unfreezing the encoder is explored as an ablation.

---

### Use Case 4 — Multi-Modal RAG for Transportation Queries

**Role:** Retrieval-augmented visual question answering

**Core idea:** When a user submits a new image and a question, the system retrieves visually and semantically similar historical cases from a knowledge base, then passes both the retrieved context and the new image to a VLM for a grounded answer.

```
Input:  New pavement damage photo + "Does this require immediate repair?"
           ↓
Retrieve: 3 most similar historical images + their maintenance decisions
           ↓
[VLM + retrieved context]
           ↓
Output: "Similar to cases from 2022-08 and 2023-03 (both Level 2 cracking).
         Both were scheduled within 30 days. Recommend same."
```

**Research angle:** *Multi-Modal RAG for Infrastructure Maintenance Decision Support* — extends text-only RAG (explored in LLM UC5–6) to include visual similarity search; demonstrates how a maintenance history database can be made queryable.

**Dataset:** RoadDamage Dataset 2022 (same as UC3) used as the retrieval knowledge base
- CLIP embeddings used to index all images
- Query set: 100 held-out images not in the knowledge base
- Retrieval evaluated by whether the top-k results share the same damage category

**Pipeline components:**

| Component | Tool / Model | Role |
|---|---|---|
| Visual embedding | `openai/clip-vit-base-patch32` | Convert images to vectors for similarity search |
| Vector index | FAISS | Fast approximate nearest-neighbor retrieval |
| Generator | Qwen2.5-VL-7B-Instruct | Synthesize retrieved context + query image into a final answer |

> **Note:** This use case is the visual analog of LLM Use Case 5 (RAG for anomaly explanation). The key new concept is image embedding for retrieval — CLIP maps images and text into a shared vector space, enabling cross-modal retrieval ("find images similar to this damage description").

---

### Use Case 5 — VLM for Remote Sensing and Infrastructure Analysis

**Role:** Aerial and satellite image interpretation for transportation planning

**Core idea:** Satellite imagery captures infrastructure at a scale that ground-level cameras cannot — parking lot occupancy, road construction progress, flood extent, and regional traffic density. VLMs with high-resolution tiling can interpret these images without task-specific detectors.

```
Input:  Satellite image of highway interchange + "Is construction ongoing?
        Estimate the percentage of the interchange affected."
           ↓
[InternVL2 with dynamic high-res tiling]
           ↓
Output: "Active construction detected in the northeast quadrant (~30% of interchange).
         Equipment: 3 excavators, 1 crane. Lane closure visible on ramp segment."
```

**Research angle:** *VLM-Based Remote Sensing for Transportation Infrastructure Monitoring* — evaluate high-resolution tiling strategies for satellite images; compare against traditional object detection baselines on the same imagery.

**Dataset:** [DOTA v2.0](https://captain-whu.github.io/DOTA/dataset.html) — 11,268 aerial images with 18 object categories including vehicles, bridges, and storage tanks
- Resolution: 800×800 to 4000×4000 pixels per image
- Transportation-relevant categories used: large vehicles, small vehicles, roundabouts, bridges
- Task framing: visual counting + scene description rather than bounding box detection

**Models explored:**

| Model | Developer | Parameters | Key Characteristics |
|---|---|---|---|
| InternVL2-8B | Shanghai AI Lab | 8B | Dynamic high-resolution tiling up to 4K; strongest open-source VLM for remote sensing tasks; 4-bit QLoRA for training |
| Qwen2.5-VL-7B-Instruct | Alibaba | 7B | Also supports dynamic tiling; comparison baseline for high-res handling |

> **Note:** Remote sensing is where the choice of VLM architecture matters most. Models without high-resolution tiling (e.g., standard LLaVA-1.5) resize images to 336×336, losing the detail that makes satellite imagery useful. InternVL2 and Qwen2.5-VL both handle this through tile-based processing.

---

### Use Case 6 — VLM-Powered Multi-Agent Traffic Management

**Role:** Visual perception integrated into multi-agent decision-making

**Core idea:** Extends the text-based multi-agent framework (LLM Use Case 7) by giving agents visual perception. Each agent receives an image of its domain — field camera, network diagram, demand heatmap — and reasons jointly about visual evidence and operational constraints.

```
Agent A (Field Monitor):   Receives work zone camera frame
                           → "Lane 2 blocked, queue forming ~500m upstream"
Agent B (Control Center):  Receives traffic density map image
                           → "Reroute via SR-45; capacity available"
Agent C (Driver Advisory): Receives incident map image
                           → "Expect 12-min delay; alternate route recommended"
              ↓
         Visual evidence + text negotiation
              ↓
         Coordinated incident response plan
```

**Research angle:** *Vision-Language Multi-Agent Systems for Real-Time Traffic Incident Management* — extends LLM multi-agent work to include visual grounding; connects to connected and automated vehicle (CAV) and infrastructure-to-vehicle (I2V) research directions.

**Dataset:** BDD100K (same as UC1) for field camera frames + [OpenStreetMap](https://www.openstreetmap.org/) rendered traffic network tiles for map-based agents
- 50 simulated incident scenarios constructed from BDD100K event frames
- Each scenario provides a different image to each agent role

**Models explored:**

| Setup | Model | Key Characteristics |
|---|---|---|
| Homogeneous agents | Qwen2.5-VL-7B-Instruct (all roles) | Single model plays all roles sequentially; consistent reasoning style |
| Heterogeneous agents | Qwen2.5-VL-3B (field) + InternVL2-8B (control) | Larger model for the decision-making role; lighter model for perception |

> **Note:** As in LLM UC7, multi-agent simulation does not require multiple model instances — a single model plays all roles sequentially with role-specific system prompts. The key new element is that each agent's prompt now includes an image in addition to text context. This tests whether VLMs can maintain role coherence when visual evidence and textual instructions conflict.

---

## Dataset Overview

| # | Dataset | Source | Size | Visual Content | VLM Role |
|---|---|---|---|---|---|
| 1 | BDD100K | UC Berkeley | 100K images | Driving frames | Scene analyst |
| 2 | MUTCD / SHRP2 work zone subset | FHWA | ~500 images | Work zone photos | Safety inspector |
| 3 | RoadDamage Dataset 2022 | Sekilab / Tohoku | 26K images | Pavement damage | Classifier |
| 4 | RoadDamage Dataset 2022 | Sekilab / Tohoku | 26K images (index) | Pavement damage | RAG retriever |
| 5 | DOTA v2.0 | CAPTAIN Group | 11K aerial images | Aerial / satellite | Remote sensing analyst |
| 6 | BDD100K + OSM tiles | UC Berkeley / OSM | 50 scenarios | Mixed visual inputs | Agent perception |

All datasets are publicly available for research use. Download instructions and directory placement are documented in `datasets/README.md` for each use case.

---

## How to Navigate This Project

The six use cases are organized into three phases of increasing complexity:

```
Phase 1 — Core VLM mechanics
  Use Case 1: Zero-shot scene understanding      (BDD100K; multiple models)
  Use Case 2: Domain-specific scene description  (Work zone safety; few-shot)

Phase 2 — Learning and retrieval
  Use Case 3: Fine-tuning with LoRA              (Road damage classification)
  Use Case 4: Multi-modal RAG pipeline           (Visual retrieval + generation)

Phase 3 — Advanced applications
  Use Case 5: High-resolution remote sensing     (DOTA; InternVL2)
  Use Case 6: Visual multi-agent coordination    (Incident management)
```

Each phase builds on the previous one, but all use cases are self-contained — you can start anywhere depending on your research interest.

---

## Background

This project uses transportation visual data as a concrete application domain to explore a wide range of VLM techniques. The six use cases are organized around real public datasets and real operational problems — traffic monitoring, work zone safety, pavement inspection, infrastructure surveillance, and incident response.

The framework is intentionally modular: each use case addresses a different research question, uses a different dataset and VLM paradigm, and can be extended independently. Use Cases 1–6 progressively introduce the key technical concepts: zero-shot inference, few-shot prompting, LoRA fine-tuning, RAG with visual embeddings, high-resolution tiling, and multi-agent visual grounding.

This project complements [LLM-EVPrediction](https://github.com/xbhu/llm-evprediction), which covers the same progression of techniques applied to text-based EV charging demand forecasting.

---

## Related Resources

- 🏠 **Smart Mobility Lab:** [sites.psu.edu/xbhu](https://sites.psu.edu/xbhu/)
- 📖 **Research Atlas:** [atlas.mobilitypsu.com](https://atlas.mobilitypsu.com)
- 🔗 **Companion project (LLM):** [LLM-EVPrediction](https://github.com/xbhu/llm-evprediction)

---

## License

Code: [MIT License](LICENSE)
