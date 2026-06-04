"""
UC7a: Planning RAG Pipeline
Retrieval: sentence-transformers encodes planning questions into text embeddings to retrieve semantically similar frames
Only three types of real planning questions are used: target_action / safe_actions / dangerous_actions
3 questions per frame, 27 inference pairs total (zero-shot vs RAG)
"""

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DRIVELM_JSON  = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm/v1_1_train_nus.json"
IMAGE_ROOT    = "/home/xzh5180/Research/vlm-mobility/datasets/drivelm"
EMBED_CACHE   = "/home/xzh5180/Research/vlm-mobility/outputs/uc7_planning_embeddings.npy"
META_CACHE    = "/home/xzh5180/Research/vlm-mobility/outputs/uc7_planning_meta.json"
OUTPUT_FILE   = "/home/xzh5180/Research/vlm-mobility/outputs/usecase7a_planning_rag_qwen25vl_drivelm.json"

SBERT_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
VLM_MODEL_ID  = "Qwen/Qwen2.5-VL-7B-Instruct"

TOP_K          = 3
MAX_NEW_TOKENS = 256

TARGET_KEYWORDS = ["target action", "safe actions", "dangerous actions"]

EVAL_FRAMES = {
    "f0f120e4d4b0441da90ec53b16ee169d": ["4a0798f849ca477ab18009c3a20b7df2",
                                          "ffd1bdf020d145759224c629b501d2b2",
                                          "d9075c2a5f864a2b8abf41e703f4cf1c"],
    "54cdaaae372d421fa4734d66f51a8c48": ["542eaf1fc9b34895a9e55fab57cb4cf4",
                                          "1b45a97a0e5e49fe9cd345dd4bd729c3",
                                          "d5e16062410f4e329d31a881b28e5c1c"],
    "1977a1c98a6c4eb79fbc2a6dc0da9b0f": ["bd8a5e326b804b069d497d29dbf19c2b",
                                          "7903e67446c64958b0a660f10bdadf19",
                                          "b6bf5a2bcb094969ace1023f8fe0b9e2"],
}

# ─── IMPORTS ──────────────────────────────────────────────────────────────────
import os, json, time
from collections import defaultdict
import numpy as np
import torch
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sentence_transformers import SentenceTransformer
from qwen_vl_utils import process_vision_info
from rouge_score import rouge_scorer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ─── DATA HELPERS ─────────────────────────────────────────────────────────────

def load_drivelm(path):
    with open(path) as f:
        return json.load(f)

def resolve_image_path(raw_path, image_root):
    p = raw_path.replace("../nuscenes/", "nuscenes/")
    return os.path.join(image_root, p)

def get_cam_front_image(frame_data, image_root):
    image_paths = frame_data.get("image_paths", {})
    img_path = image_paths.get("CAM_FRONT", "")
    if img_path:
        return resolve_image_path(img_path, image_root)
    return None

def get_q_type(q):
    q_lower = q.lower()
    if "target action" in q_lower:
        return "target_action"
    elif "safe actions" in q_lower:
        return "safe_actions"
    elif "dangerous actions" in q_lower:
        return "dangerous_actions"
    return "other"

def get_target_planning_qas(frame_data):
    """Retrieve only three types of real planning questions, excluding coordinate placeholder types."""
    qas = []
    for qa in frame_data.get("QA", {}).get("planning", []):
        q = qa.get("Q", "").strip()
        a = qa.get("A", "").strip()
        if any(k in q.lower() for k in TARGET_KEYWORDS) and a:
            qas.append((q, a))
    return qas

def get_all_planning_qas(frame_data):
    """For building the index: index only three types of real planning questions."""
    qas = []
    for qa in frame_data.get("QA", {}).get("planning", []):
        q = qa.get("Q", "").strip()
        a = qa.get("A", "").strip()
        if any(k in q.lower() for k in TARGET_KEYWORDS) and a:
            qas.append((q, a))
    return qas

# ─── BUILD INDEX ──────────────────────────────────────────────────────────────

def build_planning_index(data, sbert_model, embed_cache, meta_cache):
    print("[Index] Collecting all target planning QA...")
    records = []
    for scene_id, scene_data in data.items():
        for frame_token, frame_data in scene_data.get("key_frames", {}).items():
            for q, a in get_all_planning_qas(frame_data):
                records.append({
                    "scene_id": scene_id,
                    "frame_token": frame_token,
                    "q": q,
                    "a": a,
                    "q_type": get_q_type(q),
                })

    total = len(records)
    unique_qs = set(r["q"] for r in records)
    print(f"[Index] Total: {total}, unique Q: {len(unique_qs)} (duplicate rate {1 - len(unique_qs)/total:.1%})")

    questions = [r["q"] for r in records]
    embeddings = sbert_model.encode(
        questions, batch_size=256, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True
    )
    np.save(embed_cache, embeddings.astype(np.float32))
    with open(meta_cache, "w") as f:
        json.dump(records, f)
    print(f"[Index] Cached: {embed_cache}")
    return embeddings, records

def load_planning_index(embed_cache, meta_cache):
    embeddings = np.load(embed_cache).astype(np.float32)
    with open(meta_cache) as f:
        records = json.load(f)
    return embeddings, records

# ─── RETRIEVAL ────────────────────────────────────────────────────────────────

def retrieve_top_k(query_embedding, all_embeddings, all_records, top_k, exclude_scene_id):
    q = query_embedding.flatten()
    sims = all_embeddings @ q
    results = []
    for idx in np.argsort(sims)[::-1]:
        rec = all_records[idx]
        if rec["scene_id"] == exclude_scene_id:
            continue
        results.append({
            "rank": len(results) + 1,
            "scene_id": rec["scene_id"],
            "frame_token": rec["frame_token"],
            "retrieved_q": rec["q"],
            "retrieved_a": rec["a"],
            "similarity": round(float(sims[idx]), 4),
        })
        if len(results) >= top_k:
            break
    return results

# ─── VLM INFERENCE ────────────────────────────────────────────────────────────

def build_zero_shot_messages(image_path, question):
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text",  "text": question},
        ],
    }]

def build_rag_messages(image_path, question, retrieved):
    context_lines = []
    for r in retrieved:
        context_lines.append(
            f"[Reference {r['rank']} | sim={r['similarity']:.3f}]\n"
            f"Q: {r['retrieved_q']}\n"
            f"A: {r['retrieved_a']}"
        )
    context_text = "\n\n".join(context_lines)
    rag_prompt = (
        f"Here are planning QA examples from similar driving scenes:\n\n"
        f"{context_text}\n\n"
        f"Now answer the following question based on what you observe in the current image:\n"
        f"{question}"
    )
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text",  "text": rag_prompt},
        ],
    }]

def run_inference(messages, model, processor, max_new_tokens):
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to("cuda")
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

# ─── EVALUATION ───────────────────────────────────────────────────────────────

def compute_rouge(pred, ref):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s = scorer.score(ref, pred)
    return {k: round(v.fmeasure, 4) for k, v in s.items()}

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("UC7a: Planning RAG Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading DriveLM...")
    data = load_drivelm(DRIVELM_JSON)
    print(f"  Scenes: {len(data)}")

    # 2. Load SBERT
    print("\n[2/5] Loading SBERT model...")
    sbert = SentenceTransformer(SBERT_MODEL)

    # 3. Build index or load cache
    os.makedirs(os.path.dirname(EMBED_CACHE), exist_ok=True)
    if os.path.exists(EMBED_CACHE) and os.path.exists(META_CACHE):
        print(f"\n[3/5] Loading cached index...")
        all_embeddings, all_records = load_planning_index(EMBED_CACHE, META_CACHE)
        unique_qs = set(r["q"] for r in all_records)
        print(f"  Total: {len(all_records)}, unique Q: {len(unique_qs)} (duplicate rate {1 - len(unique_qs)/len(all_records):.1%})")
    else:
        print(f"\n[3/5] Building planning embedding index...")
        all_embeddings, all_records = build_planning_index(
            data, sbert, EMBED_CACHE, META_CACHE
        )

    # 4. Load Qwen2.5-VL
    print("\n[4/5] Loading Qwen2.5-VL-7B-Instruct (4-bit)...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
    model.eval()
    print("  VLM loaded.")

    # 5. Evaluation
    print("\n[5/5] Starting evaluation (9 frames × 3 questions × 2 inferences)...")
    results = []

    for scene_id, frame_tokens in EVAL_FRAMES.items():
        frames_dict = data.get(scene_id, {}).get("key_frames", {})

        for frame_token in frame_tokens:
            print(f"\n  scene={scene_id[:8]}  frame={frame_token[:8]}")
            frame_data = frames_dict.get(frame_token)
            if frame_data is None:
                print("  [WARN] frame_token not found, skipping")
                continue

            img_path = get_cam_front_image(frame_data, IMAGE_ROOT)
            if not img_path or not os.path.exists(img_path):
                print(f"  [WARN] Image not found, skipping")
                continue

            target_qas = get_target_planning_qas(frame_data)
            if not target_qas:
                print("  [WARN] No target planning QA, skipping")
                continue

            for query_q, gt_a in target_qas:
                q_type = get_q_type(query_q)
                print(f"  [{q_type}]")

                # Zero-shot
                zs_msgs = build_zero_shot_messages(img_path, query_q)
                t0 = time.time()
                zs_output = run_inference(zs_msgs, model, processor, MAX_NEW_TOKENS)
                zs_time = round(time.time() - t0, 1)
                zs_rouge = compute_rouge(zs_output, gt_a)
                print(f"    ZS  ROUGE-1={zs_rouge['rouge1']:.3f}  ROUGE-L={zs_rouge['rougeL']:.3f}  ({zs_time}s)")

                # Retrieve
                q_emb = sbert.encode([query_q], normalize_embeddings=True)
                retrieved = retrieve_top_k(
                    q_emb, all_embeddings, all_records,
                    top_k=TOP_K, exclude_scene_id=scene_id
                )

                # RAG
                rag_msgs = build_rag_messages(img_path, query_q, retrieved)
                t0 = time.time()
                rag_output = run_inference(rag_msgs, model, processor, MAX_NEW_TOKENS)
                rag_time = round(time.time() - t0, 1)
                rag_rouge = compute_rouge(rag_output, gt_a)
                delta_r1 = round(rag_rouge["rouge1"] - zs_rouge["rouge1"], 4)
                delta_rL = round(rag_rouge["rougeL"] - zs_rouge["rougeL"], 4)
                print(f"    RAG ROUGE-1={rag_rouge['rouge1']:.3f}  ROUGE-L={rag_rouge['rougeL']:.3f}  Δ={delta_r1:+.3f}  ({rag_time}s)")
                print(f"    Top-1 sim={retrieved[0]['similarity']:.3f}  A={retrieved[0]['retrieved_a'][:50]}")

                results.append({
                    "scene_id":      scene_id,
                    "frame_token":   frame_token,
                    "image_path":    img_path,
                    "q_type":        q_type,
                    "query_question": query_q,
                    "gt_answer":     gt_a,
                    "zero_shot":     {"output": zs_output, "rouge": zs_rouge, "time_s": zs_time},
                    "rag":           {"output": rag_output, "rouge": rag_rouge, "time_s": rag_time,
                                      "context": retrieved},
                    "delta":         {"rouge1": delta_r1, "rougeL": delta_rL},
                })

    # Summary
    if results:
        by_type = defaultdict(list)
        for r in results:
            by_type[r["q_type"]].append(r)

        summary_by_type = {}
        for qt, rs in by_type.items():
            zs_r1  = round(np.mean([r["zero_shot"]["rouge"]["rouge1"] for r in rs]), 4)
            zs_rL  = round(np.mean([r["zero_shot"]["rouge"]["rougeL"] for r in rs]), 4)
            rag_r1 = round(np.mean([r["rag"]["rouge"]["rouge1"] for r in rs]), 4)
            rag_rL = round(np.mean([r["rag"]["rouge"]["rougeL"] for r in rs]), 4)
            summary_by_type[qt] = {
                "n": len(rs),
                "zero_shot": {"avg_rouge1": zs_r1, "avg_rougeL": zs_rL},
                "rag":       {"avg_rouge1": rag_r1, "avg_rougeL": rag_rL},
                "delta":     {"avg_rouge1": round(rag_r1 - zs_r1, 4),
                              "avg_rougeL": round(rag_rL - zs_rL, 4)},
            }

        avg_zs_r1  = round(np.mean([r["zero_shot"]["rouge"]["rouge1"] for r in results]), 4)
        avg_zs_rL  = round(np.mean([r["zero_shot"]["rouge"]["rougeL"] for r in results]), 4)
        avg_rag_r1 = round(np.mean([r["rag"]["rouge"]["rouge1"] for r in results]), 4)
        avg_rag_rL = round(np.mean([r["rag"]["rouge"]["rougeL"] for r in results]), 4)
        avg_sim    = round(np.mean([r["rag"]["context"][0]["similarity"] for r in results]), 4)

        summary = {
            "n_total": len(results),
            "overall": {
                "zero_shot": {"avg_rouge1": avg_zs_r1, "avg_rougeL": avg_zs_rL},
                "rag":       {"avg_rouge1": avg_rag_r1, "avg_rougeL": avg_rag_rL},
                "delta":     {"avg_rouge1": round(avg_rag_r1 - avg_zs_r1, 4),
                              "avg_rougeL": round(avg_rag_rL - avg_zs_rL, 4)},
                "avg_top1_similarity": avg_sim,
            },
            "by_type": summary_by_type,
        }

        print("\n" + "=" * 40)
        print("SUMMARY")
        print(f"  Overall ({len(results)} pairs):")
        print(f"    Zero-shot : ROUGE-1={avg_zs_r1}  ROUGE-L={avg_zs_rL}")
        print(f"    RAG       : ROUGE-1={avg_rag_r1}  ROUGE-L={avg_rag_rL}")
        print(f"    Delta     : ROUGE-1={avg_rag_r1-avg_zs_r1:+.4f}  ROUGE-L={avg_rag_rL-avg_zs_rL:+.4f}")
        print(f"    Avg top-1 sim: {avg_sim}")
        print()
        for qt, st in summary_by_type.items():
            print(f"  [{qt}] n={st['n']}")
            print(f"    ZS  ROUGE-L={st['zero_shot']['avg_rougeL']}  RAG ROUGE-L={st['rag']['avg_rougeL']}  Δ={st['delta']['avg_rougeL']:+.4f}")
    else:
        summary = {}
        print("\n[WARN] No frames were successfully processed")

    output = {
        "experiment": "UC7a",
        "config": {"top_k": TOP_K, "sbert_model": SBERT_MODEL,
                   "vlm_model": VLM_MODEL_ID, "max_new_tokens": MAX_NEW_TOKENS},
        "summary": summary,
        "frame_results": results,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
