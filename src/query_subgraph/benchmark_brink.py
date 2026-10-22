# ============================================================
# benchmark_brink.py
# Full benchmarking of subgraph retrieval on BRINK-Wikidata5m.
#
# For each BRINK question:
#   1. Find the gold partition(s) containing the answer entity
#   2. Run retrieval (BM25 + embedding, equal weight)
#   3. Check if gold partition is in top-K
#   4. Report Hits@K, MRR, Recall@K
#
# Also runs ablations:
#   - No partitioning (retrieve from full KG)
#   - Random partitions
#   - Different K values
# ============================================================

import pickle
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from datasets import load_dataset
from collections import defaultdict
import os
import torch
import argparse
import re
import json

SUBG_DIR = os.getenv("SUBG_DIR", "data/subg")
MODEL_NAME = "all-MiniLM-L6-v2"
RESULTS_DIR = os.getenv("RESULTS_DIR", os.path.join(SUBG_DIR, "results"))


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def log(msg):
    print(f"[bench] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="BRINK-Wikidata5m Subgraph Retrieval Benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Number of test questions to evaluate (default: all)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Load subgraph index ----
    log("Loading subgraph index...")
    embeddings = np.load(f"{SUBG_DIR}/subgraph_embeddings.npy")
    with open(f"{SUBG_DIR}/subgraph_bm25_docs.pkl", "rb") as f:
        bm25_docs = pickle.load(f)
    with open(f"{SUBG_DIR}/subgraph_partition_ids.pkl", "rb") as f:
        partition_ids = pickle.load(f)
    with open(f"{SUBG_DIR}/subgraph_entity_map.pkl", "rb") as f:
        entity_map = pickle.load(f)
    log(f"Loaded {len(partition_ids)} subgraphs")

    # Build entity -> partition_id mapping
    log("Building entity -> partition mapping...")
    entity_to_partition = {}
    for pid, eids in entity_map.items():
        for eid in eids:
            entity_to_partition[eid] = pid
    log(f"Mapped {len(entity_to_partition):,} entities")

    # Build BM25 index
    log("Building BM25 index...")
    tokenized_docs = [tokenize(doc) for doc in bm25_docs]
    bm25 = BM25Okapi(tokenized_docs)

    # Load embedding model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Loading model: {MODEL_NAME} on {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # ---- Load BRINK dataset ----
    log("Loading BRINK-Wikidata5m...")
    ds = load_dataset("ZDZR/BRINK-Wikidata5m")
    test_data = ds["test"]
    if args.limit:
        test_data = test_data.select(range(min(args.limit, len(test_data))))
    log(f"Test set: {len(test_data)} questions")

    # ---- Build gold partition mapping ----
    # For each question, find which partition contains the answer entity
    log("Building gold partition mapping...")
    gold_partitions = []  # list of sets of partition_ids
    skipped = 0
    for item in test_data:
        answers = eval(item["answer"]) if isinstance(item["answer"], str) else item["answer"]
        partitions = set()
        for ans in answers:
            if ans in entity_to_partition:
                partitions.add(entity_to_partition[ans])
        if partitions:
            gold_partitions.append(partitions)
        else:
            gold_partitions.append(set())
            skipped += 1
    log(f"Questions with gold partition: {len(gold_partitions) - skipped:,} / {len(gold_partitions):,}")

    # ---- Run retrieval ----
    log("Running retrieval on test set...")
    all_scores = []
    retrieval_times = []

    for i, item in enumerate(test_data):
        query = item["question"]
        t0 = time.time()

        # Embed query
        query_emb = model.encode([query], normalize_embeddings=True)[0]

        # BM25 score
        bm25_scores = bm25.get_scores(tokenize(query))

        # Embedding similarity
        emb_scores = embeddings @ query_emb

        # Normalize
        bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)
        emb_norm = (emb_scores - emb_scores.min()) / (emb_scores.max() - emb_scores.min() + 1e-10)

        # Equal weight
        combined = 0.5 * bm25_norm + 0.5 * emb_scores

        # Store top-10
        top10_idx = np.argsort(combined)[-10:][::-1]
        top10_pids = [partition_ids[idx] for idx in top10_idx]
        all_scores.append(top10_pids)
        retrieval_times.append(time.time() - t0)

        if (i + 1) % 1000 == 0:
            log(f"  Processed {i + 1}/{len(test_data)} questions")

    log(f"Retrieval complete. Mean time: {np.mean(retrieval_times):.4f}s")

    # ---- Compute metrics ----
    log("Computing metrics...")
    results = {}
    for K in [1, 3, 5, 10]:
        hits = 0
        rr_sum = 0
        for i, gold in enumerate(gold_partitions):
            if not gold:
                continue
            top_k = set(all_scores[i][:K])
            if top_k & gold:  # intersection
                hits += 1
            # MRR
            for rank, pid in enumerate(all_scores[i][:K], 1):
                if pid in gold:
                    rr_sum += 1.0 / rank
                    break
        n_valid = sum(1 for g in gold_partitions if g)
        results[f"Hits@{K}"] = hits / n_valid
        results[f"MRR@{K}"] = rr_sum / n_valid

    # ---- Ablation: Random partitions ----
    log("Running random partition baseline...")
    np.random.seed(42)
    random_hits = {1: 0, 3: 0, 5: 0, 10: 0}
    for i, gold in enumerate(gold_partitions):
        if not gold:
            continue
        random_pids = np.random.choice(partition_ids, size=10, replace=False).tolist()
        for K in [1, 3, 5, 10]:
            if set(random_pids[:K]) & gold:
                random_hits[K] += 1
    n_valid = sum(1 for g in gold_partitions if g)
    for K in [1, 3, 5, 10]:
        results[f"Random_Hits@{K}"] = random_hits[K] / n_valid

    # ---- Ablation: Embedding only ----
    log("Running embedding-only baseline...")
    emb_hits = {1: 0, 3: 0, 5: 0, 10: 0}
    for i, item in enumerate(test_data):
        query = item["question"]
        query_emb = model.encode([query], normalize_embeddings=True)[0]
        emb_scores = embeddings @ query_emb
        top10_idx = np.argsort(emb_scores)[-10:][::-1]
        top10_pids = [partition_ids[idx] for idx in top10_idx]
        gold = gold_partitions[i]
        if not gold:
            continue
        for K in [1, 3, 5, 10]:
            if set(top10_pids[:K]) & gold:
                emb_hits[K] += 1
    for K in [1, 3, 5, 10]:
        results[f"EmbOnly_Hits@{K}"] = emb_hits[K] / n_valid

    # ---- Ablation: BM25 only ----
    log("Running BM25-only baseline...")
    bm25_hits = {1: 0, 3: 0, 5: 0, 10: 0}
    for i, item in enumerate(test_data):
        query = item["question"]
        bm25_scores = bm25.get_scores(tokenize(query))
        top10_idx = np.argsort(bm25_scores)[-10:][::-1]
        top10_pids = [partition_ids[idx] for idx in top10_idx]
        gold = gold_partitions[i]
        if not gold:
            continue
        for K in [1, 3, 5, 10]:
            if set(top10_pids[:K]) & gold:
                bm25_hits[K] += 1
    for K in [1, 3, 5, 10]:
        results[f"BM25Only_Hits@{K}"] = bm25_hits[K] / n_valid

    # ---- Print results ----
    log("=" * 60)
    log("BENCHMARK RESULTS — BRINK-Wikidata5m Test Set")
    log("=" * 60)
    log(f"Total questions: {len(test_data):,}")
    log(f"Questions with gold partition: {n_valid:,}")
    log(f"Mean retrieval time: {np.mean(retrieval_times):.4f}s")
    log("")
    log(f"{'Metric':<20} {'Value':>10}")
    log("-" * 30)
    for K in [1, 3, 5, 10]:
        log(f"{'Hits@' + str(K):<20} {results[f'Hits@{K}']:>10.4f}")
    for K in [1, 3, 5, 10]:
        log(f"{'MRR@' + str(K):<20} {results[f'MRR@{K}']:>10.4f}")
    log("")
    log("Ablations:")
    for K in [1, 3, 5, 10]:
        log(f"{'Random Hits@' + str(K):<20} {results[f'Random_Hits@{K}']:>10.4f}")
    for K in [1, 3, 5, 10]:
        log(f"{'EmbOnly Hits@' + str(K):<20} {results[f'EmbOnly_Hits@{K}']:>10.4f}")
    for K in [1, 3, 5, 10]:
        log(f"{'BM25Only Hits@' + str(K):<20} {results[f'BM25Only_Hits@{K}']:>10.4f}")

    # ---- Save results ----
    with open(f"{RESULTS_DIR}/brink_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"Results saved to {RESULTS_DIR}/brink_results.json")

    # ---- Save per-question results for analysis ----
    per_question = []
    for i, item in enumerate(test_data):
        per_question.append({
            "id": item["id"],
            "question": item["question"],
            "gold_partitions": list(gold_partitions[i]),
            "top10": all_scores[i],
        })
    with open(f"{RESULTS_DIR}/per_question_results.json", "w") as f:
        json.dump(per_question, f)
    log(f"Per-question results saved")

    log("DONE.")


if __name__ == "__main__":
    main()
