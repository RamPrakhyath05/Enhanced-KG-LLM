# ============================================================
# retrieve_subgraphs.py
# Query-time retrieval: embed query, BM25 score, combine with
# equal weight, return top-5 subgraphs.
# ============================================================

import pickle
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re

SUBG_DIR = "/home/pesu-rf/Documents/Capstone_Team165/partition/subg"
MODEL_NAME = "all-MiniLM-L6-v2"


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def log(msg):
    print(f"[retrieve] {msg}", flush=True)


def main():
    t_start = time.time()

    # Load index
    t0 = time.time()
    log("Loading subgraph index...")
    embeddings = np.load(f"{SUBG_DIR}/subgraph_embeddings.npy")
    with open(f"{SUBG_DIR}/subgraph_bm25_docs.pkl", "rb") as f:
        bm25_docs = pickle.load(f)
    with open(f"{SUBG_DIR}/subgraph_partition_ids.pkl", "rb") as f:
        partition_ids = pickle.load(f)
    with open(f"{SUBG_DIR}/subgraph_entity_map.pkl", "rb") as f:
        entity_map = pickle.load(f)
    log(f"Loaded {len(partition_ids)} subgraphs in {time.time() - t0:.2f}s")

    # Build BM25 index
    t0 = time.time()
    log("Building BM25 index...")
    tokenized_docs = [tokenize(doc) for doc in bm25_docs]
    bm25 = BM25Okapi(tokenized_docs)
    log(f"BM25 index built in {time.time() - t0:.2f}s")

    # Load embedding model
    t0 = time.time()
    log(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    log(f"Model loaded in {time.time() - t0:.2f}s")

    log(f"Total initialization time: {time.time() - t_start:.2f}s")

    # Interactive query loop
    log("Subgraph Retrieval Ready")

    while True:
        query = input("\nEnter query (or 'quit'): ").strip()
        if query.lower() == "quit":
            break
        if not query:
            continue

        t_query_start = time.time()

        # Embed query
        t0 = time.time()
        query_emb = model.encode([query], normalize_embeddings=True)[0]
        t_embed = time.time() - t0

        # BM25 score
        t0 = time.time()
        query_tokens = tokenize(query)
        bm25_scores = bm25.get_scores(query_tokens)
        t_bm25 = time.time() - t0

        # Embedding similarity
        t0 = time.time()
        emb_scores = embeddings @ query_emb

        # Normalize both to [0, 1]
        bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)
        emb_norm = (emb_scores - emb_scores.min()) / (emb_scores.max() - emb_scores.min() + 1e-10)

        # Equal weight combination
        combined = 0.5 * bm25_norm + 0.5 * emb_norm
        t_combine = time.time() - t0

        # Top 5
        t0 = time.time()
        top5_idx = np.argsort(combined)[-5:][::-1]
        t_top5 = time.time() - t0

        t_total = time.time() - t_query_start

        # Print results
        log(f"Top 5 subgraphs for '{query}':")
        for rank, idx in enumerate(top5_idx, 1):
            pid = partition_ids[idx]
            n_entities = len(entity_map[pid])
            log(f"  {rank}. partition_id={pid}  "
                f"entities={n_entities:,}  "
                f"bm25={bm25_norm[idx]:.4f}  "
                f"emb={emb_norm[idx]:.4f}  "
                f"combined={combined[idx]:.4f}")

        # Timing breakdown
        log("Timing:")
        log(f"  Embedding:     {t_embed:.4f}s")
        log(f"  BM25 scoring:  {t_bm25:.4f}s")
        log(f"  Combine+norm:  {t_combine:.4f}s")
        log(f"  Top-5 select:  {t_top5:.4f}s")
        log(f"  Total query:   {t_total:.4f}s")


if __name__ == "__main__":
    main()
