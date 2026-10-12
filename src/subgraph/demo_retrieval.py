# ============================================================
# demo_retrieval.py
# Non-interactive demo: runs a few example queries through
# the retrieval pipeline and prints top-5 subgraphs.
# ============================================================

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re

SUBG_DIR = "/home/pesu-rf/Documents/Capstone_Team165/partition/subg"
MODEL_NAME = "all-MiniLM-L6-v2"


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def main():
    print("Loading subgraph index...")
    embeddings = np.load(f"{SUBG_DIR}/subgraph_embeddings.npy")
    with open(f"{SUBG_DIR}/subgraph_bm25_docs.pkl", "rb") as f:
        bm25_docs = pickle.load(f)
    with open(f"{SUBG_DIR}/subgraph_partition_ids.pkl", "rb") as f:
        partition_ids = pickle.load(f)
    with open(f"{SUBG_DIR}/subgraph_entity_map.pkl", "rb") as f:
        entity_map = pickle.load(f)
    print(f"Loaded {len(partition_ids)} subgraphs")

    print("Building BM25 index...")
    tokenized_docs = [tokenize(doc) for doc in bm25_docs]
    bm25 = BM25Okapi(tokenized_docs)

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cuda")

    queries = [
        "machine learning algorithms",
        "ancient Roman architecture",
        "quantum computing applications",
        "Shakespeare plays",
        "protein folding biology",
    ]

    for query in queries:
        # Embed query
        query_emb = model.encode([query], normalize_embeddings=True)[0]

        # BM25 score
        bm25_scores = bm25.get_scores(tokenize(query))

        # Embedding similarity
        emb_scores = embeddings @ query_emb

        # Normalize to [0, 1]
        bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)
        emb_norm = (emb_scores - emb_scores.min()) / (emb_scores.max() - emb_scores.min() + 1e-10)

        # Equal weight
        combined = 0.5 * bm25_norm + 0.5 * emb_norm

        top5_idx = np.argsort(combined)[-5:][::-1]

        print(f"\n{'=' * 70}")
        print(f"Query: '{query}'")
        print(f"{'=' * 70}")
        for rank, idx in enumerate(top5_idx, 1):
            pid = partition_ids[idx]
            n_entities = len(entity_map[pid])
            print(f"  {rank}. partition_id={pid:<8}  entities={n_entities:>6,}  "
                  f"bm25={bm25_norm[idx]:.4f}  emb={emb_norm[idx]:.4f}  combined={combined[idx]:.4f}")

    print(f"\n{'=' * 70}")
    print("Demo complete.")


if __name__ == "__main__":
    main()
