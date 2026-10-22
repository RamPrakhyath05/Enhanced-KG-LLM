# ============================================================
# build_subgraph_index.py
# Creates BM25 docs and embeddings for every subgraph (partition).
# Saves to ./subg/
# ============================================================

import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

NEW_NEW = "/home/pesu-rf/Documents/Capstone_Team165/partition/new_new"
OUT_DIR = "/home/pesu-rf/Documents/Capstone_Team165/partition/subg"
MODEL_NAME = "all-MiniLM-L6-v2"
MAX_EMBED_CHARS = 5000  # max chars for embedding text per subgraph


def log(msg):
    print(f"[subg] {msg}", flush=True)


def main():
    # ---- Load partition assignment ----
    log("Loading partition assignment...")
    with open(f"{NEW_NEW}/membership_final.pkl", "rb") as f:
        graph_membership = list(pickle.load(f))
    with open(f"{NEW_NEW}/graph_node_ids.pkl", "rb") as f:
        node_ids = pickle.load(f)

    # Reconstruct capped assignment for non-graph entities
    log("Loading embeddings for centroid computation...")
    embeddings_full = np.load(f"{NEW_NEW}/entity_embeddings.npy")
    with open(f"{NEW_NEW}/entity_id_to_row.pkl", "rb") as f:
        entity_id_to_row = pickle.load(f)

    node_to_emb_row = np.array([entity_id_to_row[nid] for nid in node_ids])
    embeddings = embeddings_full[node_to_emb_row]

    # Compute centroids from graph members
    log("Computing centroids...")
    sums = defaultdict(lambda: np.zeros(embeddings.shape[1], dtype=np.float64))
    counts = defaultdict(int)
    for i, c in enumerate(graph_membership):
        sums[c] += embeddings[i]
        counts[c] += 1
    centroids = {c: s / counts[c] for c, s in sums.items()}
    target_comms = list(centroids.keys())
    target_matrix = np.array([centroids[c] for c in target_comms])
    target_matrix = target_matrix / (np.linalg.norm(target_matrix, axis=1, keepdims=True) + 1e-10)

    # Load all entity IDs
    log("Loading entities.csv...")
    entities_df = pd.read_csv(f"{NEW_NEW}/entities.csv")
    all_ids = entities_df["Entity_ID"].tolist()
    graph_set = set(node_ids)
    non_graph = [eid for eid in all_ids if eid not in graph_set]
    log(f"Non-graph entities: {len(non_graph):,}")

    # Reconstruct capped assignment
    log("Reconstructing capped assignment...")
    SIZE_CAP = 8000
    current_sizes = defaultdict(int)
    for c in graph_membership:
        current_sizes[c] += 1

    nn = NearestNeighbors(n_neighbors=min(100, len(target_comms)), metric="cosine")
    nn.fit(target_matrix)

    non_graph_rows = np.array([entity_id_to_row[eid] for eid in non_graph])
    non_graph_vecs = embeddings_full[non_graph_rows]

    full_assignment = {}  # entity_id -> partition_id
    for nid, c in zip(node_ids, graph_membership):
        full_assignment[nid] = int(c)

    chunk_size = 100000
    for chunk_start in tqdm(range(0, len(non_graph), chunk_size), desc="Assigning non-graph"):
        chunk_end = min(chunk_start + chunk_size, len(non_graph))
        chunk_vecs = non_graph_vecs[chunk_start:chunk_end]
        distances, indices = nn.kneighbors(chunk_vecs)
        for i in range(len(chunk_vecs)):
            eid = non_graph[chunk_start + i]
            for rank in range(indices.shape[1]):
                cand = target_comms[indices[i][rank]]
                if current_sizes[cand] < SIZE_CAP:
                    full_assignment[eid] = int(cand)
                    current_sizes[cand] += 1
                    break
            else:
                cand = target_comms[indices[i][0]]
                full_assignment[eid] = int(cand)
                current_sizes[cand] += 1

    del embeddings_full, non_graph_vecs

    # ---- Group entities by partition ----
    log("Grouping entities by partition...")
    partition_to_entities = defaultdict(list)
    for eid, pid in full_assignment.items():
        partition_to_entities[pid].append(eid)

    partition_ids = sorted(partition_to_entities.keys())
    log(f"Partitions: {len(partition_ids)}")

    # ---- Build BM25 docs and embedding texts ----
    log("Building BM25 docs and embedding texts...")
    # Create lookup from entity_id to row in entities_df
    eid_to_row = {eid: i for i, eid in enumerate(entities_df["Entity_ID"])}

    bm25_docs = []
    embed_texts = []

    for pid in tqdm(partition_ids, desc="Building docs"):
        eids = partition_to_entities[pid]
        rows = [eid_to_row[eid] for eid in eids]

        names = entities_df.iloc[rows]["Entity_Name"].fillna("").tolist()
        aliases = entities_df.iloc[rows]["Entity_Aliases"].fillna("").tolist()
        descs = entities_df.iloc[rows]["Entity_Description"].fillna("").tolist()

        # BM25 doc: name + aliases for every entity
        bm25_parts = []
        for name, alias in zip(names, aliases):
            parts = [name]
            if alias:
                parts.append(alias)
            bm25_parts.append(" ".join(parts))
        bm25_doc = " ".join(bm25_parts)
        bm25_docs.append(bm25_doc)

        # Embedding text: name + aliases + description, truncated
        embed_parts = []
        total_len = 0
        for name, alias, desc in zip(names, aliases, descs):
            part = name
            if alias:
                part += " " + alias
            if desc:
                part += " " + desc[:200]
            if total_len + len(part) > MAX_EMBED_CHARS:
                break
            embed_parts.append(part)
            total_len += len(part)
        embed_text = " ".join(embed_parts)
        embed_texts.append(embed_text)

    # ---- Compute embeddings ----
    log(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cuda")

    log(f"Computing embeddings for {len(embed_texts)} subgraphs...")
    subgraph_embeddings = model.encode(
        embed_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )
    subgraph_embeddings = subgraph_embeddings.astype(np.float32)
    log(f"Embeddings shape: {subgraph_embeddings.shape}")

    # ---- Save everything ----
    log("Saving to subg/...")
    np.save(f"{OUT_DIR}/subgraph_embeddings.npy", subgraph_embeddings)

    with open(f"{OUT_DIR}/subgraph_bm25_docs.pkl", "wb") as f:
        pickle.dump(bm25_docs, f)

    with open(f"{OUT_DIR}/subgraph_partition_ids.pkl", "wb") as f:
        pickle.dump(partition_ids, f)

    with open(f"{OUT_DIR}/subgraph_entity_map.pkl", "wb") as f:
        pickle.dump({pid: partition_to_entities[pid] for pid in partition_ids}, f)

    # ---- Also save a retrieval script ----
    log("DONE.")
    print(f"\nSaved to {OUT_DIR}/:")
    print(f"  subgraph_embeddings.npy  — shape {subgraph_embeddings.shape}")
    print(f"  subgraph_bm25_docs.pkl   — {len(bm25_docs)} docs")
    print(f"  subgraph_partition_ids.pkl — {len(partition_ids)} partition IDs")
    print(f"  subgraph_entity_map.pkl  — entity lists per partition")


if __name__ == "__main__":
    main()
