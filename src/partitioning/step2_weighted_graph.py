# ============================================================
# step2_weighted_graph.py
# Builds a weighted igraph from triples + embeddings.
# Edge weight = cosine similarity between entity embeddings.
# Hub relations (high-degree, low-discrimination) are filtered out.
# Input:  triples.csv, entity_embeddings.npy, entity_id_to_row.pkl
# Output: weighted_graph.pkl, graph_node_ids.pkl
# ============================================================

import pickle
import numpy as np
import pandas as pd
import igraph as ig
from tqdm import tqdm

TRIPLES_PATH = "./triples.csv"
EMBEDDINGS_PATH = "./entity_embeddings.npy"
ID_MAP_PATH = "./entity_id_to_row.pkl"
GRAPH_OUT_PATH = "./weighted_graph.pkl"
NODE_LIST_OUT_PATH = "./graph_node_ids.pkl"

CHUNK_SIZE = 500_000
HUB_RELATIONS = {"P31", "P17", "P27", "P131", "P19", "P20", "P21", "P106"}


def log(msg):
    print(f"[step2] {msg}", flush=True)


def main():
    log("Loading triples...")
    triples = pd.read_csv(TRIPLES_PATH)
    log(f"Triples: {len(triples):,}")

    log("Filtering hub relations...")
    structural_triples = triples[~triples["Relation_ID"].isin(HUB_RELATIONS)]
    log(f"Structural triples: {len(structural_triples):,}")

    log("Loading embeddings and ID map...")
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(ID_MAP_PATH, "rb") as f:
        entity_id_to_row = pickle.load(f)
    log(f"Embeddings shape: {embeddings.shape}")

    # Build node list
    log("Building node list...")
    all_nodes = pd.unique(pd.concat([structural_triples["Entity_1_ID"], structural_triples["Entity_2_ID"]]))
    log(f"Unique nodes: {len(all_nodes):,}")

    node_to_idx = {node: i for i, node in enumerate(all_nodes)}

    # Map triple endpoints to embedding rows
    log("Mapping triple endpoints to embedding rows...")
    src_emb_rows = structural_triples["Entity_1_ID"].map(entity_id_to_row).to_numpy()
    tgt_emb_rows = structural_triples["Entity_2_ID"].map(entity_id_to_row).to_numpy()

    src_emb_rows = pd.to_numeric(pd.Series(src_emb_rows), errors="coerce")
    tgt_emb_rows = pd.to_numeric(pd.Series(tgt_emb_rows), errors="coerce")
    n_missing = src_emb_rows.isna().sum() + tgt_emb_rows.isna().sum()
    if n_missing > 0:
        log(f"WARNING: {n_missing} endpoints missing from embedding map")
    src_emb_rows = src_emb_rows.fillna(0).astype(np.int64).to_numpy()
    tgt_emb_rows = tgt_emb_rows.fillna(0).astype(np.int64).to_numpy()

    # Compute cosine similarity per edge (embeddings are L2-normalized)
    log("Computing edge weights (cosine similarity)...")
    n_edges = len(structural_triples)
    weights = np.empty(n_edges, dtype=np.float32)

    for start in tqdm(range(0, n_edges, CHUNK_SIZE), desc="Weighting"):
        end = min(start + CHUNK_SIZE, n_edges)
        src_vecs = embeddings[src_emb_rows[start:end]]
        tgt_vecs = embeddings[tgt_emb_rows[start:end]]
        sims = np.sum(src_vecs * tgt_vecs, axis=1)
        weights[start:end] = sims

    log(f"Weight stats — min: {weights.min():.4f}, max: {weights.max():.4f}, "
        f"mean: {weights.mean():.4f}, median: {np.median(weights):.4f}")

    # Shift to positive
    min_w = weights.min()
    if min_w <= 0:
        shift = abs(min_w) + 1e-4
        weights = weights + shift
        log(f"Shifted weights by {shift:.4f}")

    # Apply w^2 transformation (emphasizes strong semantic ties)
    weights = weights ** 2
    log(f"Applied w^2. New stats — min: {weights.min():.4f}, max: {weights.max():.4f}, "
        f"mean: {weights.mean():.4f}")

    # Build edge list
    log("Building edge list...")
    src_idx = structural_triples["Entity_1_ID"].map(node_to_idx).to_numpy()
    tgt_idx = structural_triples["Entity_2_ID"].map(node_to_idx).to_numpy()
    edge_list = list(zip(src_idx.tolist(), tgt_idx.tolist()))

    log("Constructing igraph...")
    g = ig.Graph(n=len(all_nodes), edges=edge_list, directed=False)
    g.es["weight"] = weights.tolist()

    log("Simplifying graph...")
    g.simplify(combine_edges={"weight": "sum"})

    log(g.summary())

    # Save
    log("Saving graph...")
    g.write_pickle(GRAPH_OUT_PATH)
    with open(NODE_LIST_OUT_PATH, "wb") as f:
        pickle.dump(list(all_nodes), f)

    log(f"Saved graph to {GRAPH_OUT_PATH}")
    log(f"Saved node list to {NODE_LIST_OUT_PATH}")
    log("DONE.")


if __name__ == "__main__":
    main()
