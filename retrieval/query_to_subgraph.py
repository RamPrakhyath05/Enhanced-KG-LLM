import json
import os
import pickle

import torch
from sentence_transformers import SentenceTransformer

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

CACHE_FILE = "edge_embeddings.pkl"

embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)


def edge_to_text(edge):
    src, rel, dst, props = edge
    prop_text = " ".join(f"{k} {v}" for k, v in props.items())
    return f"{src} {rel} {dst} {prop_text}"


def build_edge_cache(edges):
    print("Building edge embedding cache...")
    edge_texts = [edge_to_text(e) for e in edges]
    embeddings = embedder.encode(edge_texts, convert_to_tensor=True)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump((edges, embeddings), f)

    print("Cache saved!")
    return edges, embeddings


def load_edge_cache(edges):
    if os.path.exists(CACHE_FILE):
        print("Loading cached embeddings...")
        with open(CACHE_FILE, "rb") as f:
            cached_edges, embeddings = pickle.load(f)

        # Simple consistency check
        if len(cached_edges) == len(edges):
            return cached_edges, embeddings

    return build_edge_cache(edges)


def expand_subgraph(selected_edges, all_edges, hops=1):
    node_set = set()
    for s, _, t, _ in selected_edges:
        node_set.add(s)
        node_set.add(t)

    expanded = list(selected_edges)
    for _ in range(hops):
        new_edges = []
        for s, r, t, props in all_edges:
            if s in node_set or t in node_set:
                new_edges.append((s, r, t, props))
                node_set.add(s)
                node_set.add(t)

        expanded.extend(new_edges)

    # Deduplicate edges — dicts are unhashable so we can't use set() directly
    seen = set()
    unique = []
    for edge in expanded:
        s, r, t, props = edge
        key = (s, r, t, json.dumps(props, sort_keys=True, default=str))
        if key not in seen:
            seen.add(key)
            unique.append(edge)
    return unique


def extract_subgraph(query, edges, top_k=5):
    # Load or build cache.
    edges, edge_embs = load_edge_cache(edges)

    # Encode query.
    query_emb = embedder.encode(query, convert_to_tensor=True)

    # Fast similarity scoring.
    scores = torch.matmul(edge_embs, query_emb)

    # Top-K selection.
    k = min(top_k, len(edges))
    top_indices = torch.topk(scores, k=k).indices.tolist()
    selected = [edges[i] for i in top_indices]

    # Structure-aware expansion.
    subgraph = expand_subgraph(selected, edges, hops=1)
    return subgraph
