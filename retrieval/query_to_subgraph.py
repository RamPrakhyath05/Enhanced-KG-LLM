import hashlib
import json
import os
import pickle
from collections import Counter

import torch
from sentence_transformers import SentenceTransformer

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

CACHE_FILE = "edge_embeddings.pkl"

embedder = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

_QUERY_STOPWORDS = {
    "who", "what", "where", "when", "how", "why", "is", "are", "was", "were",
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "from", "by", "with",
    "and", "or", "list", "all", "any", "every", "show", "find", "give", "get",
}


def _compute_cache_hash(edges):
    """Compute a content-based hash for cache validation."""
    edge_strs = sorted(str(e[:3]) for e in edges)  # hash (src, rel, dst) only for speed
    return hashlib.md5("".join(edge_strs).encode()).hexdigest()


def edge_to_text(edge):
    """Convert an edge to a descriptive text string for embedding.

    Focuses on node names and relation name — skips verbose aliases
    and internal IDs that add noise to the embedding.
    """
    src, rel, dst, props = edge

    # Collect only semantically meaningful properties
    skip_keys = {"Relation_Aliases", "Relation_ID", "Relation_Name"}
    useful_props = {k: v for k, v in props.items() if k not in skip_keys}
    prop_text = " ".join(f"{k} {v}" for k, v in useful_props.items())

    text = f"{src} {rel} {dst}"
    if prop_text:
        text += f" {prop_text}"
    return text


def _query_keywords(query):
    tokens = []
    for token in query.lower().strip().rstrip("?!.").split():
        if len(token) > 2 and token not in _QUERY_STOPWORDS:
            tokens.append(token)
    return set(tokens)


def _is_intent_list_query(query):
    q = query.lower()
    return "list" in q or "what are" in q or "which are" in q


def _edge_matches_query_keywords(edge, keywords):
    if not keywords:
        return True
    src, rel, dst, _ = edge
    text = f"{src} {rel} {dst}".lower()
    return any(k in text for k in keywords)


def _keyword_overlap_score(edge, keywords):
    if not keywords:
        return 0.0
    src, rel, dst, _ = edge
    text = f"{src} {rel} {dst}".lower()
    matched = sum(1 for k in keywords if k in text)
    return matched / max(1, len(keywords))


def _list_relation_prior(rel):
    r = rel.lower()
    if "instance of" in r:
        return 0.40
    if "subclass of" in r:
        return 0.35
    if "has list" in r or "is a list of" in r:
        return 0.30
    if "part of" in r:
        return 0.10
    return 0.0


def build_edge_cache(edges):
    print("Building edge embedding cache...")
    edge_texts = [edge_to_text(e) for e in edges]
    embeddings = embedder.encode(edge_texts, convert_to_tensor=True)

    edge_hash = _compute_cache_hash(edges)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump((edges, embeddings, edge_hash), f)

    print(f"Cache saved! ({len(edges)} edges)")
    return edges, embeddings


def load_edge_cache(edges):
    if os.path.exists(CACHE_FILE):
        print("Loading cached embeddings...")
        with open(CACHE_FILE, "rb") as f:
            cached = pickle.load(f)

        # New format: (edges, embeddings, hash)
        if len(cached) == 3:
            cached_edges, embeddings, cached_hash = cached
            current_hash = _compute_cache_hash(edges)
            if cached_hash == current_hash:
                print(f"Cache valid ({len(cached_edges)} edges)")
                return cached_edges, embeddings
            else:
                print("Cache hash mismatch — rebuilding...")
        else:
            # Old format without hash — always rebuild
            print("Old cache format detected — rebuilding...")

    return build_edge_cache(edges)


def _compute_hub_nodes(all_edges, min_degree=30):
    """Identify generic hub nodes that should not be expanded through.

    A node is a hub if it appears in more than `min_degree` edges.
    Examples: 'Huamn' (500+ edges), 'united stated' (100+), 'Germanz' (50+).
    Specific entities like 'Blood Hound' (3 edges) are never classified as hubs.
    """
    degree = Counter()
    for s, _, t, _ in all_edges:
        degree[s] += 1
        degree[t] += 1

    return {node for node, cnt in degree.items() if cnt > min_degree}


def expand_subgraph(selected_edges, all_edges, hops=2, max_per_node=10, max_total=150):
    """Expand the selected edges by following neighboring edges.

    Seed nodes (from top-K) are always expandable even if they are hubs.
    New nodes discovered during expansion are subject to hub filtering.

    Args:
        selected_edges: The initial set of top-K edges.
        all_edges: All candidate edges.
        hops: Number of expansion hops.
        max_per_node: Max edges to add per node (prevents explosion).
        max_total: Max total expansion edges to add per hop.
    """
    hub_nodes = _compute_hub_nodes(all_edges)

    # Seed nodes from top-K are ALWAYS expandable — they're the query-relevant core.
    seed_nodes = set()
    for s, _, t, _ in selected_edges:
        seed_nodes.add(s)
        seed_nodes.add(t)

    node_set = set(seed_nodes)

    expanded = list(selected_edges)
    for _ in range(hops):
        node_edge_count = {n: 0 for n in node_set}
        new_edges = []
        for s, r, t, props in all_edges:
            if len(new_edges) >= max_total:
                break

            # Expandable = in the node set AND (is a seed OR not a hub) AND under per-node cap
            s_expandable = (
                s in node_set
                and (s in seed_nodes or s not in hub_nodes)
                and node_edge_count.get(s, 0) < max_per_node
            )
            t_expandable = (
                t in node_set
                and (t in seed_nodes or t not in hub_nodes)
                and node_edge_count.get(t, 0) < max_per_node
            )

            if s_expandable:
                new_edges.append((s, r, t, props))
                node_edge_count[s] = node_edge_count.get(s, 0) + 1
                node_set.add(t)
            elif t_expandable:
                new_edges.append((s, r, t, props))
                node_edge_count[t] = node_edge_count.get(t, 0) + 1
                node_set.add(s)

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


def extract_subgraph(query, edges, top_k=15, hops=2, max_per_node=10, max_total=150):
    """Extract a query-relevant subgraph from edges.

    Args:
        query: Natural language query string.
        edges: All candidate edges.
        top_k: Number of top semantically similar edges to select.
        hops: Number of expansion hops for context.
    """
    # Load or build cache.
    edges, edge_embs = load_edge_cache(edges)

    # For list-style queries, reduce expansion drift in very large graphs.
    if _is_intent_list_query(query):
        hops = min(hops, 1)

    # Encode query.
    query_emb = embedder.encode(query, convert_to_tensor=True)

    # Fast similarity scoring.
    scores = torch.matmul(edge_embs, query_emb)

    score_values = scores.detach().cpu().tolist()
    keywords = _query_keywords(query)
    is_list_query = _is_intent_list_query(query)

    ranked = []
    for i, edge in enumerate(edges):
        semantic = float(score_values[i])
        lexical = _keyword_overlap_score(edge, keywords)
        rel_prior = _list_relation_prior(edge[1]) if is_list_query else 0.0
        combined = semantic + (0.35 * lexical) + rel_prior
        ranked.append((combined, semantic, lexical, rel_prior, edge))

    ranked.sort(key=lambda x: x[0], reverse=True)
    ranked_edges = [item[4] for item in ranked]

    # Select a broader, relevance-sorted set for list queries.
    k = min(top_k, len(ranked_edges))
    selected = ranked_edges[:k]
    if is_list_query:
        # Keep high-confidence list answers and avoid expansion drift.
        filtered = []
        for _, semantic, lexical, rel_prior, edge in ranked[:k]:
            if lexical > 0.0 or rel_prior >= 0.30:
                filtered.append(edge)
            elif semantic >= ranked[min(k - 1, len(ranked) - 1)][1]:
                filtered.append(edge)
        return filtered or selected

    # Use similarity-ranked candidate order so capped expansion keeps
    # high-relevance neighbors first instead of insertion-order edges.
    subgraph = expand_subgraph(
        selected,
        ranked_edges,
        hops=hops,
        max_per_node=max_per_node,
        max_total=max_total,
    )
    return subgraph
