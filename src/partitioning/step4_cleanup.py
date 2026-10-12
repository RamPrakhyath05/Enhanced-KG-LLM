# ============================================================
# step4_cleanup.py
# Post-Leiden cleanup in 4 phases:
#   A) Split oversized communities (>5000) via k-means (target ~1200)
#   B) Iteratively merge small communities (<150) into their most
#      semantically similar neighbor (with growing size caps)
#   C) Merge truly isolated small communities (no external edges)
#      into the semantically nearest community
#   D) Final split of any community still >5000 (target ~1500)
# Input:  leiden_membership.pkl, graph_node_ids.pkl, weighted_graph.pkl,
#         entity_embeddings.npy, entity_id_to_row.pkl
# Output: membership_final.pkl (+ intermediates)
# ============================================================

import pickle
import numpy as np
import igraph as ig
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

MEMBERSHIP_PATH = "./leiden_membership.pkl"
NODE_LIST_PATH = "./graph_node_ids.pkl"
EMBEDDINGS_PATH = "./entity_embeddings.npy"
ID_MAP_PATH = "./entity_id_to_row.pkl"
GRAPH_PATH = "./weighted_graph.pkl"

MAX_SIZE = 5000
SPLIT_TARGET = 1200        # phase A
MIN_SIZE = 150             # phases B/C
CAPPED_PASSES = 3
BASE_CAP = 5000
CAP_GROWTH = 1500
MAX_PASSES = 30
K_NEIGHBORS = 10
FINAL_SPLIT_TARGET = 1500  # phase D


def log(msg):
    print(f"[cleanup] {msg}", flush=True)


def community_sizes(membership):
    sizes = defaultdict(int)
    for c in membership:
        sizes[c] += 1
    return sizes


def split_oversized(membership, node_ids, embeddings, entity_id_to_row, target_size):
    """Phase A/D: split communities > MAX_SIZE into k-means sub-clusters."""
    membership = list(membership)
    num_nodes = len(membership)
    node_to_emb_row = np.array([entity_id_to_row[nid] for nid in node_ids])

    sizes = community_sizes(membership)
    oversized = [c for c, s in sizes.items() if s > MAX_SIZE]
    log(f"  Splitting {len(oversized)} communities > {MAX_SIZE}")

    next_id = max(membership) + 1
    for comm_id in oversized:
        node_indices = [i for i in range(num_nodes) if membership[i] == comm_id]
        n_nodes = len(node_indices)
        k = max(2, round(n_nodes / target_size))

        log(f"  Splitting community {comm_id} ({n_nodes} nodes) into {k} sub-clusters")

        emb_rows = node_to_emb_row[node_indices]
        vecs = embeddings[emb_rows]

        km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, batch_size=1024)
        sub_labels = km.fit_predict(vecs)

        for local_idx, sub_label in zip(node_indices, sub_labels):
            if sub_label == 0:
                continue
            membership[local_idx] = next_id + sub_label - 1

        next_id += (k - 1)
    return membership


def merge_small(membership, g, node_ids, embeddings, entity_id_to_row):
    """Phase B: merge small communities (<MIN_SIZE) into most similar neighbor."""
    membership = list(membership)
    num_nodes = g.vcount()
    node_to_emb_row = np.array([entity_id_to_row[nid] for nid in node_ids])

    def compute_centroids(mem):
        sums = defaultdict(lambda: np.zeros(embeddings.shape[1], dtype=np.float64))
        counts = defaultdict(int)
        for i in range(num_nodes):
            c = mem[i]
            sums[c] += embeddings[node_to_emb_row[i]]
            counts[c] += 1
        return {c: s / counts[c] for c, s in sums.items()}, counts

    def build_adjacency(mem):
        adj = defaultdict(set)
        for edge in g.es:
            u, v = edge.source, edge.target
            cu, cv = mem[u], mem[v]
            if cu != cv:
                adj[cu].add(cv)
                adj[cv].add(cu)
        return adj

    for pass_num in range(MAX_PASSES):
        if pass_num < CAPPED_PASSES:
            cap = BASE_CAP + pass_num * CAP_GROWTH
        else:
            cap = None

        sizes = community_sizes(membership)
        small = [c for c, s in sizes.items() if s < MIN_SIZE]
        if not small:
            log(f"  Phase B converged at pass {pass_num} — no small communities left")
            break
        log(f"  Phase B pass {pass_num} (cap={cap}): {len(small)} small communities")

        adj = build_adjacency(membership)
        centroids, counts = compute_centroids(membership)

        small.sort(key=lambda c: sizes[c])

        remap = {}
        merges_done = 0

        for comm in small:
            if comm in remap:
                continue

            candidates = []
            for nb in adj.get(comm, set()):
                nb_resolved = remap.get(nb, nb)
                if cap is not None and counts.get(nb_resolved, 0) >= cap:
                    continue
                sim = np.dot(centroids[comm], centroids.get(nb_resolved, centroids[nb]))
                candidates.append((sim, nb_resolved))

            if not candidates:
                continue

            candidates.sort(reverse=True)
            target = candidates[0][1]

            remap[comm] = target
            counts[target] = counts.get(target, 0) + sizes[comm]
            merges_done += 1

        for i in range(num_nodes):
            c = membership[i]
            if c in remap:
                membership[i] = remap[c]

        log(f"  Phase B pass {pass_num}: {merges_done} communities merged")

        if merges_done == 0:
            log("  No merges possible — remaining are truly isolated. Stopping.")
            break

    return membership


def merge_isolated(membership, g, node_ids, embeddings, entity_id_to_row):
    """Phase C: merge isolated small communities into semantically nearest community."""
    membership = list(membership)
    num_nodes = len(membership)
    node_to_emb_row = np.array([entity_id_to_row[nid] for nid in node_ids])

    sizes = community_sizes(membership)

    has_external_edge = set()
    for edge in g.es:
        u, v = edge.source, edge.target
        cu, cv = membership[u], membership[v]
        if cu != cv:
            has_external_edge.add(cu)
            has_external_edge.add(cv)

    small = {c for c, s in sizes.items() if s < MIN_SIZE}
    isolated = small - has_external_edge
    log(f"  Phase C: {len(isolated)} truly isolated small communities")

    if not isolated:
        return membership

    # Centroids for all communities
    sums = defaultdict(lambda: np.zeros(embeddings.shape[1], dtype=np.float64))
    counts = defaultdict(int)
    for i in range(num_nodes):
        c = membership[i]
        sums[c] += embeddings[node_to_emb_row[i]]
        counts[c] += 1
    centroids = {c: s / counts[c] for c, s in sums.items()}

    target_comms = [c for c in centroids if c not in isolated]
    target_matrix = np.array([centroids[c] for c in target_comms])

    log(f"  Building nearest-neighbor index over {len(target_comms)} target communities...")
    nn = NearestNeighbors(n_neighbors=min(K_NEIGHBORS, len(target_comms)), metric="cosine")
    nn.fit(target_matrix)

    isolated_sorted = sorted(isolated, key=lambda c: sizes[c])
    isolated_matrix = np.array([centroids[c] for c in isolated_sorted])

    distances, indices = nn.kneighbors(isolated_matrix)

    remap = {}
    unresolved = 0

    for row_idx, comm in enumerate(isolated_sorted):
        comm_size = sizes[comm]
        placed = False
        for rank in range(indices.shape[1]):
            target = target_comms[indices[row_idx][rank]]
            if counts.get(target, 0) + comm_size <= MAX_SIZE:
                remap[comm] = target
                counts[target] = counts.get(target, 0) + comm_size
                placed = True
                break
        if not placed:
            unresolved += 1

    log(f"  Phase C: placed {len(remap)}, unresolved {unresolved}")

    for i in range(num_nodes):
        c = membership[i]
        if c in remap:
            membership[i] = remap[c]

    return membership


def main():
    log("Loading data...")
    g = ig.Graph.Read_Pickle(GRAPH_PATH)
    with open(NODE_LIST_PATH, "rb") as f:
        node_ids = pickle.load(f)
    with open(MEMBERSHIP_PATH, "rb") as f:
        membership = pickle.load(f)
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(ID_MAP_PATH, "rb") as f:
        entity_id_to_row = pickle.load(f)

    log(f"Graph: {g.summary()}")
    log(f"Initial communities: {len(set(membership))}")

    # Phase A
    log("Phase A: split oversized communities")
    membership = split_oversized(membership, node_ids, embeddings, entity_id_to_row, SPLIT_TARGET)
    log(f"After phase A: {len(set(membership))} communities")
    with open("./membership_after_split.pkl", "wb") as f:
        pickle.dump(membership, f)

    # Phase B
    log("Phase B: merge small communities into similar neighbors")
    membership = merge_small(membership, g, node_ids, embeddings, entity_id_to_row)
    log(f"After phase B: {len(set(membership))} communities")
    with open("./membership_merged.pkl", "wb") as f:
        pickle.dump(membership, f)

    # Phase C
    log("Phase C: merge isolated small communities by semantics")
    membership = merge_isolated(membership, g, node_ids, embeddings, entity_id_to_row)
    log(f"After phase C: {len(set(membership))} communities")
    with open("./membership_isolated_merged.pkl", "wb") as f:
        pickle.dump(membership, f)

    # Phase D
    log("Phase D: final split of remaining oversized communities")
    membership = split_oversized(membership, node_ids, embeddings, entity_id_to_row, FINAL_SPLIT_TARGET)

    sizes = community_sizes(membership)
    sizes_arr = np.array(list(sizes.values()))
    log(f"Final: {len(sizes)} communities")
    log(f"  max size: {sizes_arr.max()}")
    log(f"  <{MIN_SIZE}: {(sizes_arr < MIN_SIZE).sum()}")
    log(f"  >{MAX_SIZE}: {(sizes_arr > MAX_SIZE).sum()}")
    log(f"  percentiles [10,25,50,75,90,95,99]: {np.percentile(sizes_arr, [10, 25, 50, 75, 90, 95, 99])}")

    with open("./membership_final.pkl", "wb") as f:
        pickle.dump(membership, f)
    log("Saved to ./membership_final.pkl")
    log("DONE.")


if __name__ == "__main__":
    main()
