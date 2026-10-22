#!/usr/bin/env python3
# ============================================================
# build_partition_index.py
#
# Builds a retrieval index for an ARBITRARY partition (membership)
# using the CENTROID + BM25 representation (NOT truncated text
# embeddings):
#
#   * per-partition centroid  = mean of its entity embeddings,
#                               L2-normalized  (vector ranking)
#   * per-partition BM25 doc  = concatenation of entity names +
#                               aliases, tokenized        (keyword ranking)
#
# Non-graph entities (hub-only) are assigned to the nearest centroid
# under a size cap, so the index covers ALL entities in the KG.
#
# Outputs (into --out-dir):
#   centroids.npy        (n_partitions, dim) float32
#   partition_ids.pkl    list of partition ids
#   entity_map.pkl       partition_id -> [entity_id, ...]
#   bm25_docs.pkl        list of tokenized docs
#   meta.json            build metadata
# ============================================================

import argparse
import json
import os
import pickle
import re
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def log(msg):
    print(f"[index] {msg}", flush=True)


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--membership", required=True)
    ap.add_argument("--node-ids", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--id-map", required=True)
    ap.add_argument("--entities", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", default="partition")
    ap.add_argument("--size-cap", type=int, default=8000)
    ap.add_argument("--assign-topk", type=int, default=100)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    log("Loading inputs...")
    with open(args.membership, "rb") as f:
        membership = list(pickle.load(f))
    with open(args.node_ids, "rb") as f:
        node_ids = list(pickle.load(f))
    embeddings = np.load(args.embeddings, mmap_mode="r")
    with open(args.id_map, "rb") as f:
        entity_id_to_row = pickle.load(f)
    entities = pd.read_csv(args.entities, usecols=["Entity_ID", "Entity_Name", "Entity_Aliases"])
    name_lookup = dict(zip(entities["Entity_ID"], entities["Entity_Name"].fillna("").astype(str)))
    alias_lookup = dict(zip(entities["Entity_ID"], entities["Entity_Aliases"].fillna("").astype(str)))

    log(f"Graph nodes: {len(node_ids):,}  embeddings: {embeddings.shape}")

    # ---- centroids from graph members (used to place non-graph entities) ----
    node_to_emb_row = np.array([entity_id_to_row[nid] for nid in node_ids])
    graph_emb = np.asarray(embeddings[node_to_emb_row])

    uniq = sorted(set(membership))
    pid_to_idx = {pid: i for i, pid in enumerate(uniq)}
    mem_idx = np.array([pid_to_idx[c] for c in membership])
    mem_list = mem_idx.tolist()          # indices into `uniq`, for ALL entities
    dim = graph_emb.shape[1]

    sums = np.zeros((len(uniq), dim), dtype=np.float64)
    np.add.at(sums, mem_idx, graph_emb)
    counts = np.bincount(mem_idx, minlength=len(uniq)).astype(np.float64)
    centroids = sums / counts[:, None]
    centroids /= (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)

    # ---- assign non-graph entities (hub-only) to nearest centroid ----
    graph_set = set(node_ids)
    all_ids = entities["Entity_ID"].tolist()
    non_graph = [eid for eid in all_ids if eid not in graph_set]
    log(f"Non-graph entities: {len(non_graph):,}")

    current_sizes = counts.copy()
    non_graph_rows = np.array([entity_id_to_row[eid] for eid in non_graph])
    non_graph_vecs = np.asarray(embeddings[non_graph_rows])
    overflow = 0
    n_centroids = len(uniq)

    if n_centroids > 5000:
        # Coarse-to-fine assignment: brute-force NN over >5k centroids is
        # O(n_query * n_centroids * d) and far too slow, so route each query
        # through a small set of coarse clusters first.
        from sklearn.cluster import MiniBatchKMeans
        n_coarse = 256
        log(f"  coarse-to-fine assignment ({n_coarse} coarse clusters)")
        centroids32 = centroids.astype(np.float32)
        km = MiniBatchKMeans(n_clusters=n_coarse, random_state=42,
                             batch_size=4096, n_init=3)
        km.fit(centroids32)
        coarse_labels = km.predict(non_graph_vecs)
        fine_of_coarse = defaultdict(list)
        for i, c in enumerate(km.labels_):
            fine_of_coarse[c].append(i)
        assigned = np.empty(len(non_graph), dtype=np.int64)
        for c in range(n_coarse):
            mask = coarse_labels == c
            if not mask.any():
                continue
            fine_idx = np.array(fine_of_coarse[c])
            sims = non_graph_vecs[mask] @ centroids32[fine_idx].T
            assigned[mask] = fine_idx[np.argmax(sims, axis=1)]
        for i in range(len(non_graph)):
            fi = int(assigned[i])
            if current_sizes[fi] < args.size_cap:
                mem_list.append(fi)
                current_sizes[fi] += 1
            else:
                mem_list.append(fi)
                current_sizes[fi] += 1
                overflow += 1
    else:
        nn = NearestNeighbors(n_neighbors=min(args.assign_topk, n_centroids), metric="cosine")
        nn.fit(centroids)
        chunk = 100000
        for start in range(0, len(non_graph), chunk):
            end = min(start + chunk, len(non_graph))
            _, idx = nn.kneighbors(non_graph_vecs[start:end])
            for i, eid in enumerate(non_graph[start:end]):
                placed = False
                for rank in range(idx.shape[1]):
                    cand = uniq[idx[i][rank]]
                    if current_sizes[pid_to_idx[cand]] < args.size_cap:
                        mem_list.append(pid_to_idx[cand])
                        current_sizes[pid_to_idx[cand]] += 1
                        placed = True
                        break
                if not placed:
                    cand = uniq[idx[i][0]]
                    mem_list.append(pid_to_idx[cand])
                    current_sizes[pid_to_idx[cand]] += 1
                    overflow += 1
    log(f"Assigned {len(non_graph):,} non-graph entities (overflow: {overflow:,})")

    # ---- group ALL entities by partition ----
    entity_map = {}
    for i, pid_idx in enumerate(mem_list):
        eid = node_ids[i] if i < len(node_ids) else non_graph[i - len(node_ids)]
        entity_map.setdefault(uniq[pid_idx], []).append(eid)

    partition_ids = sorted(entity_map.keys())
    log(f"Partitions: {len(partition_ids):,}  total entities: {sum(len(v) for v in entity_map.values()):,}")

    # ---- final centroids over ALL entities in each partition ----
    log("Computing final centroids over all entities...")
    final_centroids = np.zeros((len(partition_ids), dim), dtype=np.float32)
    for i, pid in enumerate(partition_ids):
        rows = np.fromiter((entity_id_to_row[e] for e in entity_map[pid]),
                           dtype=np.int64, count=len(entity_map[pid]))
        final_centroids[i] = np.asarray(embeddings[rows]).mean(axis=0)
    final_centroids /= (np.linalg.norm(final_centroids, axis=1, keepdims=True) + 1e-10)

    # ---- BM25 documents ----
    log("Building BM25 documents...")
    bm25_docs = []
    for pid in partition_ids:
        words = []
        for e in entity_map[pid]:
            nm = name_lookup.get(e)
            if nm:
                words.append(nm)
            al = alias_lookup.get(e)
            if al:
                words.extend(al.split("|"))
        bm25_docs.append(tokenize(" ".join(words)))

    # ---- save ----
    np.save(os.path.join(args.out_dir, "centroids.npy"), final_centroids)
    with open(os.path.join(args.out_dir, "partition_ids.pkl"), "wb") as f:
        pickle.dump(partition_ids, f)
    with open(os.path.join(args.out_dir, "entity_map.pkl"), "wb") as f:
        pickle.dump(entity_map, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.out_dir, "bm25_docs.pkl"), "wb") as f:
        pickle.dump(bm25_docs, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "label": args.label,
        "membership": args.membership,
        "n_graph_nodes": len(node_ids),
        "n_non_graph": len(non_graph),
        "n_partitions": len(partition_ids),
        "size_cap": args.size_cap,
        "overflow": overflow,
        "build_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"Saved index to {args.out_dir} in {time.time()-t0:.1f}s")
    log("DONE.")


if __name__ == "__main__":
    main()
