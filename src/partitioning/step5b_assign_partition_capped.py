# ============================================================
# step5b_assign_partition_capped.py
# Re-assigns partition_id to every Entity in Neo4j FinalKG.
# Graph members keep their Leiden community ID.
# Non-graph (hub-only) entities are assigned to their nearest
# centroid that hasn't exceeded a size cap, preventing
# mega-communities from forming.
# ============================================================

import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from neo4j import GraphDatabase
from tqdm import tqdm

MEMBERSHIP_PATH = "./membership_final.pkl"
NODE_LIST_PATH = "./graph_node_ids.pkl"
EMBEDDINGS_PATH = "./entity_embeddings.npy"
ID_MAP_PATH = "./entity_id_to_row.pkl"
ENTITIES_PATH = "./entities.csv"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "admin@123"
NEO4J_DATABASE = "FinalKG"

BATCH_SIZE = 10000
SIZE_CAP = 8000  # max entities per community (graph + assigned)


def log(msg):
    print(f"[step5b] {msg}", flush=True)


def main():
    log("Loading membership, node list, embeddings, ID map...")
    with open(MEMBERSHIP_PATH, "rb") as f:
        membership = pickle.load(f)
    with open(NODE_LIST_PATH, "rb") as f:
        node_ids = pickle.load(f)
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(ID_MAP_PATH, "rb") as f:
        entity_id_to_row = pickle.load(f)

    membership = list(membership)
    log(f"Graph nodes: {len(node_ids):,}, communities: {len(set(membership))}")

    # entity_id -> partition_id for graph members (unchanged)
    entity_partition = dict(zip(node_ids, [int(c) for c in membership]))

    # ---- Compute centroids from graph members only ----
    node_to_emb_row = np.array([entity_id_to_row[nid] for nid in node_ids])
    sums = defaultdict(lambda: np.zeros(embeddings.shape[1], dtype=np.float64))
    counts = defaultdict(int)
    for i, c in enumerate(membership):
        sums[c] += embeddings[node_to_emb_row[i]]
        counts[c] += 1
    centroids = {c: s / counts[c] for c, s in sums.items()}

    target_comms = list(centroids.keys())
    target_matrix = np.array([centroids[c] for c in target_comms])
    # Normalize for cosine similarity
    target_matrix = target_matrix / (np.linalg.norm(target_matrix, axis=1, keepdims=True) + 1e-10)

    # ---- Assign non-graph entities with size cap ----
    graph_set = set(node_ids)
    log("Loading all entity IDs from entities.csv...")
    all_ids = pd.read_csv(ENTITIES_PATH, usecols=["Entity_ID"])["Entity_ID"].tolist()
    non_graph = [eid for eid in all_ids if eid not in graph_set]
    log(f"Non-graph entities (hub-only connections): {len(non_graph):,}")

    # Current sizes = graph member counts
    current_sizes = defaultdict(int)
    for c in membership:
        current_sizes[c] += 1

    # Build NN index over centroids
    log(f"Building NN index over {len(target_comms)} community centroids...")
    nn = NearestNeighbors(n_neighbors=min(100, len(target_comms)), metric="cosine")
    nn.fit(target_matrix)

    log("Assigning non-graph entities to nearest centroid with capacity...")
    non_graph_rows = np.array([entity_id_to_row[eid] for eid in non_graph])
    non_graph_vecs = embeddings[non_graph_rows]

    # Process in chunks to show progress
    chunk_size = 100000
    overflow = 0
    for chunk_start in tqdm(range(0, len(non_graph), chunk_size), desc="Assigning"):
        chunk_end = min(chunk_start + chunk_size, len(non_graph))
        chunk_vecs = non_graph_vecs[chunk_start:chunk_end]
        chunk_ids = non_graph[chunk_start:chunk_end]

        distances, indices = nn.kneighbors(chunk_vecs)

        for i, eid in enumerate(chunk_ids):
            placed = False
            for rank in range(indices.shape[1]):
                cand_idx = indices[i][rank]
                cand_comm = target_comms[cand_idx]
                if current_sizes[cand_comm] < SIZE_CAP:
                    entity_partition[eid] = int(cand_comm)
                    current_sizes[cand_comm] += 1
                    placed = True
                    break
            if not placed:
                # All top-100 centroids full — assign to nearest anyway
                cand_comm = target_comms[indices[i][0]]
                entity_partition[eid] = int(cand_comm)
                current_sizes[cand_comm] += 1
                overflow += 1

    log(f"Overflow assignments (all top-100 full): {overflow:,}")

    # ---- Report size distribution ----
    final_sizes = defaultdict(int)
    for pid in entity_partition.values():
        final_sizes[pid] += 1
    sizes_arr = np.array(list(final_sizes.values()))
    log(f"Final: {len(final_sizes)} communities")
    log(f"  max size: {sizes_arr.max()}")
    log(f"  mean size: {sizes_arr.mean():.1f}")
    log(f"  median size: {np.median(sizes_arr):.0f}")
    log(f"  >{SIZE_CAP}: {(sizes_arr > SIZE_CAP).sum()}")

    log(f"Total entities with partition_id: {len(entity_partition):,}")

    # ---- Write to Neo4j FinalKG in batches ----
    log(f"Connecting to Neo4j {NEO4J_URI}, database {NEO4J_DATABASE}...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    pairs = list(entity_partition.items())
    total_updated = 0
    n_batches = (len(pairs) + BATCH_SIZE - 1) // BATCH_SIZE

    with driver.session(database=NEO4J_DATABASE) as session:
        for i in tqdm(range(0, len(pairs), BATCH_SIZE), desc="Writing to Neo4j"):
            batch = pairs[i:i + BATCH_SIZE]
            result = session.run(
                """
                UNWIND $pairs AS pair
                MATCH (n:Entity {Entity_ID: pair[0]})
                SET n.partition_id = pair[1]
                RETURN count(n) AS updated
                """,
                parameters={"pairs": [[eid, pid] for eid, pid in batch]}
            )
            updated = result.single()["updated"]
            total_updated += updated

    log(f"Total nodes updated in Neo4j: {total_updated:,}")

    # Verify
    with driver.session(database=NEO4J_DATABASE) as session:
        stats = session.run(
            """
            MATCH (e:Entity)
            RETURN count(e) AS total_nodes,
                   count(e.partition_id) AS nodes_with_partition,
                   count(DISTINCT e.partition_id) AS num_partitions
            """
        ).single()
        log(f"Verification — total: {stats['total_nodes']:,}, "
            f"with partition_id: {stats['nodes_with_partition']:,}, "
            f"distinct partitions: {stats['num_partitions']:,}")

        # Top 10 largest
        top10 = session.run(
            """
            MATCH (e:Entity)
            WITH e.partition_id AS pid, count(*) AS cnt
            RETURN pid, cnt ORDER BY cnt DESC LIMIT 10
            """
        ).data()
        log("Top 10 largest partitions:")
        for row in top10:
            log(f"  partition_id={row['pid']}: {row['cnt']:,} entities")

    driver.close()
    log("DONE.")


if __name__ == "__main__":
    main()
