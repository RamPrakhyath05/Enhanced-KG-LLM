# ============================================================
# step5_assign_partition.py
# Assigns partition_id to EVERY Entity node in Neo4j FinalKG.
#   - Entities in the structural graph get their Leiden community id.
#   - Entities outside the graph (only connected via hub relations)
#     get assigned to the semantically nearest community centroid.
# Input:  membership_final.pkl, graph_node_ids.pkl, entity_embeddings.npy,
#         entity_id_to_row.pkl, entities.csv, Neo4j connection
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


def log(msg):
    print(f"[step5] {msg}", flush=True)


def main():
    log("Loading membership, node list, embeddings, ID map...")
    with open(MEMBERSHIP_PATH, "rb") as f:
        membership = pickle.load(f)
    with open(NODE_LIST_PATH, "rb") as f:
        node_ids = pickle.load(f)
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(ID_MAP_PATH, "rb") as f:
        entity_id_to_row = pickle.load(f)

    log(f"Graph nodes: {len(node_ids):,}, communities: {len(set(membership))}")

    # entity_id -> partition_id for graph members
    entity_partition = dict(zip(node_ids, [int(c) for c in membership]))

    # ---- Assign non-graph entities to nearest community centroid ----
    graph_set = set(node_ids)
    log("Loading all entity IDs from entities.csv...")
    all_ids = pd.read_csv(ENTITIES_PATH, usecols=["Entity_ID"])["Entity_ID"].tolist()
    log(f"Total entities in KG: {len(all_ids):,}")

    non_graph = [eid for eid in all_ids if eid not in graph_set]
    log(f"Non-graph entities (hub-only connections): {len(non_graph):,}")

    if non_graph:
        log("Computing community centroids...")
        node_to_emb_row = np.array([entity_id_to_row[nid] for nid in node_ids])
        sums = defaultdict(lambda: np.zeros(embeddings.shape[1], dtype=np.float64))
        counts = defaultdict(int)
        for i, c in enumerate(membership):
            sums[c] += embeddings[node_to_emb_row[i]]
            counts[c] += 1
        centroids = {c: s / counts[c] for c, s in sums.items()}

        target_comms = list(centroids.keys())
        target_matrix = np.array([centroids[c] for c in target_comms])

        log(f"Building NN index over {len(target_comms)} community centroids...")
        nn = NearestNeighbors(n_neighbors=1, metric="cosine")
        nn.fit(target_matrix)

        log("Assigning non-graph entities to nearest community...")
        non_graph_rows = np.array([entity_id_to_row[eid] for eid in non_graph])
        non_graph_vecs = embeddings[non_graph_rows]
        _, indices = nn.kneighbors(non_graph_vecs)

        for eid, j in zip(non_graph, indices[:, 0]):
            entity_partition[eid] = int(target_comms[j])

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

    driver.close()
    log("DONE.")


if __name__ == "__main__":
    main()
