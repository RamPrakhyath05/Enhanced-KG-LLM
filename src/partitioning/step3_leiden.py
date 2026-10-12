# ============================================================
# step3_leiden.py
# Runs weighted Leiden partitioning (RBConfiguration) on the graph.
# Input:  weighted_graph.pkl
# Output: leiden_membership.pkl
# ============================================================

import pickle
import igraph as ig
import leidenalg

GRAPH_PATH = "./weighted_graph.pkl"
OUTPUT_MEMBERSHIP_PATH = "./leiden_membership.pkl"

RESOLUTION = 3.0
N_ITERATIONS = 2
SEED = 42


def log(msg):
    print(f"[step3] {msg}", flush=True)


def main():
    log("Loading graph...")
    g = ig.Graph.Read_Pickle(GRAPH_PATH)
    log(g.summary())

    log(f"Running Leiden (RBConfiguration, resolution={RESOLUTION})...")

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=RESOLUTION,
        weights=g.es["weight"],
        n_iterations=N_ITERATIONS,
        seed=SEED,
    )

    membership = partition.membership
    sizes = sorted([len(c) for c in partition], reverse=True)
    n_singletons = sum(1 for s in sizes if s == 1)

    log(f"Communities: {len(partition)}")
    log(f"Largest: {sizes[0]}")
    log(f"Singletons: {n_singletons} ({n_singletons / g.vcount() * 100:.1f}%)")
    log(f"Top 20 sizes: {sizes[:20]}")

    with open(OUTPUT_MEMBERSHIP_PATH, "wb") as f:
        pickle.dump(membership, f)
    log(f"Saved membership to {OUTPUT_MEMBERSHIP_PATH}")
    log("DONE.")


if __name__ == "__main__":
    main()
