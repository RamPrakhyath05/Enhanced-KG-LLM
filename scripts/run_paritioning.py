import argparse
import time
from src.db.neo4j_client import Neo4jClient
from src.preprocessing.graph_extractor import (
    extract_entities,
    extract_edges_for_entities,
    extract_all_edges,
)
from src.preprocessing.leiden_partitioner import LeidenPartitioner
from src.preprocessing.partition_storage import (
    save_partitions,
    write_partitions_to_neo4j,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Leiden Community Detection on Knowledge Graph.")
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Number of entities to partition.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Partition the entire knowledge graph.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Resolution parameter for Leiden algorithm (default 1.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save partition artifacts.",
    )
    parser.add_argument(
        "--write-to-neo4j",
        action="store_true",
        help="Write community_id property back to Neo4j :Entity nodes.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    is_full_graph = args.all or (args.limit and args.limit <= 0)
    limit = None if is_full_graph else args.limit

    if is_full_graph:
        output_dir = args.output_dir or "data/partitions_all"
        print("=" * 65)
        print("LEIDEN PARTITIONING: FULL GRAPH")
        print("=" * 65)
    else:
        output_dir = args.output_dir or (f"data/partitions_{limit // 1000}k" if limit >= 1000 else f"data/partitions_{limit}")
        print("=" * 65)
        print(f"LEIDEN PARTITIONING: {limit:,} ENTITY SUBGRAPH")
        print("=" * 65)

    with Neo4jClient() as client:
        print(f"\n[1/4] Extracting entities from Neo4j...")
        t0 = time.perf_counter()
        entities = extract_entities(client, limit=limit)
        t_entities = time.perf_counter() - t0
        print(f"  Extracted {len(entities):,} entities in {t_entities:.2f}s")

        print(f"\n[2/4] Extracting graph edges...")
        t0 = time.perf_counter()
        if is_full_graph:
            edges = extract_all_edges(client)
        else:
            edges = extract_edges_for_entities(client, set(entities.keys()))
        t_edges = time.perf_counter() - t0
        print(f"  Extracted {len(edges):,} edges connecting the entities in {t_edges:.2f}s")

        print(f"\n[3/4] Running Leiden Algorithm (resolution={args.resolution})...")
        t0 = time.perf_counter()
        partitioner = LeidenPartitioner(resolution=args.resolution)
        partition_map, community_metadata, modularity = partitioner.partition(entities, edges)
        t_leiden = time.perf_counter() - t0
        print(f"  Leiden completed in {t_leiden:.2f}s (Modularity = {modularity:.4f})")

        print(f"\n[4/4] Saving partition artifacts to {output_dir}...")
        save_partitions(partition_map, community_metadata, output_dir=output_dir)

        if args.write_to_neo4j:
            write_partitions_to_neo4j(client, partition_map)

    sizes = [c["size"] for c in community_metadata.values()]
    print("\n" + "=" * 65)
    print("PARTITIONING SUMMARY REPORT")
    print("=" * 65)
    print(f"Total Entities Partitioned:  {len(partition_map):,}")
    print(f"Total Communities Detected:  {len(community_metadata):,}")
    print(f"Graph Modularity Score:      {modularity:.4f}")
    print(f"Largest Community Size:      {max(sizes) if sizes else 0:,} entities")
    print(f"Smallest Community Size:     {min(sizes) if sizes else 0} entities")
    print(f"Average Community Size:      {sum(sizes)/len(sizes):.1f} entities")

    print("\nTop 5 Largest Communities:")
    sorted_communities = sorted(community_metadata.values(), key=lambda x: x["size"], reverse=True)
    for rank, comm in enumerate(sorted_communities[:5], start=1):
        names = [entities.get(eid, {}).get("name", eid) for eid in comm["top_entities"][:4]]
        print(f"  {rank}. Community #{comm['community_id']:04d} | Size: {comm['size']:,} | Hubs: {', '.join(names)}")

    print("=" * 65)
    print(f"[COMPLETE] Partition artifacts ready in: {output_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
