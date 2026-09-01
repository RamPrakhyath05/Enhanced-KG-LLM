import json
import os
import pickle
from typing import Dict
from src.db.neo4j_client import Neo4jClient


def save_partitions(
    partition_map: Dict[str, int],
    community_metadata: Dict[int, dict],
    output_dir: str = "data/partitions",
):
    os.makedirs(output_dir, exist_ok=True)

    map_file = os.path.join(output_dir, "partition_map.pkl")
    meta_file = os.path.join(output_dir, "communities.pkl")
    summary_file = os.path.join(output_dir, "summary.json")

    with open(map_file, "wb") as f:
        pickle.dump(partition_map, f)

    with open(meta_file, "wb") as f:
        pickle.dump(community_metadata, f)

    sizes = [c["size"] for c in community_metadata.values()]
    summary_data = {
        "total_entities": len(partition_map),
        "total_communities": len(community_metadata),
        "max_community_size": max(sizes) if sizes else 0,
        "min_community_size": min(sizes) if sizes else 0,
        "avg_community_size": sum(sizes) / len(sizes) if sizes else 0,
        "top_communities": [
            {
                "community_id": c["community_id"],
                "size": c["size"],
                "top_entities": c["top_entities"][:5],
            }
            for c in sorted(community_metadata.values(), key=lambda x: x["size"], reverse=True)[:10]
        ],
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"Partitions saved to: {output_dir}")
    print(f"  - {map_file}")
    print(f"  - {meta_file}")
    print(f"  - {summary_file}")


def load_partitions(input_dir: str = "data/partitions"):
    map_file = os.path.join(input_dir, "partition_map.pkl")
    meta_file = os.path.join(input_dir, "communities.pkl")

    if not os.path.exists(map_file) or not os.path.exists(meta_file):
        raise FileNotFoundError(f"Partition files not found in {input_dir}")

    with open(map_file, "rb") as f:
        partition_map = pickle.load(f)

    with open(meta_file, "rb") as f:
        community_metadata = pickle.load(f)

    return partition_map, community_metadata


def write_partitions_to_neo4j(
    client: Neo4jClient,
    partition_map: Dict[str, int],
    batch_size: int = 5000,
):
    items = list(partition_map.items())
    total = len(items)
    print(f"Writing {total:,} partition IDs to Neo4j nodes...")

    query = """
    UNWIND $batch AS item
    MATCH (e:Entity {Entity_ID: item.id})
    SET e.community_id = item.community_id
    """

    for i in range(0, total, batch_size):
        chunk = items[i : i + batch_size]
        batch_payload = [{"id": eid, "community_id": cid} for eid, cid in chunk]
        client.run_query(query, {"batch": batch_payload})
        print(f"  Updated {min(i + batch_size, total):,}/{total:,} nodes...", end="\r")

    print("\n[SUCCESS] Neo4j nodes updated with community_id.")
