import sys
from config.settings import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
)
from src.db.neo4j_client import Neo4jClient


def test_connection():
    print("=" * 60)
    print("Testing Neo4j Connection")
    print("=" * 60)
    print(f"URI:      {NEO4J_URI}")
    print(f"User:     {NEO4J_USER}")
    print(f"Password: {'*' * len(NEO4J_PASSWORD) if NEO4J_PASSWORD else '(Not Set - check .env)'}")
    print(f"Database: {NEO4J_DATABASE}")
    print("-" * 60)

    try:
        with Neo4jClient() as client:
            res = client.run_query("RETURN 1 AS ping")
            if res and res[0].get("ping") == 1:
                print("[SUCCESS] Connected to Neo4j successfully!\n")

            print("Checking Graph Data...")
            node_count = client.run_query("MATCH (n) RETURN count(n) AS count")
            entity_count = client.run_query("MATCH (e:Entity) RETURN count(e) AS count")
            rel_count = client.run_query("MATCH ()-[r]->() RETURN count(r) AS count")

            total_nodes = node_count[0]["count"] if node_count else 0
            total_entities = entity_count[0]["count"] if entity_count else 0
            total_rels = rel_count[0]["count"] if rel_count else 0

            print(f"  - Total Nodes:        {total_nodes:,}")
            print(f"  - Total :Entity Nodes: {total_entities:,}")
            print(f"  - Total Relationships: {total_rels:,}")

            if total_entities > 0:
                print("\nFetching sample entity:")
                sample = client.run_query(
                    "MATCH (e:Entity) RETURN e.Entity_ID AS id, e.Entity_Name AS name, e.Entity_Description AS desc LIMIT 1"
                )
                if sample:
                    print(f"  Sample: {sample[0]}")

            print("=" * 60)
            print("Connection test completed successfully.")
            print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Failed to connect to Neo4j:\n{e}")
        sys.exit(1)


if __name__ == "__main__":
    test_connection()
