import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.db.neo4j_client import Neo4jClient
from src.db.queries import get_entity_by_id, search_entities_by_name, get_neighbors


def main():
    with Neo4jClient() as client:
        print("Search: 'Trump'")
        results = search_entities_by_name(client, "Trump", limit=5)
        for r in results:
            print(r)

        if results:
            first_id = results[0]["id"]

            print(f"\n--- Entity by ID: {first_id} ---")
            print(get_entity_by_id(client, first_id))

            print(f"\n--- Neighbors of {first_id} ---")
            for n in get_neighbors(client, first_id, limit=5):
                print(n)


if __name__ == "__main__":
    main()
