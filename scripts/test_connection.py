import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.db.neo4j_client import Neo4jClient

def main():
    with Neo4jClient() as client:
        result = client.run_query("MATCH (n) RETURN count(n) AS node_count")
        print("Node count:", result[0]["node_count"])

if __name__ == "__main__":
    main()
