import os
import sys
import time
import argparse
from src.retrieval.partition_retriever import PartitionRetriever
from src.db.neo4j_client import Neo4jClient

FULL_INDEX_DIR = "indexes/partitions_full"
FULL_DATA_DIR = "data/partitions_full"
SAMPLE_INDEX_DIR = "indexes/partitions_10k"
SAMPLE_DATA_DIR = "data/partitions_10k"


def run_single_query(retriever, query: str, client=None, top_k: int = 3):
    """Executes end-to-end pipeline on a single query and displays results."""
    print("\n" + "=" * 75)
    print(f"QUERY: '{query}'")
    print("=" * 75)

    start_time = time.perf_counter()
    results = retriever.search(query, top_k=top_k, client=client)
    latency_ms = (time.perf_counter() - start_time) * 1000

    if not results:
        print("  [WARN] No matching partition found.")
        return

    print(f"End-to-End Retrieval Latency: {latency_ms:.2f} ms")
    print("-" * 75)

    for rank, res in enumerate(results, start=1):
        badge = "[PRIMARY TARGET PARTITION]" if rank == 1 else f"  [Candidate {rank}]"
        print(f"\n{badge}")
        print(f"  • Community Partition ID: #{res['community_id']}")
        print(f"  • RRF Fusion Score:        {res.get('rrf_score', res.get('score', 0.0)):.6f} (k=60)")
        
        if "vector_similarity" in res:
            print(f"  • Dense Cosine Similarity: {res['vector_similarity']:.4f}")
        if "bm25_score" in res:
            print(f"  • BM25 Lexical Score:      {res['bm25_score']:.2f}")
        if "size" in res:
            print(f"  • Subgraph Size:           {res['size']:,} entities")

        top_entities = res.get("top_entities", [])
        if top_entities:
            hubs_str = ", ".join([str(e) for e in top_entities[:5]])
            print(f"  • Key Entities in Cluster: {hubs_str}")

    print("-" * 75)


def main():
    parser = argparse.ArgumentParser(description="End-to-End 50% Milestone Retrieval Pipeline")
    parser.add_argument("--query", "-q", type=str, default=None, help="Single query to evaluate directly.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of candidate partitions to return (default: 3).")
    args = parser.parse_args()

    use_full = os.path.exists(os.path.join(FULL_INDEX_DIR, "community_centroids.npy"))
    index_dir = FULL_INDEX_DIR if use_full else SAMPLE_INDEX_DIR
    data_dir = FULL_DATA_DIR if use_full else SAMPLE_DATA_DIR

    print("=" * 75)
    print("END-TO-END QUERY-TO-PARTITION RETRIEVAL PIPELINE")
    print(f"Mode: {'FULL GRAPH (3.53M Nodes / v6 Partition)' if use_full else '10k BASELINE'}")
    print("=" * 75)

    # 1. Initialize Partition Retriever
    retriever = PartitionRetriever(rrf_k=60)
    print(f"Loading Partition Indexes from '{index_dir}'...")
    
    if use_full:
        retriever.load_full_scale(indexes_dir=index_dir, data_dir=data_dir)
    else:
        retriever.load(index_dir)

    # 2. Connect to Neo4j for live entity resolution
    client = None
    try:
        client = Neo4jClient()
        res = client.run_query("RETURN 1 AS ping")
        if res and res[0].get("ping") == 1:
            print("Connected to Neo4j successfully (Live Entity Resolution ACTIVE).")
    except Exception as e:
        print(f"Note: Neo4j live entity resolution disabled ({e}).")
        client = None

    # 3. Execution Modes
    if args.query:
        # Single CLI argument execution
        run_single_query(retriever, args.query, client=client, top_k=args.top_k)
    else:
        # Pure Interactive Terminal Mode
        print("\n" + "=" * 75)
        print("INTERACTIVE MODE: Type any query below to find its KG Partition.")
        print("Type 'exit', 'quit', or 'q' to stop.")
        print("=" * 75)

        while True:
            try:
                user_query = input("\nEnter Query > ").strip()
                if not user_query:
                    continue
                if user_query.lower() in ("exit", "quit", "q"):
                    print("\nExiting End-to-End Pipeline. Ready for Review 1!")
                    break

                run_single_query(retriever, user_query, client=client, top_k=args.top_k)

            except KeyboardInterrupt:
                print("\n\nSession terminated by user.")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")

    if client:
        try:
            client.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()