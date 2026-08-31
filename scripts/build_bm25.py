import argparse
import time
from src.db.neo4j_client import Neo4jClient
from src.retrieval.entity_documents import (
    ENTITY_QUERY,
    entity_to_document,
)
from src.retrieval.bm25_retriever import BM25Retriever


def parse_args():
    parser = argparse.ArgumentParser(description="Build BM25 index from Neo4j entities.")
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Number of entities to index.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Index all entities in the database.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Batch size when streaming from Neo4j.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the BM25 index pickle file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.all or (args.limit and args.limit <= 0):
        limit = None
        output_path = args.output or "indexes/bm25/entities_all.pkl"
        query = ENTITY_QUERY
        print("Mode: Indexing ALL entities in the database...")
    else:
        limit = args.limit
        output_path = args.output or f"indexes/bm25/entities_{limit // 1000}k.pkl" if limit >= 1000 else (args.output or f"indexes/bm25/entities_{limit}.pkl")
        query = ENTITY_QUERY + f"\nLIMIT {limit}"
        print(f"Mode: Indexing {limit:,} entities...")

    print("Fetching entities from Neo4j...")
    start_time = time.perf_counter()
    documents = []

    with Neo4jClient() as db:
        for batch in db.stream_query(query, batch_size=args.batch_size):
            for entity in batch:
                documents.append(entity_to_document(entity))
            print(f"  Loaded {len(documents):,} entities...", end="\r")

    fetch_time = time.perf_counter() - start_time
    print(f"\nFetched {len(documents):,} entities in {fetch_time:.2f}s")

    print("Building BM25 index...")
    build_start = time.perf_counter()
    retriever = BM25Retriever()
    retriever.build(documents)
    build_time = time.perf_counter() - build_start
    print(f"BM25 index built in {build_time:.2f}s")

    retriever.save(output_path)
    print(f"[SUCCESS] BM25 index saved to: {output_path}")


if __name__ == "__main__":
    main()
