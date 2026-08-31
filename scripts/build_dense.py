import argparse
import time
import torch
from src.db.neo4j_client import Neo4jClient
from src.retrieval.entity_documents import (
    ENTITY_QUERY,
    entity_to_document,
)
from src.retrieval.dense_retriever import DenseRetriever


def parse_args():
    parser = argparse.ArgumentParser(description="Build Dense Vector index from Neo4j entities.")
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
        "--stream-batch-size",
        type=int,
        default=5000,
        help="Batch size when streaming from Neo4j.",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=128,
        help="Batch size for SentenceTransformer encoding.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the dense index pickle file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.all or (args.limit and args.limit <= 0):
        limit = None
        output_path = args.output or "indexes/dense/entities_all.pkl"
        query = ENTITY_QUERY
        print("Mode: Indexing ALL entities in the database...")
    else:
        limit = args.limit
        output_path = args.output or f"indexes/dense/entities_{limit // 1000}k.pkl" if limit >= 1000 else (args.output or f"indexes/dense/entities_{limit}.pkl")
        query = ENTITY_QUERY + f"\nLIMIT {limit}"
        print(f"Mode: Indexing {limit:,} entities...")

    print("Fetching entities from Neo4j...")
    start_time = time.perf_counter()
    documents = []

    with Neo4jClient() as db:
        for batch in db.stream_query(query, batch_size=args.stream_batch_size):
            for entity in batch:
                documents.append(entity_to_document(entity))
            print(f"  Loaded {len(documents):,} entities...", end="\r")

    fetch_time = time.perf_counter() - start_time
    print(f"\nFetched {len(documents):,} entities in {fetch_time:.2f}s")

    print(f"Loading SentenceTransformer ('all-MiniLM-L6-v2') on {device.upper()}...")
    retriever = DenseRetriever()
    retriever.model = retriever.model.to(device)

    print("Building dense embeddings...")
    build_start = time.perf_counter()
    retriever.build(documents, batch_size=args.encode_batch_size)
    build_time = time.perf_counter() - build_start
    print(f"\nDense index built in {build_time:.2f}s ({(len(documents) / build_time if build_time > 0 else 0):.1f} docs/sec)")

    retriever.save(output_path)
    print(f"[SUCCESS] Dense index saved to: {output_path}")


if __name__ == "__main__":
    main()
