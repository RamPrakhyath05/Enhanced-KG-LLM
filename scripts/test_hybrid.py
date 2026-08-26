import time

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


BM25_INDEX_PATH = "indexes/bm25/entities_10k.pkl"
DENSE_INDEX_PATH = "indexes/dense/entities_10k.pkl"


QUERIES = [
    "American rock band",
    "investment bank Iceland",
    "Italian Renaissance painter",
    "Hong Kong actress fashion designer",
    "airport Sochi Russia",
]


def main():

    # ============================================================
    # LOAD RETRIEVERS
    # ============================================================

    print("Loading indexes...")

    load_start = time.perf_counter()

    bm25 = BM25Retriever()
    bm25.load(BM25_INDEX_PATH)

    dense = DenseRetriever()
    dense.load(DENSE_INDEX_PATH)

    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense
    )

    load_time = time.perf_counter() - load_start

    print(f"Index/model loading time: {load_time:.4f} seconds")

    # ============================================================
    # BENCHMARK
    # ============================================================

    print("\n" + "=" * 70)
    print("HYBRID RETRIEVAL BENCHMARK")
    print("=" * 70)

    total_bm25 = 0.0
    total_dense = 0.0
    total_hybrid = 0.0

    for query in QUERIES:

        print("\n" + "-" * 70)
        print(f"QUERY: {query}")
        print("-" * 70)

        # --------------------------------------------------------
        # BM25
        # --------------------------------------------------------

        start = time.perf_counter()

        bm25_results = bm25.search(
            query,
            top_k=5
        )

        bm25_time = time.perf_counter() - start

        # --------------------------------------------------------
        # Dense
        # --------------------------------------------------------

        start = time.perf_counter()

        dense_results = dense.search(
            query,
            top_k=5
        )

        dense_time = time.perf_counter() - start

        # --------------------------------------------------------
        # Hybrid / Weighted Score Fusion
        # --------------------------------------------------------

        start = time.perf_counter()

        hybrid_results = hybrid.search(
            query,
            top_k=5
        )

        hybrid_time = time.perf_counter() - start

        # --------------------------------------------------------
        # Accumulate timings
        # --------------------------------------------------------

        total_bm25 += bm25_time
        total_dense += dense_time
        total_hybrid += hybrid_time

        # --------------------------------------------------------
        # Timing results
        # --------------------------------------------------------

        print(f"\nBM25 retrieval:   {bm25_time * 1000:.3f} ms")
        print(f"Dense retrieval:  {dense_time * 1000:.3f} ms")
        print(f"Hybrid retrieval: {hybrid_time * 1000:.3f} ms")

        # --------------------------------------------------------
        # Hybrid results
        # --------------------------------------------------------

        print("\nHybrid Top-5:")

        for rank, result in enumerate(
            hybrid_results,
            start=1
        ):
            print(
                f"{rank}. "
                f"{result['name']} "
                f"({result['id']}) "
                f"[Hybrid Score={result['hybrid_score']:.6f}]"
            )

    # ============================================================
    # AVERAGES
    # ============================================================

    n = len(QUERIES)

    avg_bm25 = total_bm25 / n
    avg_dense = total_dense / n
    avg_hybrid = total_hybrid / n

    print("\n" + "=" * 70)
    print("AVERAGE RETRIEVAL LATENCY")
    print("=" * 70)

    print(f"BM25:    {avg_bm25 * 1000:.3f} ms")
    print(f"Dense:   {avg_dense * 1000:.3f} ms")
    print(f"Hybrid:  {avg_hybrid * 1000:.3f} ms")

    # ============================================================
    # SETUP
    # ============================================================

    print("\n" + "=" * 70)
    print("SETUP")
    print("=" * 70)

    print("Corpus:              10,000 entities")
    print("Dense model:         all-MiniLM-L6-v2")
    print("Embedding dimension: 384")
    print("BM25 weight:         0.5")
    print("Dense weight:        0.5")   
    print("Retrieval top-k:     5")
    print(f"Queries benchmarked: {n}")


if __name__ == "__main__":
    main()
