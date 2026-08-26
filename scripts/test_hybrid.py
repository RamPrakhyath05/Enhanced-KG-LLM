from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever

BM25_INDEX_PATH = "indexes/bm25/entities_10k.pkl"
DENSE_INDEX_PATH = "indexes/dense/entities_10k.pkl"

def main():

    # Load BM25 retriever
    bm25 = BM25Retriever()
    bm25.load(BM25_INDEX_PATH)

    # Load dense retriever
    dense = DenseRetriever()
    dense.load(DENSE_INDEX_PATH)

    # Create hybrid retriever
    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense
    )

    queries = [
        "American rock band",
        "investment bank Iceland",
        "Italian Renaissance painter",
        "Hong Kong actress fashion designer",
        "airport Sochi Russia",
    ]

    for query in queries:

        print("\n" + "=" * 60)
        print(f"QUERY: {query}")
        print("=" * 60)

        results = hybrid.search(
            query,
            top_k=5
        )

        for rank, result in enumerate(results, start=1):

            print(
                f"{rank}. "
                f"{result['name']} "
                f"({result['id']}) "
                f"[RRF={result['rrf_score']:.6f}]"
            )


if __name__ == "__main__":
    main()
