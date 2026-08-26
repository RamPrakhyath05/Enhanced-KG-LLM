from src.retrieval.bm25_retriever import BM25Retriever


INDEX_PATH = "indexes/bm25/entities_10k.pkl"


def main():

    retriever = BM25Retriever()
    retriever.load(INDEX_PATH)

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

        results = retriever.search(query, top_k=5)

        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. "
                f"{result['name']} "
                f"({result['id']}) "
                f"[score={result['score']:.3f}]"
            )


if __name__ == "__main__":
    main()
