from src.retrieval.dense_retriever import DenseRetriever

INDEX_PATH = "indexes/dense/entities_10k.pkl"


def main():
    retriever = DenseRetriever()
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
            print(f"{rank}. {result['name']} ({result['id']}) [score={result['score']:.4f}]")


if __name__ == "__main__":
    main()
