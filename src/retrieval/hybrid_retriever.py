class HybridRetriever:

    def __init__(self, bm25_retriever, dense_retriever, rrf_k=60):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k

    def search(self, query, top_k=10):

        bm25_results = self.bm25_retriever.search(
            query,
            top_k=top_k
        )

        dense_results = self.dense_retriever.search(
            query,
            top_k=top_k
        )

        scores = {}
        documents = {}

        # BM25 rankings
        for rank, result in enumerate(bm25_results, start=1):

            entity_id = result["id"]

            scores[entity_id] = scores.get(entity_id, 0.0) + (
                1.0 / (self.rrf_k + rank)
            )

            documents[entity_id] = result

        # Dense rankings
        for rank, result in enumerate(dense_results, start=1):

            entity_id = result["id"]

            scores[entity_id] = scores.get(entity_id, 0.0) + (
                1.0 / (self.rrf_k + rank)
            )

            documents[entity_id] = result

        ranked_entities = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        results = []

        for entity_id, rrf_score in ranked_entities[:top_k]:

            result = documents[entity_id].copy()

            result["rrf_score"] = rrf_score

            results.append(result)

        return results
