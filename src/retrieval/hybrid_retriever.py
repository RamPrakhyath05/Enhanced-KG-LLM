from concurrent.futures import ThreadPoolExecutor
class HybridRetriever:

    def __init__(self, bm25_retriever, dense_retriever, bm25_weight=0.5, dense_weight=0.5):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        
    def search(self, query, top_k=10):
        # Run BM25 and dense retrieval concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(
                self.bm25_retriever.search,
                query,
                top_k
            )
            dense_future = executor.submit(
                self.dense_retriever.search,
                query,
                top_k
            )
            bm25_results = bm25_future.result()
            dense_results = dense_future.result()

        scores = {}
        documents = {}

        # Normalize BM25 scores
        if bm25_results:
            max_bm25 = max(
                result["score"]
                for result in bm25_results
            )
        else:
            max_bm25 = 1.0

        # Normalize dense scores
        if dense_results:
            max_dense = max(
                result["score"]
                for result in dense_results
            )
        else:
            max_dense = 1.0

        # BM25 scores
        for result in bm25_results:
            entity_id = result["id"]
            normalized_score = (
                result["score"] / max_bm25
                if max_bm25 != 0
                else 0.0
            )
            scores[entity_id] = (
                self.bm25_weight * normalized_score
            )
            documents[entity_id] = result
        
        # Dense scores
        for result in dense_results:
            entity_id = result["id"]
            normalized_score = (
                result["score"] / max_dense
                if max_dense != 0
                else 0.0
            )
            scores[entity_id] = scores.get(entity_id, 0.0) + (
                self.dense_weight * normalized_score
            )
            documents[entity_id] = result

        # Rank by combined score
        ranked_entities = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        results = []
        for entity_id, hybrid_score in ranked_entities[:top_k]:
            result = documents[entity_id].copy()
            result["hybrid_score"] = hybrid_score
            results.append(result)
        return results
