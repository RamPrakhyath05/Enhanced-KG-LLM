from concurrent.futures import ThreadPoolExecutor
import numpy as np


class HybridRetriever:
    def __init__(
        self,
        bm25_retriever,
        dense_retriever,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
    ):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 10, fusion_method: str = None) -> list:
        method = fusion_method or self.fusion_method

        with ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(
                self.bm25_retriever.search,
                query,
                top_k * 3,
            )
            dense_future = executor.submit(
                self.dense_retriever.search,
                query,
                top_k * 3,
            )
            bm25_results = bm25_future.result()
            dense_results = dense_future.result()

        if method.lower() == "rrf":
            return self._fuse_rrf(dense_results, bm25_results, top_k=top_k)
        else:
            return self._fuse_weighted(dense_results, bm25_results, top_k=top_k)

    def _fuse_rrf(self, dense_results: list, bm25_results: list, top_k: int = 10) -> list:
        scores = {}
        documents = {}

        for rank, res in enumerate(dense_results):
            doc_id = res["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank + 1))
            documents[doc_id] = res

        for rank, res in enumerate(bm25_results):
            doc_id = res["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank + 1))
            if doc_id not in documents:
                documents[doc_id] = res

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        results = []
        for doc_id, rrf_score in ranked[:top_k]:
            res = documents[doc_id].copy()
            res["hybrid_score"] = rrf_score
            res["rrf_score"] = rrf_score
            results.append(res)
        return results

    def _fuse_weighted(self, dense_results: list, bm25_results: list, top_k: int = 10) -> list:
        scores = {}
        documents = {}

        max_bm25 = max([r["score"] for r in bm25_results], default=1.0) or 1.0
        max_dense = max([r["score"] for r in dense_results], default=1.0) or 1.0

        for res in bm25_results:
            doc_id = res["id"]
            norm_score = res["score"] / max_bm25 if max_bm25 != 0 else 0.0
            scores[doc_id] = self.bm25_weight * norm_score
            documents[doc_id] = res

        for res in dense_results:
            doc_id = res["id"]
            norm_score = res["score"] / max_dense if max_dense != 0 else 0.0
            scores[doc_id] = scores.get(doc_id, 0.0) + (self.dense_weight * norm_score)
            documents[doc_id] = res

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        results = []
        for doc_id, score in ranked[:top_k]:
            res = documents[doc_id].copy()
            res["hybrid_score"] = score
            results.append(res)
        return results
