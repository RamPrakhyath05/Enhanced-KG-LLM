import os
import pickle
import re
from typing import Dict, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


class PartitionRetriever:
    def __init__(
        self,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        model_name: str = "all-MiniLM-L6-v2",
        rrf_k: int = 60,
    ):
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.model_name = model_name
        self.rrf_k = rrf_k

        self.model = None
        self.centroids = None
        self.community_id_order = []
        self.bm25 = None
        self.community_entities = {}
        self.community_metadata = {}
        self.is_full_scale = False

        # Legacy / text-based hybrid retriever
        self.bm25_retriever = BM25Retriever()
        self.dense_retriever = DenseRetriever(model_name=model_name)
        self.hybrid_retriever = None

    def load_full_scale(
        self,
        indexes_dir: str = "indexes/partitions_full",
        data_dir: str = "data/partitions_full",
    ):
        """Loads precomputed full-scale centroids and BM25 index."""
        centroids_path = os.path.join(indexes_dir, "community_centroids.npy")
        bm25_path = os.path.join(indexes_dir, "community_bm25_index.pkl")
        id_order_path = os.path.join(indexes_dir, "community_id_order.pkl")
        comm_entities_path = os.path.join(data_dir, "community_entities.pkl")

        if not os.path.exists(centroids_path) or not os.path.exists(bm25_path):
            raise FileNotFoundError(f"Full scale partition index files not found in {indexes_dir}")

        self.centroids = np.load(centroids_path)

        with open(id_order_path, "rb") as f:
            self.community_id_order = pickle.load(f)

        with open(bm25_path, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]

        if os.path.exists(comm_entities_path):
            with open(comm_entities_path, "rb") as f:
                self.community_entities = pickle.load(f)

        self.model = SentenceTransformer(self.model_name)
        self.is_full_scale = True
        print(f"Loaded full-scale partition index ({len(self.community_id_order):,} communities, {self.centroids.shape[0]} centroids).")

    def build_from_communities(
        self,
        community_metadata: Dict[int, dict],
        save_dir: str = None,
    ):
        """Builds text-based hybrid partition index from community summaries."""
        self.community_metadata = community_metadata

        documents = []
        for comm_id, comm in community_metadata.items():
            doc = {
                "id": str(comm_id),
                "name": f"Community #{comm_id:04d}",
                "text": comm.get("summary_text") or f"Community #{comm_id} (Size: {comm.get('size', 0)})",
            }
            documents.append(doc)

        print(f"Building partition indexes over {len(documents):,} communities...")
        self.bm25_retriever.build(documents)
        self.dense_retriever.build(documents)

        self.hybrid_retriever = HybridRetriever(
            self.bm25_retriever,
            self.dense_retriever,
            bm25_weight=self.bm25_weight,
            dense_weight=self.dense_weight,
            fusion_method="rrf",
            rrf_k=self.rrf_k,
        )

        if save_dir:
            self.save(save_dir)

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.bm25_retriever.save(os.path.join(save_dir, "bm25.pkl"))
        self.dense_retriever.save(os.path.join(save_dir, "dense.pkl"))
        with open(os.path.join(save_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(self.community_metadata, f)
        print(f"Partition retriever saved to {save_dir}")

    def load(self, save_dir: str):
        """Loads index - auto-detects full-scale vs text-based."""
        if os.path.exists(os.path.join(save_dir, "community_centroids.npy")):
            self.load_full_scale(indexes_dir=save_dir)
            return

        bm25_path = os.path.join(save_dir, "bm25.pkl")
        dense_path = os.path.join(save_dir, "dense.pkl")
        meta_path = os.path.join(save_dir, "metadata.pkl")

        if not os.path.exists(bm25_path) or not os.path.exists(dense_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing partition index files in {save_dir}")

        self.bm25_retriever.load(bm25_path)
        self.dense_retriever.load(dense_path)
        with open(meta_path, "rb") as f:
            self.community_metadata = pickle.load(f)

        self.hybrid_retriever = HybridRetriever(
            self.bm25_retriever,
            self.dense_retriever,
            bm25_weight=self.bm25_weight,
            dense_weight=self.dense_weight,
            fusion_method="rrf",
            rrf_k=self.rrf_k,
        )
        self.is_full_scale = False

    def search(self, query: str, top_k: int = 5, client=None) -> List[dict]:
        """Executes hybrid partition search with RRF score fusion."""
        if self.is_full_scale:
            return self._search_full_scale(query, top_k=top_k, client=client)
        else:
            return self._search_text_based(query, top_k=top_k)

    def _search_full_scale(self, query: str, top_k: int = 5, client=None) -> List[dict]:
        query_emb = self.model.encode([query], normalize_embeddings=True)[0]
        sims = self.centroids @ query_emb
        vec_ranking = np.argsort(-sims)

        tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_ranking = np.argsort(-bm25_scores)

        rrf_scores = np.zeros(len(self.community_id_order))
        for rank, idx in enumerate(vec_ranking):
            rrf_scores[idx] += 1.0 / (self.rrf_k + rank + 1)
        for rank, idx in enumerate(bm25_ranking):
            rrf_scores[idx] += 1.0 / (self.rrf_k + rank + 1)

        fused_ranking = np.argsort(-rrf_scores)[:top_k]

        results = []
        for idx in fused_ranking:
            comm_id = self.community_id_order[idx]
            member_eids = self.community_entities.get(comm_id, [])

            # Resolve top entity names if client is available
            hub_names = []
            if client and member_eids:
                sample_eids = member_eids[:5]
                query_neo4j = """
                MATCH (e:Entity)
                WHERE e.Entity_ID IN $eids
                RETURN e.Entity_ID AS id, e.Entity_Name AS name
                """
                records = client.run_query(query_neo4j, {"eids": sample_eids})
                name_map = {r["id"]: r["name"] for r in records if r.get("name")}
                hub_names = [name_map.get(eid, eid) for eid in sample_eids]

            results.append({
                "community_id": int(comm_id),
                "score": float(rrf_scores[idx]),
                "rrf_score": float(rrf_scores[idx]),
                "vector_similarity": float(sims[idx]),
                "bm25_score": float(bm25_scores[idx]),
                "size": len(member_eids),
                "top_entities": hub_names if hub_names else member_eids[:5],
                "entity_ids": member_eids,
            })
        return results

    def _search_text_based(self, query: str, top_k: int = 5) -> List[dict]:
        if self.hybrid_retriever is None:
            raise RuntimeError("PartitionRetriever is not initialized. Call load() or build_from_communities().")

        raw_results = self.hybrid_retriever.search(query, top_k=top_k)
        results = []
        for res in raw_results:
            comm_id = int(res["id"])
            meta = self.community_metadata.get(comm_id, {})
            results.append({
                "community_id": comm_id,
                "score": res.get("hybrid_score", 0.0),
                "rrf_score": res.get("rrf_score", res.get("hybrid_score", 0.0)),
                "size": meta.get("size", 0),
                "top_entities": meta.get("top_entities", []),
                "summary_text": meta.get("summary_text", ""),
                "entity_ids": meta.get("entity_ids", []),
            })
        return results
