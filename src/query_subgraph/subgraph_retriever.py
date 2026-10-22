from __future__ import annotations

import os
import pickle
import re
import time
from typing import Any, List, Optional, Sequence
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _resolve_file(subg_dir: str, candidates: list[str]) -> str:
    """Find the first matching file among candidate filenames in subg_dir."""
    for name in candidates:
        p = os.path.join(subg_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of {candidates} found in '{subg_dir}'")


def validate_partition_artifacts(subg_dir: str) -> dict[str, Any]:
    """Validate that a partition directory contains all necessary files and matching dimensions.

    Supports both current naming ('centroids.npy', 'bm25_docs.pkl') and legacy ('subgraph_*').
    """
    files = {
        "embeddings": _resolve_file(subg_dir, ["centroids.npy", "subgraph_embeddings.npy"]),
        "bm25_docs": _resolve_file(subg_dir, ["bm25_docs.pkl", "subgraph_bm25_docs.pkl"]),
        "partition_ids": _resolve_file(subg_dir, ["partition_ids.pkl", "subgraph_partition_ids.pkl"]),
        "entity_map": _resolve_file(subg_dir, ["entity_map.pkl", "subgraph_entity_map.pkl"]),
    }

    embeddings = np.load(files["embeddings"], mmap_mode="r")
    with open(files["partition_ids"], "rb") as f:
        pids = pickle.load(f)

    if len(embeddings) != len(pids):
        raise ValueError(f"Mismatch: embeddings count ({len(embeddings)}) != partition IDs count ({len(pids)})")

    return {
        "subg_dir": subg_dir,
        "partition_count": len(pids),
        "embedding_shape": embeddings.shape,
        "files_verified": list(files.keys()),
        "resolved_paths": files,
    }


class SubgraphRetriever:
    """Retriever for the subgraphs scheme created from Leiden graph clustering.
    
    Loosely coupled: pass any subg_dir or set the SUBG_DIR environment variable
    to instantly swap in new partition schemes without touching code.
    """

    def __init__(
        self,
        subg_dir: Optional[str] = None,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ) -> None:
        self.subg_dir = subg_dir or os.getenv("SUBG_DIR", "data/subg")
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.embeddings: Optional[np.ndarray] = None
        self.bm25_docs: Optional[list] = None
        self.partition_ids: Optional[list] = None
        self.entity_map: Optional[dict] = None
        self.bm25: Optional[BM25Okapi] = None
        self.model: Optional[SentenceTransformer] = None
        self._is_loaded = False

    def load(self) -> None:
        """Loads precomputed subgraph artifacts and initializes BM25 & SentenceTransformer."""
        if self._is_loaded:
            return

        t0 = time.perf_counter()
        meta = validate_partition_artifacts(self.subg_dir)
        paths = meta["resolved_paths"]

        self.embeddings = np.load(paths["embeddings"])
        with open(paths["bm25_docs"], "rb") as f:
            self.bm25_docs = pickle.load(f)
        with open(paths["partition_ids"], "rb") as f:
            self.partition_ids = pickle.load(f)
        with open(paths["entity_map"], "rb") as f:
            self.entity_map = pickle.load(f)

        # Support pre-tokenized lists of tokens or raw document strings
        if self.bm25_docs and isinstance(self.bm25_docs[0], list):
            tokenized_docs = self.bm25_docs
        else:
            tokenized_docs = [tokenize(doc) for doc in self.bm25_docs]

        self.bm25 = BM25Okapi(tokenized_docs)

        self.model = SentenceTransformer(self.model_name, device=self.device)
        self._is_loaded = True
        elapsed = time.perf_counter() - t0
        print(f"[SubgraphRetriever] Loaded {len(self.partition_ids)} subgraphs on {self.device} in {elapsed:.2f}s")

    def search(self, query: str, top_k: int = 5, client: Any = None) -> List[dict]:
        """Runs dense embedding + BM25 scoring over subgraphs.
        
        Returns hit dicts adhering to the contract expected by PartitionRAGTraversalAdapter:
        {
            'partition_id': int(pid),
            'community_id': int(pid),
            'score': float(combined_score),
            'bm25_score': float(bm25_norm),
            'vector_similarity': float(emb_norm),
            'size': int(num_entities),
            'seed_nodes': list[str],
            'top_entities': list[str],
            'entity_ids': list[str]
        }
        """
        if not self._is_loaded:
            self.load()

        assert self.model is not None
        assert self.bm25 is not None
        assert self.embeddings is not None
        assert self.partition_ids is not None
        assert self.entity_map is not None

        # 1. Dense query embedding
        query_emb = self.model.encode([query], normalize_embeddings=True)[0]
        emb_scores = self.embeddings @ query_emb

        # 2. Lexical BM25 score
        query_tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)

        # 3. Normalize both to [0, 1]
        bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
        emb_min, emb_max = emb_scores.min(), emb_scores.max()
        bm25_norm = (bm25_scores - bm25_min) / (bm25_max - bm25_min + 1e-10)
        emb_norm = (emb_scores - emb_min) / (emb_max - emb_min + 1e-10)

        # 4. Equal 50/50 weighting
        combined = 0.5 * bm25_norm + 0.5 * emb_norm

        top_indices = np.argsort(combined)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            pid = int(self.partition_ids[idx])
            entities = self.entity_map.get(pid, [])
            seed_nodes = entities[:5]
            results.append({
                "partition_id": pid,
                "community_id": pid,
                "score": float(combined[idx]),
                "bm25_score": float(bm25_norm[idx]),
                "vector_similarity": float(emb_norm[idx]),
                "size": len(entities),
                "seed_nodes": seed_nodes,
                "top_entities": seed_nodes,
                "entity_ids": entities,
            })

        return results
