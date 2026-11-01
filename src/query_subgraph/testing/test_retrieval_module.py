from __future__ import annotations

import os
import time
from src.query_subgraph.subgraph_retriever import SubgraphRetriever


def test_subgraph_retriever_fast_load():
    """Verify that SubgraphRetriever loads pre-serialized BM25 index in < 3.0 seconds."""
    retriever = SubgraphRetriever()
    t0 = time.perf_counter()
    retriever.load()
    elapsed = time.perf_counter() - t0

    assert retriever._is_loaded is True
    assert retriever.bm25 is not None
    assert retriever.embeddings is not None
    assert len(retriever.partition_ids) == 3625
    assert elapsed < 3.0, f"Expected load time < 3.0s, got {elapsed:.2f}s"
