from __future__ import annotations

import time
from src.generation.pipeline import EnhancedKGRAGPipeline


def test_pipeline_warmup():
    """Verify that EnhancedKGRAGPipeline.warmup initializes all components quickly."""
    pipeline = EnhancedKGRAGPipeline(allow_offline_fallback=True)
    t0 = time.perf_counter()
    pipeline.warmup(verbose=False)
    elapsed = time.perf_counter() - t0

    assert pipeline.retriever._is_loaded is True
    assert pipeline.retriever.model is not None
    assert elapsed < 5.0, f"Expected warmup < 5.0s, got {elapsed:.2f}s"
