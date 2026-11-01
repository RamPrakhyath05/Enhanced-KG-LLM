"""Generation and End-to-End Pipeline module."""

from .ollama_client import GroundedAnswer, OllamaClient
from .pipeline import (
    AdaptiveGatedKGRAGPipeline,
    AdaptiveMechanismsKGRAGPipeline,
    AdaptivePipelineResult,
    EnhancedKGRAGPipeline,
    PipelineResult,
    QueryIntent,
    detect_query_intent,
)

__all__ = [
    "GroundedAnswer",
    "OllamaClient",
    "EnhancedKGRAGPipeline",
    "AdaptiveMechanismsKGRAGPipeline",
    "AdaptiveGatedKGRAGPipeline",
    "PipelineResult",
    "AdaptivePipelineResult",
    "QueryIntent",
    "detect_query_intent",
]
