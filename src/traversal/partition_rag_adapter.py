from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .evidence_context import EvidenceBundle, EvidenceContextBuilder
from .path_scoring_traverser import PathScoringTraverser


@dataclass(frozen=True)
class PartitionTraversalResponse:
    """Final RAG retrieval payload with both prompt text and provenance."""

    bundle: EvidenceBundle
    selected_partition_ids: tuple[int, ...]


class PartitionRAGTraversalAdapter:
    """Consume existing ``PartitionRetriever.search`` results without mutation.

    Pass the top partition hits returned by the current retriever. Their
    ``entity_ids`` fields become the strictly bounded traversal scope.
    """

    def __init__(
        self,
        traverser: PathScoringTraverser,
        context_builder: EvidenceContextBuilder | None = None,
    ) -> None:
        self.traverser = traverser
        self.context_builder = context_builder or EvidenceContextBuilder()

    def retrieve(
        self,
        query: str,
        seed_entity_ids: Iterable[str],
        partition_hits: Sequence[dict],
    ) -> PartitionTraversalResponse:
        partition_ids = tuple(
            int(hit["community_id"])
            for hit in partition_hits
            if hit.get("community_id") is not None
        )
        allowed_entity_ids = {
            entity_id
            for hit in partition_hits
            for entity_id in hit.get("entity_ids", [])
            if entity_id
        }
        if not allowed_entity_ids and not partition_ids:
            raise ValueError("partition_hits must contain non-empty entity_ids or community_id")
        result = self.traverser.traverse(
            query=query,
            seed_entity_ids=seed_entity_ids,
            allowed_entity_ids=allowed_entity_ids,
            allowed_partition_ids=partition_ids,
        )
        return PartitionTraversalResponse(
            bundle=self.context_builder.build(result),
            selected_partition_ids=partition_ids,
        )
