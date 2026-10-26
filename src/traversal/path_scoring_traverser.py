from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {"a", "an", "and", "for", "from", "is", "of", "the", "to", "what", "which", "who"}
STRUCTURAL_TERMS = {"administrative", "described", "given", "instance", "source", "subclass", "taxon", "time", "timezone"}
RELATION_INTENTS = {
    "author": {"author", "authored", "creator", "wrote", "writes", "written", "writer"},
    "place of birth": {"birthplace", "born", "birth"},
    "date of birth": {"birthdate", "birthday", "born", "birth"},
    "capital": {"capital", "city"},
    "located in": {"located", "location", "where"},
}


def relation_intent_name(relation: str) -> str | None:
    """Return the normalized intent family for a KG relation, when known."""
    relation_terms = set(TOKEN_RE.findall(relation.lower()))
    for intent in RELATION_INTENTS:
        if relation_terms & set(TOKEN_RE.findall(intent)):
            return intent
    return None


def relation_matches_query(relation: str, query: str) -> bool:
    """Match common question wording to a KG relation intent."""
    relation_terms = set(TOKEN_RE.findall(relation.lower()))
    query_terms = set(TOKEN_RE.findall(query.lower())) - STOPWORDS
    if relation_terms & query_terms:
        return True
    intent = relation_intent_name(relation)
    return bool(intent and query_terms & RELATION_INTENTS[intent])


class QueryClient(Protocol):
    def run_query(self, query: str, parameters: dict | None = None) -> list[dict]: ...


class Encoder(Protocol):
    def encode(self, sentences: Sequence[str], normalize_embeddings: bool = True) -> Any: ...


@dataclass(frozen=True)
class KGEdge:
    source_id: str
    source_name: str
    relation_id: str
    relation_name: str
    target_id: str
    target_name: str
    target_description: str = ""

    @property
    def citation_id(self) -> str:
        return f"{self.source_id}|{self.relation_id}|{self.target_id}"

    @property
    def text(self) -> str:
        return " ".join(part for part in (self.source_name, self.relation_name, self.target_name, self.target_description) if part)


@dataclass(frozen=True)
class EvidencePath:
    score: float
    edges: tuple[KGEdge, ...]

    @property
    def answer_entity_ids(self) -> tuple[str, ...]:
        return (self.edges[-1].target_id,) if self.edges else ()


@dataclass(frozen=True)
class TraversalResult:
    query: str
    seed_entity_ids: tuple[str, ...]
    scoped_entity_count: int
    paths: tuple[EvidencePath, ...]
    edges_examined: int
    candidate_answer_ids: tuple[str, ...]


@dataclass(frozen=True)
class PathScoringConfig:
    """Defaults selected from the T10 experiment, adapted to partition scope."""

    max_hops: int = 3
    beam_width: int = 5
    max_paths: int = 3
    max_neighbors_per_node: int = 100
    dense_weight: float = 0.35
    lexical_weight: float = 0.35
    relation_weight: float = 0.20
    relation_novelty_weight: float = 0.10
    hop_penalty: float = 0.05
    filter_mode: str = "auto"  # "auto", "partition_id", or "entity_id"

    def __post_init__(self) -> None:
        if self.max_hops < 1 or self.beam_width < 1 or self.max_paths < 1 or self.max_neighbors_per_node < 1:
            raise ValueError("Traversal bounds must be positive.")
        if min(self.dense_weight, self.lexical_weight, self.relation_weight, self.relation_novelty_weight, self.hop_penalty) < 0:
            raise ValueError("Traversal weights must be non-negative.")
        if self.filter_mode not in ("auto", "partition_id", "entity_id"):
            raise ValueError(f"Invalid filter_mode '{self.filter_mode}'. Must be 'auto', 'partition_id', or 'entity_id'.")


class PathScoringTraverser:
    """Retrieve complete, query-relevant paths inside supplied partition IDs.

    ``allowed_entity_ids`` should be the de-duplicated union of the entity IDs
    from the existing top-k partition retrieval result. The traverser does not
    select partitions and cannot cross the supplied partition scope.
    """

    def __init__(
        self,
        client: QueryClient,
        config: PathScoringConfig | None = None,
        model: Encoder | None = None,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.client = client
        self.config = config or PathScoringConfig()
        self._model = model
        self.model_name = model_name

    @property
    def model(self) -> Encoder:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def traverse(
        self,
        query: str,
        seed_entity_ids: Iterable[str],
        allowed_entity_ids: Iterable[str] | None = None,
        allowed_partition_ids: Iterable[int] | None = None,
    ) -> TraversalResult:
        seeds = tuple(dict.fromkeys(entity_id for entity_id in seed_entity_ids if entity_id))
        allowed_entities = frozenset(entity_id for entity_id in (allowed_entity_ids or []) if entity_id)
        allowed_partitions = tuple(int(pid) for pid in (allowed_partition_ids or []) if pid is not None)
        if not query.strip():
            raise ValueError("query must not be empty")
        if not seeds:
            raise ValueError("At least one seed entity ID is required")
        if not allowed_entities and not allowed_partitions:
            raise ValueError("allowed_entity_ids or allowed_partition_ids must be provided")

        query_embedding = self._encode([query])[0]
        beam: list[tuple[float, str, tuple[KGEdge, ...], frozenset[str], frozenset[str]]] = [
            (0.0, seed, (), frozenset({seed}), frozenset()) for seed in seeds
        ]
        completed: list[EvidencePath] = []
        examined = 0

        # Adjust effective beam width when multiple seeds are present so each seed gets explored
        effective_beam_width = max(self.config.beam_width, len(seeds) * 3)

        for _depth in range(self.config.max_hops):
            frontier = [node for _, node, _, _, _ in beam]
            neighbors = self._fetch_neighbors(frontier, allowed_entities, allowed_partitions)
            by_source: dict[str, list[KGEdge]] = {}
            for edge in neighbors:
                by_source.setdefault(edge.source_id, []).append(edge)

            proposals: list[tuple[float, str, tuple[KGEdge, ...], frozenset[str], frozenset[str]]] = []
            for path_score, node, path, visited, relation_names in beam:
                candidates = [edge for edge in by_source.get(node, []) if edge.target_id not in visited]
                examined += len(candidates)
                if not candidates:
                    if path:
                        completed.append(EvidencePath(path_score, path))
                    continue
                for edge, edge_score in self._score_candidates(query, query_embedding, candidates):
                    novelty = self.config.relation_novelty_weight if edge.relation_name not in relation_names else 0.0
                    total_score = path_score + edge_score + novelty - self.config.hop_penalty
                    proposals.append((
                        total_score,
                        edge.target_id,
                        path + (edge,),
                        visited | {edge.target_id},
                        relation_names | {edge.relation_name},
                    ))
            if not proposals:
                break
            proposals.sort(key=lambda item: item[0], reverse=True)
            beam = proposals[: effective_beam_width]
            completed.extend(EvidencePath(score, path) for score, _, path, _, _ in beam)

        completed.sort(key=lambda item: item.score, reverse=True)
        selected = self._distinct_paths(completed, self.config.max_paths)
        answers = tuple(dict.fromkeys(edge.target_id for path in selected for edge in path.edges[-1:]))
        return TraversalResult(
            query=query,
            seed_entity_ids=seeds,
            scoped_entity_count=len(allowed_entities) if allowed_entities else len(allowed_partitions),
            paths=tuple(selected),
            edges_examined=examined,
            candidate_answer_ids=answers,
        )

    def _fetch_neighbors(
        self,
        source_ids: Sequence[str],
        allowed: frozenset[str],
        allowed_partitions: Sequence[int] = (),
    ) -> list[KGEdge]:
        if not source_ids:
            return []

        # Determine whether to filter by partition_id property or fallback to Entity_ID
        if self.config.filter_mode == "partition_id":
            use_partition_id = bool(allowed_partitions)
        elif self.config.filter_mode == "entity_id":
            use_partition_id = False
        else:  # "auto"
            use_partition_id = bool(allowed_partitions)

        if use_partition_id:
            where_clause = "WHERE target.partition_id IN $allowed_partition_ids"
            params = {
                "source_ids": list(dict.fromkeys(source_ids)),
                "allowed_partition_ids": list(allowed_partitions),
                "per_source_limit": self.config.max_neighbors_per_node,
            }
        else:
            where_clause = "WHERE target.Entity_ID IN $allowed_entity_ids"
            params = {
                "source_ids": list(dict.fromkeys(source_ids)),
                "allowed_entity_ids": list(allowed),
                "per_source_limit": self.config.max_neighbors_per_node,
            }

        cypher = f"""
        UNWIND $source_ids AS sid
        CALL {{
            WITH sid
            MATCH (source:Entity {{Entity_ID: sid}})-[rel:RELATED_TO]->(target:Entity)
            {where_clause}
            RETURN source.Entity_ID AS source_id,
                   source.Entity_Name AS source_name,
                   rel.Relation_ID AS relation_id,
                   rel.Relation_Name AS relation_name,
                   target.Entity_ID AS target_id,
                   target.Entity_Name AS target_name,
                   target.Entity_Description AS target_description
            ORDER BY target.Entity_ID, rel.Relation_ID
            LIMIT $per_source_limit
        }}
        RETURN source_id, source_name, relation_id, relation_name,
               target_id, target_name, target_description
        """
        records = self.client.run_query(cypher, params)
        return [
            KGEdge(
                source_id=str(row["source_id"]),
                source_name=str(row.get("source_name") or row["source_id"]),
                relation_id=str(row.get("relation_id") or ""),
                relation_name=str(row.get("relation_name") or "related to"),
                target_id=str(row["target_id"]),
                target_name=str(row.get("target_name") or row["target_id"]),
                target_description=str(row.get("target_description") or ""),
            )
            for row in records
        ]

    def _score_candidates(self, query: str, query_embedding: np.ndarray, candidates: Sequence[KGEdge]) -> list[tuple[KGEdge, float]]:
        dense = self._encode([edge.text for edge in candidates]) @ query_embedding
        lexical = self._local_bm25(query, [edge.text for edge in candidates])
        relation = np.array([self._relation_relevance(edge.relation_name, query) for edge in candidates], dtype=float)
        scores = (
            self.config.dense_weight * self._min_max(dense)
            + self.config.lexical_weight * self._min_max(lexical)
            + self.config.relation_weight * relation
        )
        return list(zip(candidates, scores.tolist()))

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        encoded = self.model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(encoded, dtype=float)

    @staticmethod
    def _min_max(values: np.ndarray) -> np.ndarray:
        lower, upper = float(values.min()), float(values.max())
        return np.zeros_like(values, dtype=float) if upper == lower else (values - lower) / (upper - lower)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]

    def _local_bm25(self, query: str, documents: Sequence[str]) -> np.ndarray:
        """BM25 over the current bounded candidate set, avoiding a global edge index."""
        tokenized = [self._tokens(document) for document in documents]
        document_count = len(tokenized)
        average_length = sum(len(document) for document in tokenized) / document_count if document_count else 1.0
        frequencies = Counter(token for document in tokenized for token in set(document))
        query_terms = set(self._tokens(query))
        k1, b = 1.5, 0.75
        scores = []
        for document in tokenized:
            counts = Counter(document)
            score = 0.0
            for term in query_terms:
                frequency = counts[term]
                if not frequency:
                    continue
                idf = math.log(1 + (document_count - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
                denominator = frequency + k1 * (1 - b + b * len(document) / average_length)
                score += idf * frequency * (k1 + 1) / denominator
            scores.append(score)
        return np.asarray(scores, dtype=float)

    def _relation_relevance(self, relation: str, query: str) -> float:
        relation_terms = set(self._tokens(relation))
        query_terms = set(self._tokens(query))
        if relation_matches_query(relation, query):
            return 1.0
        return 0.0 if relation_terms & STRUCTURAL_TERMS else 0.25

    @staticmethod
    def _distinct_paths(paths: Sequence[EvidencePath], maximum: int) -> list[EvidencePath]:
        selected, seen = [], set()
        for path in paths:
            identity = tuple(edge.citation_id for edge in path.edges)
            if identity and identity not in seen:
                selected.append(path)
                seen.add(identity)
            if len(selected) == maximum:
                break
        return selected
