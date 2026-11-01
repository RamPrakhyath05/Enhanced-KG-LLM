"""Enhanced Knowledge Graph RAG Pipeline with Adaptive Mechanisms.

This is the primary production pipeline for the Enhanced-KG-LLM system.
It integrates:
  1. Semantic-Structural Subgraph Retrieval (Hybrid Dense-BM25 over 3,625 partitions).
  2. Syntactic Interrogative & Prepositional Complement Extraction (resolves phrasing traps).
  3. Partition-Centric Co-Location Prior (+0.60 boost for nodes in retrieved subgraphs).
  4. Graph Centrality & Degree Prior (favors canonical entities over isolated stubs).
  5. Adaptive Query-Intent Traversal Gating (SINGULAR_FACT: 3 paths vs SET_AGGREGATION: 12 paths).
  6. Dynamic 3-Hop Traversal Depth (expands traversal when bridging entities appear).
  7. Fair-Share Multi-Seed Traversal Quota (guarantees balanced exploration across all seeds).
  8. Intent-Conditioned Grounded LLM Synthesis with Verified Citation Tracking.
"""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, List, Optional, Sequence, Set, Tuple

from src.db.neo4j_client import Neo4jClient
from src.generation.ollama_client import CITATION_REGEX, GroundedAnswer, OllamaClient
from src.query_subgraph.subgraph_retriever import SubgraphRetriever
from src.traversal.evidence_context import EvidenceBundle, EvidenceContextBuilder
from src.traversal.partition_rag_adapter import PartitionRAGTraversalAdapter
from src.traversal.path_scoring_traverser import (
    EvidencePath,
    KGEdge,
    PathScoringConfig,
    PathScoringTraverser,
    TraversalResult,
    relation_matches_query,
)
from src.traversal.seed_resolver import EntitySeedResolver, ResolvedSeed

# ============================================================================
# 1. Query Intent Gating Definitions
# ============================================================================

class QueryIntent(str, Enum):
    SINGULAR_FACT = "SINGULAR_FACT"
    SET_AGGREGATION = "SET_AGGREGATION"


AGGREGATION_PATTERNS = [
    r"\bhow many\b",
    r"\bnumber of\b",
    r"\bcount of\b",
    r"\btotal number\b",
    r"\bamount of\b",
    r"\blist all\b",
    r"\blist the\b",
    r"\bname all\b",
    r"\bname the\b",
    r"\bwho are all\b",
    r"\bwhich are all\b",
    r"\bwhat are all\b",
    r"\ball the\b",
    r"\bchildren of\b",
    r"\bchildren did\b",
    r"\bkids did\b",
    r"\bspouses of\b",
    r"\bspouses did\b",
    r"\bwives did\b",
    r"\bwives of\b",
    r"\bsons of\b",
    r"\bsons did\b",
    r"\bdaughters of\b",
    r"\bdaughters did\b",
    r"\bwho were the\b",
]

COMPILED_AGGREGATION_REGEX = re.compile("|".join(AGGREGATION_PATTERNS), re.IGNORECASE)


def detect_query_intent(query: str) -> tuple[QueryIntent, dict[str, Any]]:
    """Classify query intent to dynamically gate traversal cardinality."""
    match = COMPILED_AGGREGATION_REGEX.search(query.strip())
    if match:
        return QueryIntent.SET_AGGREGATION, {
            "matched_pattern": match.group(0),
            "max_paths": 12,
            "beam_width": 15,
            "mode_description": "Set / Aggregation Traversal (Expanded Breadth for Sibling Harvesting)",
        }
    return QueryIntent.SINGULAR_FACT, {
        "matched_pattern": None,
        "max_paths": 3,
        "beam_width": 5,
        "mode_description": "Focused Needle Traversal (Constrained Paths for Minimal Distractor Noise)",
    }


# ============================================================================
# 2. Linguistic Preposition & Entity Mention Definitions
# ============================================================================

ROBUST_QUERY_PREFIXES = (
    "how many of the ", "how many of ", "how many ",
    "how much of the ", "how much of ", "how much ",
    "who were the ", "who was the ", "who are the ", "who is the ",
    "what were the ", "what was the ", "what are the ", "what is the ",
    "what is a ", "what is an ", "what is ", "what are ",
    "when is the ", "when is ", "when was the ", "when was ",
    "who is ", "who was ", "tell me about the ", "tell me about ",
    "where is the ", "where is ", "list all of the ", "list all the ",
    "list all ", "list the ", "name all of the ", "name all the ", "name all ",
)

ROBUST_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from",
    "further", "had", "hadn't", "has", "hasn't", "have", "haven't",
    "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its",
    "itself", "me", "more", "most", "mustn't", "my", "myself", "no",
    "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "should", "shouldn't", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "were", "weren't", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "won't", "would",
    "wouldn't", "you", "your", "yours", "yourself", "yourselves",
    "tell", "show", "find", "give", "name", "list", "present",
    "many", "much", "total", "count", "number", "amount", "several",
}

ROBUST_QUERY_TERMS = {
    "direct", "directed", "director", "directing", "made", "make", "makes",
    "manufacturer", "manufactured", "found", "founder", "founded", "capital",
    "located", "location", "country", "countries", "film", "movie", "book", "company",
    "produced", "producer", "created", "creator", "wrote", "write", "written",
    "state", "states", "border", "borders", "bordering", "boundary", "boundaries",
    "share", "shares", "shared", "member", "members", "child", "children",
    "city", "headquarters", "headquarter", "building", "architectural", "style",
    "situated", "coordinates", "institution", "academic", "discoverer", "inventor",
    "graduate", "graduated", "river", "birthplace", "island", "continent", "official",
    "language", "filmed", "developer", "currency", "airline", "hub", "operating",
    "system", "systems", "actor", "played", "console", "designed", "president",
}

ROBUST_MENTION_STOP_WORDS = ROBUST_STOP_WORDS | ROBUST_QUERY_TERMS

DANGLING_TOKENS = {
    "of", "the", "in", "to", "for", "at", "from", "where", "which", "that",
    "with", "is", "was", "are", "were", "did", "does", "by", "a", "an", "as", "on"
}


# ============================================================================
# 3. Robust Entity Seed Resolver
# ============================================================================

class RobustEntitySeedResolver(EntitySeedResolver):
    """Enhanced seed resolver with syntactic interrogative cleaning and centrality priors."""

    def extract_mentions(self, query: str) -> list[str]:
        q = query.strip()
        q_lower = q.lower()
        for prefix in ROBUST_QUERY_PREFIXES:
            if q_lower.startswith(prefix):
                q = q[len(prefix):].strip()
                break

        q_clean = re.sub(r"[\?\.!]+$", "", q).strip()
        q_norm = q_clean.replace("’", "'")
        raw_tokens = [t for t in re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", q_norm) if t]
        if not raw_tokens:
            return []

        # 1. Syntactic Prepositional Complement Extraction:
        # In queries like "...headquarters of Nintendo" or "...devices manufactured by Sony",
        # the entity is the complement following "of" / "in" / "by"
        complement_phrases: list[str] = []
        for marker in (" of ", " in ", " by ", " at ", " for "):
            if marker in q_norm.lower():
                parts = q_norm.lower().split(marker)
                for part in parts[1:]:
                    part_clean = re.sub(r"^(the|a|an)\s+", "", part.strip())
                    part_tokens = [w for w in part_clean.split() if w not in DANGLING_TOKENS]
                    if part_tokens:
                        complement_phrases.append(" ".join(part_tokens))

        candidates: list[str] = list(complement_phrases)

        # 2. Significant content-word phrases
        significant = [w for w in raw_tokens if w.lower() not in ROBUST_MENTION_STOP_WORDS and len(w) >= 2]
        for gram_size in range(min(4, len(significant)), 0, -1):
            for i in range(len(significant) - gram_size + 1):
                phrase = " ".join(significant[i : i + gram_size]).strip()
                candidates.append(phrase)
                candidates.append(phrase.title())

        # 3. Sliding n-grams with boundary pruning
        for gram_size in range(min(5, len(raw_tokens)), 0, -1):
            for i in range(len(raw_tokens) - gram_size + 1):
                gram_tokens = raw_tokens[i : i + gram_size]
                while gram_tokens and gram_tokens[0].lower() in DANGLING_TOKENS:
                    gram_tokens = gram_tokens[1:]
                while gram_tokens and gram_tokens[-1].lower() in DANGLING_TOKENS:
                    gram_tokens = gram_tokens[:-1]
                if gram_tokens:
                    phrase = " ".join(gram_tokens).strip()
                    if not all(w.lower() in ROBUST_MENTION_STOP_WORDS for w in gram_tokens):
                        candidates.append(phrase)
                        candidates.append(phrase.title())

        seen: set[str] = set()
        unique_candidates: list[str] = []
        filter_words = ROBUST_MENTION_STOP_WORDS | DANGLING_TOKENS

        def candidate_salience(c: str) -> tuple[int, int, int]:
            c_words = c.lower().split()
            pure_content_count = sum(1 for w in c_words if w not in filter_words)
            return (pure_content_count > 0, pure_content_count, len(c))

        candidates.sort(key=candidate_salience, reverse=True)

        for candidate in candidates:
            c_key = candidate.casefold()
            if c_key not in seen and len(candidate) >= 2 and c_key not in filter_words:
                seen.add(c_key)
                unique_candidates.append(candidate)
        return unique_candidates

    def resolve(
        self,
        query: str,
        partition_hits: Sequence[dict] = (),
        max_seeds: int = 3,
        min_confidence: float = 0.35,
    ) -> list[ResolvedSeed]:
        candidates = self.extract_mentions(query)
        if not candidates:
            return []

        allowed_partition_ids: Set[int] = set()
        allowed_entity_ids: Set[str] = set()
        for hit in partition_hits:
            pid = hit.get("partition_id", hit.get("community_id"))
            if pid is not None:
                allowed_partition_ids.add(int(pid))
            for eid in hit.get("entity_ids", []):
                allowed_entity_ids.add(eid)

        cypher = """
        MATCH (e:Entity)
        WHERE e.Entity_Name IN $candidates
        RETURN e.Entity_ID AS id,
               e.Entity_Name AS name,
               e.Entity_Description AS description,
               e.partition_id AS partition_id,
               e.Entity_Name AS matched_term,
               COUNT { (e)-[]-() } AS degree
        LIMIT 60
        """
        try:
            records = self.db_client.run_query(cypher, {"candidates": candidates[:100]})
        except Exception:
            records = []

        alias_terms = list(dict.fromkeys(term.casefold() for term in candidates))
        alias_query = " OR ".join(
            '"' + term.replace("\\", "\\\\").replace('"', '\\"') + '"'
            for term in alias_terms[:80]
        )
        try:
            alias_records = self.db_client.run_query(
                "CALL db.index.fulltext.queryNodes('entity_alias_lookup', $query, {limit: 3000}) "
                "YIELD node, score "
                "RETURN node.Entity_ID AS id, node.Entity_Name AS name, "
                "node.Entity_Description AS description, node.partition_id AS partition_id, "
                "node.Entity_Aliases AS aliases, score, COUNT { (node)-[]-() } AS degree",
                {"query": alias_query},
            )
            records.extend(alias_records)
        except Exception:
            pass

        scored_seeds: dict[str, ResolvedSeed] = {}
        q_tokens = set(re.findall(r"[a-z0-9]+", query.lower())) - ROBUST_STOP_WORDS

        for rec in records:
            eid = str(rec.get("id", ""))
            name = str(rec.get("name") or eid)
            desc = str(rec.get("description") or "")
            pid = rec.get("partition_id")
            degree = int(rec.get("degree") or 0)
            aliases = {str(alias).casefold() for alias in (rec.get("aliases") or [])}
            term = str(rec.get("matched_term") or "")
            if term.lower() == name.lower():
                term = next(
                    (candidate for candidate in candidates if candidate.casefold() == name.casefold()),
                    name,
                )
            else:
                term = next(
                    (candidate for candidate in candidates if candidate.casefold() in aliases),
                    "",
                )
            if not term:
                continue
            if not eid:
                continue

            term_lower = term.lower()
            name_lower = name.lower()

            if term_lower == name_lower or term_lower in aliases:
                base_score = 1.0
            elif term_lower in name_lower:
                base_score = 0.5 + 0.4 * (len(term) / max(len(name), 1))
            else:
                base_score = 0.4

            length_bonus = min(0.3, len(term_lower.split()) * 0.1)

            # 1. Partition-centric relevance boost
            partition_boost = 0.0
            if pid is not None and int(pid) in allowed_partition_ids:
                partition_boost = 0.60
            elif eid in allowed_entity_ids:
                partition_boost = 0.60

            # 2. Graph degree / centrality prior
            degree_bonus = min(0.25, 0.04 * math.log(degree + 1)) if degree > 0 else 0.0

            # 3. Context clue description overlap bonus
            context_bonus = 0.0
            if desc and q_tokens:
                desc_tokens = set(re.findall(r"[a-z0-9]+", desc.lower()))
                overlap = len(q_tokens & desc_tokens)
                context_bonus = min(0.30, overlap * 0.08)

            # 4. Description substance bonus
            desc_bonus = min(0.15, len(desc.split()) / 500.0) if desc else 0.0

            # 5. Media penalty for definitional queries
            media_penalty = 0.0
            q_lower = query.lower()
            if any(q_lower.startswith(dp) for dp in ("what is a ", "what is an ", "what is ", "what are ")):
                desc_lower = desc.lower()
                media_markers = ("album", "studio album", "single by", "song by", "track by", "film", "movie")
                if any(marker in desc_lower for marker in media_markers):
                    media_penalty = 0.25

            confidence = base_score + length_bonus + partition_boost + degree_bonus + context_bonus + desc_bonus - media_penalty
            if confidence < min_confidence:
                continue

            current = scored_seeds.get(eid)
            if current is None or confidence > current.confidence:
                scored_seeds[eid] = ResolvedSeed(
                    entity_id=eid,
                    entity_name=name,
                    description=desc,
                    confidence=round(confidence, 4),
                    matched_term=term,
                    partition_id=int(pid) if pid is not None else None,
                )

        ranked = sorted(scored_seeds.values(), key=lambda s: s.confidence, reverse=True)
        return ranked[:max_seeds]


# ============================================================================
# 4. Fair-Share Path Scoring Traverser
# ============================================================================

class FairSharePathScoringTraverser(PathScoringTraverser):
    """Path traverser that enforces proportional multi-seed beam allocation."""

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
        beam: list[tuple[float, str, tuple[KGEdge, ...], frozenset[str], frozenset[str], str]] = [
            (0.0, seed, (), frozenset({seed}), frozenset(), seed) for seed in seeds
        ]
        completed: list[tuple[EvidencePath, str]] = []
        examined = 0

        effective_beam_width = max(self.config.beam_width, len(seeds) * 4)

        for _depth in range(self.config.max_hops):
            frontier = [node for _, node, _, _, _, _ in beam]
            neighbors = self._fetch_neighbors(frontier, allowed_entities, allowed_partitions)
            by_source: dict[str, list[KGEdge]] = {}
            for edge in neighbors:
                by_source.setdefault(edge.source_id, []).append(edge)

            proposals: list[tuple[float, str, tuple[KGEdge, ...], frozenset[str], frozenset[str], str]] = []
            for path_score, node, path, visited, relation_names, root_seed in beam:
                candidates = [edge for edge in by_source.get(node, []) if edge.target_id not in visited]
                examined += len(candidates)
                if not candidates:
                    if path:
                        completed.append((EvidencePath(path_score, path), root_seed))
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
                        root_seed,
                    ))
            if not proposals:
                break
            proposals.sort(key=lambda item: item[0], reverse=True)
            beam = proposals[:effective_beam_width]
            completed.extend((EvidencePath(score, path), root_seed) for score, _, path, _, _, root_seed in beam)

        # Fair-share quota path allocation across seeds
        completed.sort(key=lambda item: item[0].score, reverse=True)
        by_seed: dict[str, list[EvidencePath]] = {s: [] for s in seeds}
        for path_obj, root_seed in completed:
            if root_seed in by_seed:
                by_seed[root_seed].append(path_obj)

        per_seed_quota = max(1, self.config.max_paths // len(seeds))
        selected: list[EvidencePath] = []
        seen_edges: set[tuple] = set()

        # Phase 1: Guaranteed quota per seed
        for s in seeds:
            s_paths = self._distinct_paths(by_seed[s], per_seed_quota)
            for p in s_paths:
                sig = tuple((e.source_id, e.relation_name, e.target_id) for e in p.edges)
                if sig not in seen_edges:
                    seen_edges.add(sig)
                    selected.append(p)

        # Phase 2: Fill remainder up to max_paths
        if len(selected) < self.config.max_paths:
            all_remaining = self._distinct_paths([p for p, _ in completed], self.config.max_paths * 2)
            for p in all_remaining:
                if len(selected) >= self.config.max_paths:
                    break
                sig = tuple((e.source_id, e.relation_name, e.target_id) for e in p.edges)
                if sig not in seen_edges:
                    seen_edges.add(sig)
                    selected.append(p)

        answers = tuple(dict.fromkeys(edge.target_id for path in selected for edge in path.edges[-1:]))
        return TraversalResult(
            query=query,
            seed_entity_ids=seeds,
            scoped_entity_count=len(allowed_entities) if allowed_entities else len(allowed_partitions),
            paths=tuple(selected),
            edges_examined=examined,
            candidate_answer_ids=answers,
        )


# ============================================================================
# 5. Pipeline Results & Fallback Client
# ============================================================================

@dataclass
class PipelineResult:
    """Consolidated end-to-end execution outcome."""

    query: str
    seed_entity_ids: tuple[str, ...]
    selected_partition_ids: tuple[int, ...]
    partition_hits: list[dict]
    bundle: EvidenceBundle
    grounded_answer: GroundedAnswer
    timings_ms: dict[str, float]
    resolved_seeds: list[ResolvedSeed] = field(default_factory=list)
    query_intent: QueryIntent = QueryIntent.SINGULAR_FACT
    intent_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_latency_ms(self) -> float:
        return sum(self.timings_ms.values())

    def summary(self) -> str:
        partition_details = []
        for rank, p in enumerate(self.partition_hits, 1):
            pid = p.get("partition_id", p.get("community_id"))
            score = p.get("score", 0.0)
            seeds = p.get("seed_nodes", p.get("top_entities", []))[:5]
            partition_details.append(f"  • #{rank} Partition ID: {pid:<6} (Score: {score:.4f}) | Seed Nodes: {seeds}")

        seed_details = []
        if self.resolved_seeds:
            for s in self.resolved_seeds:
                seed_details.append(
                    f"  • {s.entity_name} ({s.entity_id}) | Confidence: {s.confidence:.2f} | Matched: '{s.matched_term}'"
                )
        else:
            seed_details.append(f"  • {list(self.seed_entity_ids)} (explicit / fallback)")

        lines = [
            "=" * 75,
            f"ENHANCED-KG-LLM PIPELINE EXECUTION SUMMARY",
            "=" * 75,
            f"Query: '{self.query}'",
            f"Intent: {self.query_intent.value} ({self.intent_metadata.get('mode_description', 'Gated')})",
            f"Top-{len(self.partition_hits)} Retrieved Partitions & Seed Nodes:",
            *partition_details,
            f"Resolved Traversal Seed Entities ({len(self.seed_entity_ids)}):",
            *seed_details,
            f"Candidate Answer IDs: {list(self.bundle.candidate_answer_ids)}",
            f"Evidence Paths Found: {len(self.bundle.paths)}",
            "-" * 75,
            f"LATENCY BREAKDOWN:",
            f"  • Subgraph Retrieval: {self.timings_ms.get('retrieval_ms', 0):.2f} ms",
            f"  • Retrieval Expansion: {self.timings_ms.get('retrieval_expansion_ms', 0):.2f} ms",
            f"  • Partition Traversal: {self.timings_ms.get('traversal_ms', 0):.2f} ms",
            f"  • LLM Grounding:      {self.timings_ms.get('llm_ms', 0):.2f} ms",
            f"  • Total Pipeline Time: {self.total_latency_ms:.2f} ms",
            "-" * 75,
            f"LLM GROUNDED ANSWER ({self.grounded_answer.model_name}):",
            self.grounded_answer.answer,
            "=" * 75,
        ]
        return "\n".join(lines)


# Backwards-compatibility alias
AdaptivePipelineResult = PipelineResult


class FallbackGraphClient:
    """Mock offline graph client providing multi-hop neighbors when Neo4j is offline."""

    def run_query(self, query: str, parameters: dict = None) -> list[dict]:
        params = parameters or {}
        if "toLower(e.Entity_Name)" in query or "candidates" in params or "name_fragment" in params:
            candidates = params.get("candidates") or [params.get("name_fragment") or params.get("term", "")]
            results = []
            for c in candidates:
                if not c:
                    continue
                cl = str(c).lower()
                if "australia" in cl or "commonwealth" in cl:
                    results.append({
                        "id": "Q408",
                        "name": "Commonwealth Australia",
                        "description": "Sovereign country comprising mainland Australian continent.",
                        "partition_id": 1187,
                    })
                elif "canberra" in cl:
                    results.append({
                        "id": "Q3114",
                        "name": "Canberra",
                        "description": "Federal capital city of Australia.",
                        "partition_id": 1187,
                    })
            return results

        if "MATCH (e:Entity)" in query and "RETURN e.Entity_ID" in query:
            nodes = params.get("nodes") or []
            edges = []
            for n in nodes:
                if n == "Q408":
                    edges.append({
                        "source": "Q408", "source_name": "Commonwealth Australia",
                        "rel": "capital", "rel_name": "capital",
                        "target": "Q3114", "target_name": "Canberra", "target_partition": 1187,
                    })
            return edges
        return []

    def close(self):
        pass


# ============================================================================
# 6. Production Main Pipeline: EnhancedKGRAGPipeline
# ============================================================================

class EnhancedKGRAGPipeline:
    """Production Knowledge Graph RAG Pipeline with Adaptive Mechanisms."""

    def __init__(
        self,
        subg_dir: str | None = None,
        neo4j_uri: str | None = None,
        neo4j_user: str | None = None,
        neo4j_password: str | None = None,
        ollama_model: str = "qwen2.5:7b",
        ollama_endpoint: str | None = None,
        traversal_config: PathScoringConfig | None = None,
        filter_mode: str = "auto",
        allow_offline_fallback: bool = False,
    ) -> None:
        self.allow_offline_fallback = allow_offline_fallback
        self.traversal_config = traversal_config or PathScoringConfig(
            max_hops=2,
            beam_width=5,
            max_paths=3,
            filter_mode=filter_mode,
        )
        self.retriever = SubgraphRetriever(subg_dir=subg_dir)
        self.ollama = OllamaClient(
            model_name=ollama_model,
            base_url=ollama_endpoint,
        )
        self.context_builder = EvidenceContextBuilder()

        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password
        self._db_client: Any = None
        self._seed_resolver: Optional[RobustEntitySeedResolver] = None
        self._is_warmed_up = False

    @property
    def db_client(self) -> Any:
        if self._db_client is None:
            try:
                self._db_client = Neo4jClient()
            except Exception as e:
                if self.allow_offline_fallback:
                    print(f"[Pipeline] Neo4j unavailable ({e}). Using FallbackGraphClient.")
                    self._db_client = FallbackGraphClient()
                else:
                    raise
        return self._db_client

    @property
    def seed_resolver(self) -> RobustEntitySeedResolver:
        if self._seed_resolver is None:
            self._seed_resolver = RobustEntitySeedResolver(db_client=self.db_client)
        return self._seed_resolver

    def warmup(self, verbose: bool = True) -> None:
        """Pre-loads vector models, BM25 indices, and verifies DB connectivity."""
        if self._is_warmed_up:
            return
        t0 = time.perf_counter()
        if verbose:
            print("[EnhancedKGRAGPipeline] Warming up components...")
        self.retriever.load()
        _ = self.db_client
        _ = self.seed_resolver
        self._is_warmed_up = True
        if verbose:
            print(f"[EnhancedKGRAGPipeline] Warmup completed in {time.perf_counter() - t0:.2f}s")

    def _generate_adaptive_grounded_answer(
        self,
        query: str,
        bundle: EvidenceBundle,
        intent: QueryIntent,
    ) -> GroundedAnswer:
        """Generate answer with intent-aware system instructions."""
        has_topic_entities = any(cit.get("label", "").startswith("S") for cit in bundle.citations)
        if not bundle.paths and not has_topic_entities:
            return GroundedAnswer(
                query=query,
                answer=(
                    "No supporting evidence was found in the retrieved partition scope. "
                    "Unable to ground an answer from the available knowledge graph evidence."
                ),
                citations_used=(),
                model_name=f"{self.ollama.model_name} (skipped: no evidence)",
                raw_context=bundle.context,
                was_skipped=True,
                skip_reason="no evidence",
            )

        if intent == QueryIntent.SET_AGGREGATION:
            system_instruction = (
                "You are a factual knowledge-graph reasoning assistant specializing in exact enumeration and counting. "
                "You must answer using ONLY the provided knowledge-graph evidence. "
                "Count and enumerate all distinct entities matching the requested relation present in the evidence. "
                "State the exact count of verified distinct entities found in the evidence (e.g. 'Based on the evidence, X has at least N distinct ...:'). "
                "For every entity mentioned, you MUST cite its corresponding citation tag (e.g. [P1.E1] or [S1]). "
                "Do not infer or hallucinate entities not shown in the evidence."
            )
        else:
            system_instruction = (
                "You are a factual knowledge-graph reasoning assistant. "
                "You must answer the user's question using ONLY the provided knowledge-graph evidence. "
                "Standard biographical parenthetical date ranges like (Birth Date – Death Date) denote a person's birth and death dates. "
                "For every factual assertion in your answer, you MUST cite the corresponding citation tag "
                "exactly as provided (e.g. [P1.E1] or [S1]). "
                "Answer concisely and put the citation immediately after each supported fact. "
                "If the evidence does not contain sufficient facts to answer, explicitly state that. "
                "Do not infer or hallucinate relationships not shown in the evidence."
            )

        user_prompt = f"{bundle.context}\n\nBased strictly on the evidence above, answer: {query}"

        t0 = time.perf_counter()
        if self.ollama.is_available():
            try:
                raw_response = self.ollama.generate(prompt=user_prompt, system=system_instruction)
                latency_ms = (time.perf_counter() - t0) * 1000
                citations = tuple(dict.fromkeys(CITATION_REGEX.findall(raw_response)))
                valid_citations = self.ollama._matching_evidence_citations(query, raw_response, bundle)
                if not citations:
                    citations = valid_citations
                    if citations:
                        raw_response = f"{raw_response.rstrip()} {' '.join(f'[{c}]' for c in citations)}"
                return GroundedAnswer(
                    query=query,
                    answer=raw_response,
                    citations_used=citations,
                    model_name=self.ollama.model_name,
                    is_mock=False,
                    latency_ms=latency_ms,
                    raw_context=bundle.context,
                )
            except Exception as e:
                if not self.allow_offline_fallback:
                    raise
                print(f"[OllamaClient] Generation error: {e}. Falling back to synthetic grounded answer.")

        return self.ollama.answer_with_evidence(
            query=query,
            bundle=bundle,
            mock_if_unavailable=self.allow_offline_fallback,
        )

    def run(
        self,
        query: str,
        seed_entity_ids: list[str] | None = None,
        top_k_partitions: int = 5,
        filter_mode: str | None = None,
        on_progress: Callable[[str, dict], None] | None = None,
    ) -> PipelineResult:
        """Executes full adaptive KG-RAG reasoning over the query."""
        def _notify(evt: str, payload: dict):
            if on_progress:
                on_progress(evt, payload)

        timings: dict[str, float] = {}

        # 0. Adaptive Query-Intent Classification
        intent, intent_meta = detect_query_intent(query)
        _notify("intent_classified", {"intent": intent.value, "metadata": intent_meta})

        # Check for multi-hop bridging indicators
        q_lower = query.lower()
        multihop_indicators = (
            "where the", "of the country", "of the founder", "birthplace of",
            "headquarters of", "developer of", "creator of", "who played",
            "coordinates of", "discoverer of", "inventor of", "graduate from",
            "situated at", "flows through"
        )
        is_multihop = any(ind in q_lower for ind in multihop_indicators)
        effective_max_hops = 3 if is_multihop else self.traversal_config.max_hops
        effective_beam_width = max(intent_meta["beam_width"], 12 if is_multihop else 5)

        adaptive_traversal_config = PathScoringConfig(
            max_hops=effective_max_hops,
            beam_width=effective_beam_width,
            max_paths=intent_meta["max_paths"],
            filter_mode=filter_mode or self.traversal_config.filter_mode,
        )

        # 1. Subgraph Retrieval (Hybrid Dense-BM25)
        _notify("retrieval_start", {"query": query, "top_k": top_k_partitions})
        t0 = time.perf_counter()
        hits = self.retriever.search(query, top_k=top_k_partitions)
        timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000
        _notify("retrieval_done", {"hits": hits, "elapsed_ms": timings["retrieval_ms"]})

        # 2. Extract and Resolve Seed Entities WITH partition_hits passed for community boosting
        _notify("seed_resolution_start", {"query": query})
        t0 = time.perf_counter()
        if seed_entity_ids is not None:
            resolved_seeds = []
        else:
            resolved_seeds = self.seed_resolver.resolve(query, partition_hits=hits)
            seed_entity_ids = [s.entity_id for s in resolved_seeds]
        timings["seed_resolution_ms"] = (time.perf_counter() - t0) * 1000
        _notify("seed_resolution_done", {"seeds": resolved_seeds, "elapsed_ms": timings["seed_resolution_ms"]})

        if not seed_entity_ids and hits:
            fallback_nodes = [
                eid
                for hit in hits[:2]
                for eid in hit.get("entity_ids", [])[:3]
                if eid
            ]
            seed_entity_ids = list(dict.fromkeys(fallback_nodes))

        # Seed partition mapping
        seed_partition_ids: set[int] = set()
        if seed_entity_ids and not self.allow_offline_fallback:
            try:
                seed_rows = self.db_client.run_query(
                    "MATCH (e:Entity) WHERE e.Entity_ID IN $seed_ids RETURN e.Entity_ID AS id, e.partition_id AS partition_id",
                    {"seed_ids": list(seed_entity_ids)},
                )
                seed_partition_ids = set(
                    int(row["partition_id"])
                    for row in seed_rows
                    if row.get("partition_id") is not None
                )
            except Exception:
                pass

        # Bridge expansion for cross-partition edges
        bridge_partition_ids: set[int] = set()
        if seed_entity_ids and not self.allow_offline_fallback:
            try:
                rel_rows = self.db_client.run_query(
                    "MATCH (source:Entity)-[rel:RELATED_TO]-(target:Entity) "
                    "WHERE source.Entity_ID IN $seed_ids "
                    "RETURN DISTINCT rel.Relation_Name AS rel_name, target.partition_id AS target_partition_id",
                    {"seed_ids": list(seed_entity_ids)},
                )
                for row in rel_rows:
                    rel_name = row.get("rel_name") or ""
                    tgt_pid = row.get("target_partition_id")
                    if tgt_pid is not None and relation_matches_query(rel_name, query):
                        bridge_partition_ids.add(int(tgt_pid))
            except Exception:
                pass

        all_scope_partition_ids = seed_partition_ids | bridge_partition_ids

        def include_seed_partitions(base_hits: list[dict]) -> list[dict]:
            traversal_hits = list(base_hits)
            included_partition_ids = {
                int(hit.get("partition_id", hit.get("community_id")))
                for hit in traversal_hits
                if hit.get("partition_id", hit.get("community_id")) is not None
            }
            for partition_id in sorted(all_scope_partition_ids - included_partition_ids):
                entity_ids = self.retriever.entity_map.get(partition_id, [])
                entity_id_set = set(entity_ids)
                seed_nodes = [eid for eid in seed_entity_ids if eid in entity_id_set][:5]
                traversal_hits.append({
                    "partition_id": partition_id,
                    "community_id": partition_id,
                    "score": 0.0,
                    "bm25_score": 0.0,
                    "vector_similarity": 0.0,
                    "size": len(entity_ids),
                    "seed_nodes": seed_nodes,
                    "top_entities": seed_nodes,
                    "entity_ids": entity_ids,
                })
            return traversal_hits

        traversal_hits = include_seed_partitions(hits)

        # 3. Partition-Bounded Multi-Seed Traversal with Fair-Share Quota
        _notify("traversal_start", {"seeds": seed_entity_ids, "partition_hits": traversal_hits})
        traverser = FairSharePathScoringTraverser(
            client=self.db_client,
            config=adaptive_traversal_config,
            model=self.retriever.model,
        )
        adapter = PartitionRAGTraversalAdapter(traverser, context_builder=self.context_builder)

        t0 = time.perf_counter()
        traversal_resp = adapter.retrieve(
            query=query,
            seed_entity_ids=seed_entity_ids,
            partition_hits=traversal_hits,
            seed_entities=resolved_seeds,
        )
        timings["traversal_ms"] = (time.perf_counter() - t0) * 1000

        if not traversal_resp.bundle.paths:
            expanded_top_k = min(max(top_k_partitions * 4, 20), 50)
            if expanded_top_k > len(hits):
                _notify("retrieval_expansion_start", {"top_k": expanded_top_k})
                t0 = time.perf_counter()
                expanded_hits = self.retriever.search(query, top_k=expanded_top_k)
                timings["retrieval_expansion_ms"] = (time.perf_counter() - t0) * 1000
                traversal_hits = include_seed_partitions(expanded_hits)
                t0 = time.perf_counter()
                traversal_resp = adapter.retrieve(
                    query=query,
                    seed_entity_ids=seed_entity_ids,
                    partition_hits=traversal_hits,
                    seed_entities=resolved_seeds,
                )
                timings["traversal_ms"] += (time.perf_counter() - t0) * 1000

        _notify("traversal_done", {
            "response": traversal_resp,
            "elapsed_ms": timings["traversal_ms"],
        })

        # 4. Context Assembly
        bundle = traversal_resp.bundle
        _notify("context_done", {"bundle": bundle})

        # 5. Grounded LLM Generation with Intent-Conditioned Guidance
        _notify("llm_start", {"bundle": bundle, "model": self.ollama.model_name})
        t0 = time.perf_counter()

        grounded_answer = self._generate_adaptive_grounded_answer(query, bundle, intent)
        timings["llm_ms"] = (time.perf_counter() - t0) * 1000
        _notify("llm_done", {
            "grounded_answer": grounded_answer,
            "elapsed_ms": timings["llm_ms"],
        })

        return PipelineResult(
            query=query,
            seed_entity_ids=tuple(seed_entity_ids),
            selected_partition_ids=traversal_resp.selected_partition_ids,
            partition_hits=hits,
            bundle=bundle,
            grounded_answer=grounded_answer,
            timings_ms=timings,
            resolved_seeds=resolved_seeds,
            query_intent=intent,
            intent_metadata=intent_meta,
        )

    def close(self) -> None:
        if self._db_client is not None and hasattr(self._db_client, "close"):
            self._db_client.close()


# Backwards-compatibility aliases
AdaptiveMechanismsKGRAGPipeline = EnhancedKGRAGPipeline
AdaptiveGatedKGRAGPipeline = EnhancedKGRAGPipeline
