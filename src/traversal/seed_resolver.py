"""Automated Seed Entity Resolver for Natural Language Queries.

Identifies topic entities and seed nodes directly from query text:
1. Extracts candidate mention phrases (n-grams, capitalized named entities).
2. Searches Neo4j (or local graph client) for matching entity names and aliases.
3. Cross-references candidates with retrieved top partition IDs to boost confidence.
4. Returns all relevant seed entities for multi-seed graph traversal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Set

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its",
    "itself", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not",
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shan't", "she", "should", "shouldn't",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "were",
    "weren't", "what", "when", "where", "which", "while", "who", "whom", "why",
    "with", "won't", "would", "wouldn't", "you", "your", "yours", "yourself",
    "yourselves", "tell", "show", "find", "give", "name", "list", "present",
    "according", "work", "works"
}


@dataclass(frozen=True)
class ResolvedSeed:
    """A candidate seed entity identified from the query."""

    entity_id: str
    entity_name: str
    description: str
    confidence: float
    matched_term: str
    partition_id: Optional[int] = None


class EntitySeedResolver:
    """Resolves one or more seed entities from query text using Graph search and Partition context."""

    def __init__(self, db_client: Any) -> None:
        self.db_client = db_client

    def extract_mentions(self, query: str) -> list[str]:
        """Extract candidate entity mention phrases from the query string."""
        cleaned = re.sub(r"[^\w\s]", " ", query).strip()
        words = cleaned.split()
        if not words:
            return []

        candidates: list[str] = []

        # 1. Title-cased / Proper noun sequences (e.g. "King Yima", "Persian Literature", "Commonwealth Australia")
        capitalized_chunks: list[str] = []
        current_chunk: list[str] = []
        for w in words:
            if w[0].isupper() and w.lower() not in STOP_WORDS:
                current_chunk.append(w)
            else:
                if current_chunk:
                    capitalized_chunks.append(" ".join(current_chunk))
                    current_chunk = []
        if current_chunk:
            capitalized_chunks.append(" ".join(current_chunk))

        candidates.extend(capitalized_chunks)

        # 2. Multi-word n-grams (2-grams, 3-grams, 4-grams)
        n = len(words)
        for gram_size in (4, 3, 2):
            for i in range(n - gram_size + 1):
                gram_words = words[i : i + gram_size]
                # Filter out n-grams that are purely stopwords
                non_stops = [w for w in gram_words if w.lower() not in STOP_WORDS]
                if non_stops:
                    phrase = " ".join(gram_words)
                    candidates.append(phrase)

        # 3. Individual significant words (length >= 3, not in stopwords)
        for w in words:
            if w.lower() not in STOP_WORDS and len(w) >= 3:
                candidates.append(w)

        # De-duplicate preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            c_clean = c.strip()
            c_lower = c_clean.lower()
            if c_lower not in seen and len(c_clean) >= 2:
                seen.add(c_lower)
                unique_candidates.append(c_clean)

        return unique_candidates

    def resolve(
        self,
        query: str,
        partition_hits: Sequence[dict] = (),
        max_seeds: int = 3,
        min_confidence: float = 0.35,
    ) -> list[ResolvedSeed]:
        """Resolve relevant seed entities from the query, prioritizing matching partition members.
        
        Args:
            query: The user query string.
            partition_hits: Hits returned from SubgraphRetriever.
            max_seeds: Maximum number of distinct seed entities to return.
            min_confidence: Threshold confidence score for accepting a seed.
        """
        candidates = self.extract_mentions(query)
        if not candidates:
            return []

        # Gather partition scope for boosting
        allowed_partition_ids: Set[int] = set()
        allowed_entity_ids: Set[str] = set()
        for hit in partition_hits:
            pid = hit.get("partition_id", hit.get("community_id"))
            if pid is not None:
                allowed_partition_ids.add(int(pid))
            for eid in hit.get("entity_ids", []):
                allowed_entity_ids.add(eid)

        # Query Neo4j / DB Client for candidate entities
        cypher = """
        UNWIND $candidates AS term
        MATCH (e:Entity)
        WHERE toLower(e.Entity_Name) = toLower(term)
           OR toLower(term) IN [a IN coalesce(e.Entity_Aliases, []) | toLower(a)]
           OR (size(term) >= 4 AND toLower(e.Entity_Name) CONTAINS toLower(term))
        RETURN DISTINCT e.Entity_ID AS id,
                        e.Entity_Name AS name,
                        e.Entity_Description AS description,
                        e.partition_id AS partition_id,
                        term AS matched_term
        LIMIT 40
        """
        try:
            records = self.db_client.run_query(cypher, {"candidates": candidates[:15]})
        except Exception:
            records = []

        if not records and hasattr(self.db_client, "run_query"):
            # Try single-term search fallback if bulk UNWIND unsupported
            for term in candidates[:5]:
                try:
                    fallback_records = self.db_client.run_query(
                        "MATCH (e:Entity) WHERE toLower(e.Entity_Name) CONTAINS toLower($term) "
                        "RETURN e.Entity_ID AS id, e.Entity_Name AS name, e.Entity_Description AS description, "
                        "e.partition_id AS partition_id, $term AS matched_term LIMIT 5",
                        {"term": term, "candidates": [term]},
                    )
                    records.extend(fallback_records)
                except Exception:
                    pass

        # Score and rank candidates
        scored_seeds: dict[str, ResolvedSeed] = {}
        for rec in records:
            eid = str(rec.get("id", ""))
            name = str(rec.get("name") or eid)
            desc = str(rec.get("description") or "")
            pid = rec.get("partition_id")
            term = str(rec.get("matched_term", ""))

            if not eid:
                continue

            term_lower = term.lower()
            name_lower = name.lower()

            # Base score from matching quality
            if term_lower == name_lower:
                base_score = 1.0
            elif term_lower in name_lower:
                base_score = 0.5 + 0.4 * (len(term) / max(len(name), 1))
            else:
                base_score = 0.4

            # Length bonus for specific multi-word mentions
            length_bonus = min(0.3, len(term_lower.split()) * 0.1)

            # High-confidence partition alignment boost
            partition_boost = 0.0
            if pid is not None and int(pid) in allowed_partition_ids:
                partition_boost = 0.5
            elif eid in allowed_entity_ids:
                partition_boost = 0.5

            total_confidence = base_score + length_bonus + partition_boost

            if total_confidence < min_confidence:
                continue

            # Keep highest scoring instance for each entity
            if eid not in scored_seeds or total_confidence > scored_seeds[eid].confidence:
                scored_seeds[eid] = ResolvedSeed(
                    entity_id=eid,
                    entity_name=name,
                    description=desc,
                    confidence=total_confidence,
                    matched_term=term,
                    partition_id=int(pid) if pid is not None else None,
                )

        ranked = sorted(scored_seeds.values(), key=lambda s: s.confidence, reverse=True)
        return ranked[:max_seeds]
