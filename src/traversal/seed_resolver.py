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
    "according", "work", "works", "write", "writes", "wrote", "written",
    "author", "authored", "city", "birthplace", "born"
}

# Query verbs and relation/type words describe what the user is asking about;
# they should not become entity mentions (e.g. "the founder" or "capital").
QUERY_TERMS = {
    "direct", "directed", "director", "directing", "made", "make", "makes",
    "manufacturer", "manufactured", "found", "founder", "founded", "capital",
    "located", "location", "country", "film", "movie", "book", "company",
    "produced", "producer", "created", "creator", "wrote", "write", "written",
}
MENTION_STOP_WORDS = STOP_WORDS | QUERY_TERMS


QUERY_PREFIXES = (
    "what is a ", "what is an ", "what is the ", "what is ",
    "what are the ", "what are ",
    "when is the ", "when is ", "when was the ", "when was ",
    "who is the ", "who is ", "who was the ", "who was ",
    "tell me about the ", "tell me about ", "where is the ", "where is ",
)


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
        """Extract likely subject names, preserving natural entity phrases and prepositions."""
        q = query.strip()
        q_lower = q.lower()
        for prefix in QUERY_PREFIXES:
            if q_lower.startswith(prefix):
                q = q[len(prefix):].strip()
                break

        q_clean = re.sub(r"[\?\.!]+$", "", q).strip()
        q_norm = q_clean.replace("’", "'")
        raw_tokens = [t for t in re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", q_norm) if t]
        if not raw_tokens:
            return []

        candidates: list[str] = []

        # 1. Full subject phrase (e.g. "independence day of india", "india's independence day")
        if len(raw_tokens) > 1:
            candidates.append(q_norm)
            candidates.append(q_norm.title())
            if "'s" in q_norm:
                candidates.append(q_norm.replace("'s", "s"))
                candidates.append(q_norm.replace("'s", ""))

        # 2. Sliding n-grams from length 5 down to 1 over raw tokens (preserving prepositions like 'of' in phrases)
        for gram_size in range(min(5, len(raw_tokens)), 0, -1):
            for i in range(len(raw_tokens) - gram_size + 1):
                gram_tokens = raw_tokens[i : i + gram_size]
                phrase = " ".join(gram_tokens).strip()
                candidates.append(phrase)
                candidates.append(phrase.title())
                if "'s" in phrase:
                    no_apos = phrase.replace("'s", "").strip()
                    candidates.append(no_apos)
                    candidates.append(no_apos.title())

        # 3. Stop-word stripped n-grams for robust coverage of complex sentences
        significant = [w for w in raw_tokens if w.lower() not in MENTION_STOP_WORDS and len(w) >= 2]
        for gram_size in range(min(4, len(significant)), 0, -1):
            for i in range(len(significant) - gram_size + 1):
                phrase = " ".join(significant[i : i + gram_size]).strip()
                candidates.append(phrase)
                candidates.append(phrase.title())

        seen: set[str] = set()
        unique_candidates: list[str] = []
        filter_words = {"the", "is", "of", "a", "an", "in", "to", "for", "are", "was", "were", "and"}
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
        MATCH (e:Entity)
        WHERE e.Entity_Name IN $candidates
        RETURN e.Entity_ID AS id,
                        e.Entity_Name AS name,
                        e.Entity_Description AS description,
                        e.partition_id AS partition_id,
                        e.Entity_Name AS matched_term
        LIMIT 40
        """
        try:
            records = self.db_client.run_query(cypher, {"candidates": candidates[:80]})
        except Exception:
            records = []

        # Always supplement exact-name hits with aliases. A generic exact hit
        # (such as "the founder") must not suppress a relevant alias hit (Apple).
        alias_terms = list(dict.fromkeys(term.casefold() for term in candidates))
        alias_query = " OR ".join(
            '"' + term.replace("\\", "\\\\").replace('"', '\\"') + '"'
            for term in alias_terms[:40]
        )
        try:
            alias_records = self.db_client.run_query(
                "CALL db.index.fulltext.queryNodes('entity_alias_lookup', $query, {limit: 3000}) "
                "YIELD node, score "
                "RETURN node.Entity_ID AS id, node.Entity_Name AS name, "
                "node.Entity_Description AS description, node.partition_id AS partition_id, "
                "node.Entity_Aliases AS aliases, score",
                {"query": alias_query},
            )
            records.extend(alias_records)
        except Exception:
            pass

        # Score and rank candidates
        scored_seeds: dict[str, ResolvedSeed] = {}
        for rec in records:
            eid = str(rec.get("id", ""))
            name = str(rec.get("name") or eid)
            desc = str(rec.get("description") or "")
            pid = rec.get("partition_id")
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

            # Base score from matching quality
            if term_lower == name_lower or term_lower in aliases:
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

            # Description substance boost: comprehensive entities get prioritized over stubs
            desc_bonus = min(0.15, len(desc.split()) / 500.0) if desc else 0.0

            # Penalize entertainment homonyms (albums, songs, singles) for general concept queries
            media_penalty = 0.0
            query_lower = query.lower()
            if any(query_lower.startswith(p) for p in ("what is a ", "what is an ", "what is ")) and not any(m in query_lower for m in ("album", "song", "film", "movie", "band")):
                desc_sample = desc[:140].lower()
                if any(m in desc_sample for m in ("studio album", "compilation album", "single by", "song by", "film directed")):
                    media_penalty = 0.25

            total_confidence = base_score + length_bonus + partition_boost + desc_bonus - media_penalty

            if total_confidence < min_confidence:
                continue

            # Keep highest scoring instance for each entity
            if eid not in scored_seeds or total_confidence > scored_seeds[eid].confidence:
                scored_seeds[eid] = ResolvedSeed(
                    entity_id=eid,
                    # Show the user-facing mention when the KG's canonical label
                    # is a poor alias (e.g. Q2766 is labelled "Jesus phone").
                    entity_name=name if term_lower == name_lower else term.title(),
                    description=desc,
                    confidence=total_confidence,
                    matched_term=term,
                    partition_id=int(pid) if pid is not None else None,
                )

        ranked = sorted(scored_seeds.values(), key=lambda s: s.confidence, reverse=True)
        # For general queries with a single mention, allow up to 2 homonyms so graph traversal explores both.
        unique_candidate_phrases = {c.casefold() for c in candidates}
        max_per_mention = 2 if len(unique_candidate_phrases) <= 1 else 1
        selected: list[ResolvedSeed] = []
        mention_counts: dict[str, int] = {}
        for seed in ranked:
            mention_key = seed.matched_term.casefold()
            if mention_counts.get(mention_key, 0) >= max_per_mention:
                continue
            selected.append(seed)
            mention_counts[mention_key] = mention_counts.get(mention_key, 0) + 1
            if len(selected) >= max_seeds:
                break
        return selected
