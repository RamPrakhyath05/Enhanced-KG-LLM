from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from src.db.neo4j_client import Neo4jClient
from src.generation.ollama_client import GroundedAnswer, OllamaClient
from src.query_subgraph.subgraph_retriever import SubgraphRetriever
from src.traversal.evidence_context import EvidenceBundle, EvidenceContextBuilder
from src.traversal.partition_rag_adapter import PartitionRAGTraversalAdapter
from src.traversal.path_scoring_traverser import PathScoringConfig, PathScoringTraverser
from src.traversal.seed_resolver import EntitySeedResolver, ResolvedSeed


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


class FallbackGraphClient:
    """Mock offline graph client providing realistic multi-hop neighbors when Neo4j service is paused."""

    def run_query(self, query: str, parameters: dict = None) -> list[dict]:
        params = parameters or {}

        # 1. Candidate Entity Search Handling
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
                        "matched_term": c,
                    })
                if "tasmania" in cl:
                    results.append({
                        "id": "Q34366",
                        "name": "Tasmania",
                        "description": "Island state of Australia.",
                        "partition_id": 1187,
                        "matched_term": c,
                    })
                if "continent" in cl or "oceania" in cl:
                    results.append({
                        "id": "Q3960",
                        "name": "australian continent",
                        "description": "Continent of Australia.",
                        "partition_id": 1187,
                        "matched_term": c,
                    })
                if "yima" in cl or "king" in cl:
                    results.append({
                        "id": "Q28954",
                        "name": "King Yima",
                        "description": "Mythological hero in Persian tradition.",
                        "partition_id": 88,
                        "matched_term": c,
                    })
                if "persian" in cl or "literature" in cl:
                    results.append({
                        "id": "Q8261",
                        "name": "Persian literature",
                        "description": "Literary traditions and texts written in Persian.",
                        "partition_id": 88,
                        "matched_term": c,
                    })
            return results

        # 2. Multi-hop Neighbor Traversal Handling
        source_ids = params.get("source_ids", ["Q408"])
        records = []
        for sid in source_ids:
            if sid == "Q28954":  # King Yima
                records.extend([
                    {
                        "source_id": "Q28954",
                        "source_name": "King Yima",
                        "relation_id": "P1441",
                        "relation_name": "present in work",
                        "target_id": "Q8279",
                        "target_name": "Book of Kings (Shahnameh)",
                        "target_description": "Epic poem by Ferdowsi and core work of Persian literature.",
                    },
                    {
                        "source_id": "Q8279",
                        "source_name": "Book of Kings (Shahnameh)",
                        "relation_id": "P50",
                        "relation_name": "author",
                        "target_id": "Q43459",
                        "target_name": "Ferdowsi",
                        "target_description": "Celebrated Persian poet.",
                    },
                ])
            elif sid == "Q8261":  # Persian literature
                records.extend([
                    {
                        "source_id": "Q8261",
                        "source_name": "Persian literature",
                        "relation_id": "P527",
                        "relation_name": "has part",
                        "target_id": "Q8279",
                        "target_name": "Book of Kings (Shahnameh)",
                        "target_description": "Epic poem by Ferdowsi and core work of Persian literature.",
                    },
                ])
            else:  # Australia / Default
                records.extend([
                    {
                        "source_id": sid,
                        "source_name": "Commonwealth Australia",
                        "relation_id": "P361",
                        "relation_name": "part of",
                        "target_id": "Q3960",
                        "target_name": "australian continent",
                        "target_description": "The continent of Australia sits on Australia's continental plate.",
                    },
                    {
                        "source_id": "Q3960",
                        "source_name": "australian continent",
                        "relation_id": "P527",
                        "relation_name": "has part",
                        "target_id": "Q34366",
                        "target_name": "Tasmania",
                        "target_description": "Tasmania is an island state of Australia located 240 km south of mainland.",
                    },
                    {
                        "source_id": sid,
                        "source_name": "Commonwealth Australia",
                        "relation_id": "P2633",
                        "relation_name": "geography of topic",
                        "target_id": "Q1134461",
                        "target_name": "coast in australia",
                        "target_description": "The geography of Australia encompasses diverse coastal and oceanic regions.",
                    },
                ])
        return records


class EnhancedKGRAGPipeline:
    """End-to-End Master KG-RAG Pipeline."""

    def __init__(
        self,
        subg_dir: Optional[str] = None,
        db_client: Optional[Any] = None,
        traversal_config: Optional[PathScoringConfig] = None,
        ollama_client: Optional[OllamaClient] = None,
        filter_mode: str = "auto",
        allow_offline_fallback: bool = False,
    ) -> None:
        default_dir = "data/Partitions" if os.path.exists(os.path.join("data/Partitions", "centroids.npy")) else "data/subg"
        self.subg_dir = subg_dir or os.getenv("SUBG_DIR", default_dir)
        self.retriever = SubgraphRetriever(subg_dir=self.subg_dir)
        
        # Lazy/managed DB client with graceful offline fallback
        self._db_client = db_client
        self.allow_offline_fallback = allow_offline_fallback
        self._seed_resolver = None
        self.traversal_config = traversal_config or PathScoringConfig(
            max_hops=2, beam_width=3, max_paths=3, filter_mode=filter_mode
        )
        self.ollama = ollama_client or OllamaClient()
        self.context_builder = EvidenceContextBuilder()

    @property
    def seed_resolver(self) -> EntitySeedResolver:
        if self._seed_resolver is None:
            self._seed_resolver = EntitySeedResolver(db_client=self.db_client)
        return self._seed_resolver

    @property
    def db_client(self) -> Any:
        if self._db_client is None:
            try:
                if not os.getenv("NEO4J_PASSWORD"):
                    raise RuntimeError("NEO4J_PASSWORD is not set")
                client = Neo4jClient()
                res = client.run_query("RETURN 1 AS ping")
                if res and res[0].get("ping") == 1:
                    self._db_client = client
                else:
                    client.close()
                    raise ConnectionError("Neo4j health check did not return the expected ping result")
            except Exception as exc:
                if not self.allow_offline_fallback:
                    raise ConnectionError(
                        "Could not connect to Neo4j. Check NEO4J_URI, NEO4J_USER, "
                        "NEO4J_PASSWORD, and NEO4J_DATABASE. For a clearly labeled "
                        "mock run, pass allow_offline_fallback=True."
                    ) from exc
                print(f"[EnhancedKGRAGPipeline] Neo4j unavailable ({exc}). Using explicit offline graph fallback.")
                self._db_client = FallbackGraphClient()
        return self._db_client

    def run(
        self,
        query: str,
        seed_entity_ids: Optional[Sequence[str]] = None,
        top_k_partitions: int = 5,
        filter_mode: Optional[str] = None,
        on_progress: Optional[Any] = None,
    ) -> PipelineResult:
        """Run complete query through retrieval, traversal, context assembly, and LLM grounding."""
        timings: dict[str, float] = {}

        def _notify(event: str, data: Any = None):
            if callable(on_progress):
                try:
                    on_progress(event, data)
                except Exception:
                    pass

        # 1. Hybrid Subgraph Retrieval
        _notify("retrieval_start", {"query": query, "top_k": top_k_partitions})
        t0 = time.perf_counter()
        hits = self.retriever.search(query, top_k=top_k_partitions)
        timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000
        _notify("retrieval_done", {"hits": hits, "elapsed_ms": timings["retrieval_ms"]})

        # 2. Auto-resolve seed entities directly from query & partition scope
        _notify("seed_resolution_start", {"query": query})
        t0 = time.perf_counter()
        resolved_seeds = []
        if not seed_entity_ids:
            resolved_seeds = self.seed_resolver.resolve(
                query=query,
                partition_hits=hits,
                max_seeds=3,
            )
            if resolved_seeds:
                seed_entity_ids = tuple(s.entity_id for s in resolved_seeds)
            elif hits:
                top_seeds = hits[0].get("seed_nodes") or hits[0].get("top_entities") or []
                seed_entity_ids = tuple(top_seeds[:1]) if top_seeds else ("Q408",)
        else:
            seed_entity_ids = tuple(seed_entity_ids)
        timings["seed_resolution_ms"] = (time.perf_counter() - t0) * 1000
        _notify("seed_resolution_done", {
            "seeds": seed_entity_ids,
            "resolved_seeds": resolved_seeds,
            "elapsed_ms": timings["seed_resolution_ms"],
        })

        seed_partition_ids = {
            int(seed.partition_id)
            for seed in resolved_seeds
            if seed.partition_id is not None
        }
        if not resolved_seeds and seed_entity_ids and not self.allow_offline_fallback:
            seed_rows = self.db_client.run_query(
                "MATCH (e:Entity) WHERE e.Entity_ID IN $entity_ids "
                "RETURN DISTINCT e.partition_id AS partition_id",
                {"entity_ids": list(seed_entity_ids)},
            )
            seed_partition_ids.update(
                int(row["partition_id"])
                for row in seed_rows
                if row.get("partition_id") is not None
            )

        def include_seed_partitions(base_hits: list[dict]) -> list[dict]:
            """Keep traversal bounded while ensuring its seed nodes are in scope."""
            traversal_hits = list(base_hits)
            included_partition_ids = {
                int(hit.get("partition_id", hit.get("community_id")))
                for hit in traversal_hits
                if hit.get("partition_id", hit.get("community_id")) is not None
            }
            for partition_id in sorted(seed_partition_ids - included_partition_ids):
                entity_ids = self.retriever.entity_map.get(partition_id, [])
                if not entity_ids:
                    continue
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

        # 3. Partition-Bounded Multi-Seed Traversal
        _notify("traversal_start", {"seeds": seed_entity_ids, "partition_hits": traversal_hits})
        if filter_mode:
            self.traversal_config = PathScoringConfig(
                max_hops=self.traversal_config.max_hops,
                beam_width=self.traversal_config.beam_width,
                max_paths=self.traversal_config.max_paths,
                filter_mode=filter_mode,
            )

        # Retrieval has already loaded the sentence encoder; reuse it for edge
        # scoring instead of loading the same weights a second time.
        traverser = PathScoringTraverser(
            client=self.db_client,
            config=self.traversal_config,
            model=self.retriever.model,
        )
        adapter = PartitionRAGTraversalAdapter(traverser, context_builder=self.context_builder)

        t0 = time.perf_counter()
        traversal_resp = adapter.retrieve(
            query=query,
            seed_entity_ids=seed_entity_ids,
            partition_hits=traversal_hits,
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
                )
                timings["traversal_ms"] += (time.perf_counter() - t0) * 1000
        _notify("traversal_done", {
            "response": traversal_resp,
            "elapsed_ms": timings["traversal_ms"],
        })

        # 4. Context Assembly
        bundle = traversal_resp.bundle
        _notify("context_done", {"bundle": bundle})

        # 5. Grounded LLM Generation
        _notify("llm_start", {"bundle": bundle, "model": self.ollama.model_name})
        t0 = time.perf_counter()
        grounded_answer = self.ollama.answer_with_evidence(
            query=query,
            bundle=bundle,
            mock_if_unavailable=self.allow_offline_fallback,
        )
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
        )
