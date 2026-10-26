from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, List, Optional

try:
    import requests
except ImportError:
    requests = None

from src.traversal.evidence_context import EvidenceBundle
from src.traversal.path_scoring_traverser import relation_intent_name, relation_matches_query


CITATION_REGEX = re.compile(r"\[P\d+\.E\d+\]")


@dataclass(frozen=True)
class GroundedAnswer:
    """Structured response from LLM generation with provenance metadata."""

    query: str
    answer: str
    citations_used: tuple[str, ...]
    model_name: str
    is_mock: bool = False
    latency_ms: float = 0.0
    raw_context: str = ""
    was_skipped: bool = False
    skip_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations_used": list(self.citations_used),
            "model_name": self.model_name,
            "is_mock": self.is_mock,
            "latency_ms": self.latency_ms,
            "was_skipped": self.was_skipped,
            "skip_reason": self.skip_reason,
        }


class OllamaClient:
    """Client for local/remote Ollama instance supporting grounded Q&A."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3.2:1b")
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if the Ollama service is active and reachable."""
        try:
            if requests is not None:
                resp = requests.get(f"{self.base_url}/api/tags", timeout=3.0)
                return resp.status_code == 200
            req = urllib.request.Request(f"{self.base_url}/api/tags", headers={"User-Agent": "Enhanced-KG-LLM"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """List models installed in the local Ollama instance."""
        try:
            if requests is not None:
                resp = requests.get(f"{self.base_url}/api/tags", timeout=3.0)
                if resp.status_code == 200:
                    data = resp.json()
                    return [m.get("name", "") for m in data.get("models", [])]
            else:
                req = urllib.request.Request(f"{self.base_url}/api/tags", headers={"User-Agent": "Enhanced-KG-LLM"})
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            pass
        return []

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
    ) -> str:
        """Direct text generation call via Ollama /api/generate."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if system:
            payload["system"] = system

        if requests is not None:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json", "User-Agent": "Enhanced-KG-LLM"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()

    def answer_with_evidence(
        self,
        query: str,
        bundle: EvidenceBundle,
        mock_if_unavailable: bool = True,
    ) -> GroundedAnswer:
        """Prompt LLM using the structured Knowledge Graph EvidenceBundle context.

        If Ollama is not running and mock_if_unavailable=True, synthesizes a clean,
        grounded mock response referencing bundle edges so tests and pipelines don't crash.
        """
        if not bundle.paths:
            return GroundedAnswer(
                query=query,
                answer=(
                    "No supporting evidence was found in the retrieved partition scope. "
                    "Unable to ground an answer from the available knowledge graph evidence."
                ),
                citations_used=(),
                model_name=f"{self.model_name} (skipped: no evidence)",
                raw_context=bundle.context,
                was_skipped=True,
                skip_reason="no evidence",
            )

        direct_answer = self._answer_from_relation_evidence(query, bundle)
        if direct_answer:
            answer, citations = direct_answer
            return GroundedAnswer(
                query=query,
                answer=answer,
                citations_used=citations,
                model_name=f"{self.model_name} (direct KG evidence)",
                raw_context=bundle.context,
                was_skipped=True,
                skip_reason="direct relation in KG evidence",
            )

        system_instruction = (
            "You are a factual knowledge-graph reasoning assistant. "
            "You must answer the user's question using ONLY the provided knowledge-graph evidence. "
            "For every factual assertion in your answer, you MUST cite the corresponding citation tag "
            "exactly as provided (e.g. [P1.E1]). "
            "Answer concisely and put the citation immediately after each supported fact. "
            "If the evidence does not contain sufficient facts to answer, explicitly state that. "
            "Do not infer or hallucinate relationships not shown in the evidence."
        )

        user_prompt = f"{bundle.context}\n\nBased strictly on the evidence above, answer: {query}"

        t0 = time.perf_counter()
        if self.is_available():
            try:
                raw_response = self.generate(prompt=user_prompt, system=system_instruction)
                latency_ms = (time.perf_counter() - t0) * 1000
                citations = tuple(dict.fromkeys(CITATION_REGEX.findall(raw_response)))
                valid_citations = self._matching_evidence_citations(query, raw_response, bundle)
                if not citations:
                    citations = valid_citations
                    if citations:
                        raw_response = f"{raw_response.rstrip()} {' '.join(f'[{c}]' for c in citations)}"
                return GroundedAnswer(
                    query=query,
                    answer=raw_response,
                    citations_used=citations,
                    model_name=self.model_name,
                    is_mock=False,
                    latency_ms=latency_ms,
                    raw_context=bundle.context,
                )
            except Exception as e:
                if not mock_if_unavailable:
                    raise
                print(f"[OllamaClient] Generation error: {e}. Falling back to synthetic grounded answer.")

        if not mock_if_unavailable:
            raise ConnectionError(
                f"Cannot connect to Ollama at '{self.base_url}'. "
                "Ensure Ollama is running ('ollama serve') or set mock_if_unavailable=True."
            )

        # Synthetic fallback grounded generation
        synthetic_answer = self._synthesize_fallback(query, bundle)
        latency_ms = (time.perf_counter() - t0) * 1000
        citations = tuple(dict.fromkeys(CITATION_REGEX.findall(synthetic_answer)))
        return GroundedAnswer(
            query=query,
            answer=synthetic_answer,
            citations_used=citations,
            model_name=f"{self.model_name} (offline-mock)",
            is_mock=True,
            latency_ms=latency_ms,
            raw_context=bundle.context,
        )

    @staticmethod
    def _synthesize_fallback(query: str, bundle: EvidenceBundle) -> str:
        """Create a verifiable, citation-annotated grounded response without calling Ollama."""
        if not bundle.paths:
            return (
                f"No supporting evidence was found in the retrieved partition scope for '{query}'. "
                "Unable to ground answer without hallucinating."
            )

        claims = []
        for p_idx, path in enumerate(bundle.paths, 1):
            edge_strs = []
            for e_idx, edge in enumerate(path.edges, 1):
                citation = f"[P{p_idx}.E{e_idx}]"
                edge_strs.append(f"{citation} {edge.source_name} has relation '{edge.relation_name}' with {edge.target_name}")
            claims.append(" -> ".join(edge_strs))

        body = "\n".join(f"• {c}." for c in claims)
        return (
            f"Based strictly on verified Knowledge Graph evidence:\n{body}\n\n"
            f"(Note: Generated via offline grounded synthesizer. To run live LLM inference, start Ollama via 'ollama serve')."
        )

    @staticmethod
    def _matching_evidence_citations(
        query: str,
        answer: str,
        bundle: EvidenceBundle,
    ) -> tuple[str, ...]:
        """Attach citations when the answer names the target of an edge whose source is queried."""
        normalize = lambda value: re.sub(r"[^a-z0-9]+", "", value.lower())
        query_text = normalize(query)
        answer_text = normalize(answer)
        matched: list[str] = []
        for path_number, path in enumerate(bundle.paths, start=1):
            for edge_number, edge in enumerate(path.edges, start=1):
                source = normalize(edge.source_name)
                target = normalize(edge.target_name)
                if (
                    source
                    and target
                    and source in query_text
                    and target in answer_text
                    and relation_matches_query(edge.relation_name, query)
                ):
                    matched.append(f"P{path_number}.E{edge_number}")
        return tuple(dict.fromkeys(matched))

    @staticmethod
    def _answer_from_relation_evidence(
        query: str,
        bundle: EvidenceBundle,
    ) -> tuple[str, tuple[str, ...]] | None:
        """Build a short, cited answer from a direct edge when the query names its source."""
        normalize = lambda value: re.sub(r"[^a-z0-9]+", "", value.lower())
        query_text = normalize(query)
        candidates = []
        for path_number, path in enumerate(bundle.paths, start=1):
            for edge_number, edge in enumerate(path.edges, start=1):
                source = normalize(edge.source_name)
                if source and source in query_text and relation_matches_query(edge.relation_name, query):
                    label = f"P{path_number}.E{edge_number}"
                    candidates.append((path.score, edge, label))
        if not candidates:
            return None

        _, edge, label = max(candidates, key=lambda candidate: candidate[0])
        intent = relation_intent_name(edge.relation_name)
        if intent == "author":
            answer = (
                f"The KG lists {edge.target_name} ({edge.target_id}) as the author of "
                f"{edge.source_name}. [{label}]"
            )
        elif intent == "place of birth":
            answer = f"{edge.source_name} was born in {edge.target_name}. [{label}]"
        elif intent == "capital":
            answer = f"The capital of {edge.source_name} is {edge.target_name}. [{label}]"
        else:
            answer = (
                f"The KG evidence links {edge.source_name} to {edge.target_name} "
                f"through '{edge.relation_name}'. [{label}]"
            )
        return answer, (label,)
