"""Ollama Client for Grounded LLM Generation with Citation Provenance.

Interfaces with a local or remote Ollama server (default: http://localhost:11434).
Feeds structured EvidenceBundle contexts from traversal and enforces strict
grounding policies and citation provenance tags ([P1.E1]).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional
import requests

from src.traversal.evidence_context import EvidenceBundle


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

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations_used": list(self.citations_used),
            "model_name": self.model_name,
            "is_mock": self.is_mock,
            "latency_ms": self.latency_ms,
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
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3.2")
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if the Ollama service is active and reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """List models installed in the local Ollama instance."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
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

        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
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
        system_instruction = (
            "You are a factual knowledge-graph reasoning assistant. "
            "You must answer the user's question using ONLY the provided knowledge-graph evidence. "
            "For every factual assertion in your answer, you MUST cite the corresponding citation tag "
            "exactly as provided (e.g. [P1.E1]). "
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
