"""Run multiple KG-RAG queries in one process, reusing the loaded index/model.

Start with:
    python scripts/run_pipeline_session.py
    # or: python -m scripts.run_pipeline_session
Enter a blank query or type 'quit' to exit.
"""

from __future__ import annotations

import os
import sys

# Ensure the repository root is importable when invoked as a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set offline mode default for huggingface to use cached models seamlessly
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.generation.pipeline import EnhancedKGRAGPipeline, PipelineResult


def format_llm_grounding_payload(result: PipelineResult) -> str:
    """Format the exact context, instructions, and prompt payload passed to the LLM for grounding."""
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
    user_prompt = f"{result.bundle.context}\n\nBased strictly on the evidence above, answer: {result.query}"

    evidence_lines = []
    # 1. Topic Entities
    for cit in result.bundle.citations:
        if cit.get("label", "").startswith("S"):
            lbl = cit["label"]
            name = cit.get("entity_name", cit.get("source_id", ""))
            eid = cit.get("source_id", "")
            d = cit.get("description", "")
            d_short = f" (description: {d[:120]}...)" if d else ""
            evidence_lines.append(f"  • [{lbl}] Topic Entity: {name} ({eid}){d_short}")

    # 2. Traversed Paths
    for p_idx, path in enumerate(result.bundle.paths, 1):
        for e_idx, edge in enumerate(path.edges, 1):
            citation = f"[P{p_idx}.E{e_idx}]"
            desc = f" (target description: {edge.target_description[:120]}...)" if edge.target_description else ""
            evidence_lines.append(
                f"  • {citation} {edge.source_name} ({edge.source_id}) "
                f"--[{edge.relation_name} ({edge.relation_id})]--> "
                f"{edge.target_name} ({edge.target_id}){desc}"
            )
    evidence_str = "\n".join(evidence_lines) if evidence_lines else "  • (No traversed paths in evidence bundle)"

    lines = [
        "",
        "╔" + "═" * 73 + "╗",
        "║                     DATA SENT TO LLM FOR GROUNDING                      ║",
        "╚" + "═" * 73 + "╝",
        "",
        "▶ [1] SYSTEM INSTRUCTION (PROMPT GUIDANCE):",
        f"  \"{system_instruction}\"",
        "",
        "▶ [2] STRUCTURED GRAPH EVIDENCE TRIPLES & CITATION IDS:",
        evidence_str,
        "",
        "▶ [3] CANDIDATE TARGET ENTITY IDS (GRAPH GROUNDING TARGETS):",
        f"  {list(result.bundle.candidate_answer_ids) or 'None'}",
        "",
        "▶ [4] FULL USER PROMPT PAYLOAD (RAW CONTEXT + QUESTION):",
        "─" * 75,
        user_prompt,
        "─" * 75,
        "",
        "▶ [5] GROUNDED RESPONSE & AUDIT TRAIL:",
        f"  • Engine Mode: {result.grounded_answer.model_name}",
        f"  • Citations Detected: {list(result.grounded_answer.citations_used) or 'None'}",
        f"  • Grounding Latency: {result.timings_ms.get('llm_ms', 0):.2f} ms",
        "═" * 75,
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    print("=" * 75)
    print("ENHANCED-KG-LLM : PERSISTENT QUERY SESSION")
    print("=" * 75)
    pipeline = EnhancedKGRAGPipeline()
    pipeline.warmup(verbose=True)

    print("-" * 75)
    print("Enhanced-KG-LLM session ready. Enter a query; blank line or 'quit' exits.")
    print("-" * 75)
    try:
        while True:
            try:
                query = input("\nQuery> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query or query.lower() in {"quit", "exit"}:
                break

            try:
                result = pipeline.run(query)
            except (ConnectionError, RuntimeError) as exc:
                print(f"Pipeline stopped: {exc}")
                continue
            print(result.summary())
            print(format_llm_grounding_payload(result))
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
