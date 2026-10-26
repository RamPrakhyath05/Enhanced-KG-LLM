"""Run multiple KG-RAG queries in one process, reusing the loaded index/model.

Start with:
    python -m src.generation.scripts.run_pipeline_session
Enter a blank query or type 'quit' to exit.
"""

from __future__ import annotations

import os
import sys

# Ensure the repository root is importable when invoked as a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Set offline mode default for huggingface to use cached models seamlessly
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.generation.pipeline import EnhancedKGRAGPipeline


def main() -> None:
    pipeline = EnhancedKGRAGPipeline()
    print("Enhanced-KG-LLM session ready. Enter a query; blank line or 'quit' exits.")
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
    finally:
        client = pipeline._db_client
        if client is not None and hasattr(client, "close"):
            client.close()


if __name__ == "__main__":
    main()
