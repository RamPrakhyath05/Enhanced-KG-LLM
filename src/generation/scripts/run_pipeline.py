import argparse
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.generation.pipeline import EnhancedKGRAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Run Enhanced-KG-LLM End-to-End Pipeline")
    parser.add_argument("--query", type=str, default="Australia Commonwealth country", help="Natural language query")
    parser.add_argument("--seed", type=str, default="Q408", help="Seed entity ID (e.g. Q408 for Australia)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of candidate subgraphs to retrieve")
    parser.add_argument("--subg-dir", type=str, default=None, help="Path to subgraph partition artifacts directory (loose coupling)")
    parser.add_argument("--filter-mode", type=str, default="auto", choices=["auto", "partition_id", "entity_id"],
                        help="Traversal Cypher filter mode: 'partition_id' (fast range index), 'entity_id' (fallback), or 'auto'")
    parser.add_argument("--model", type=str, default=None, help="Ollama LLM model name (default: from $OLLAMA_MODEL or llama3.2)")
    parser.add_argument("--ollama-url", type=str, default=None, help="Ollama base URL (default: http://localhost:11434)")
    args = parser.parse_args()

    print("\n" + "=" * 75)
    print("ENHANCED-KG-LLM: INITIALIZING MASTER END-TO-END PIPELINE")
    print("=" * 75)
    print(f"• Query:        {args.query}")
    print(f"• Seed ID:      {args.seed}")
    print(f"• Subgraph Dir: {args.subg_dir or os.getenv('SUBG_DIR', 'data/subg')}")
    print(f"• Filter Mode:  {args.filter_mode}")

    pipeline = EnhancedKGRAGPipeline(
        subg_dir=args.subg_dir,
        filter_mode=args.filter_mode,
    )
    if args.model or args.ollama_url:
        pipeline.ollama.model_name = args.model or pipeline.ollama.model_name
        pipeline.ollama.base_url = args.ollama_url or pipeline.ollama.base_url

    print("\nExecuting End-to-End Pipeline...")
    result = pipeline.run(
        query=args.query,
        seed_entity_ids=[args.seed],
        top_k_partitions=args.top_k,
    )

    print("\n" + result.summary())


if __name__ == "__main__":
    main()
