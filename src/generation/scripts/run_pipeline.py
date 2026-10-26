import argparse
import os
import sys
import time

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Set offline mode default for huggingface to use cached models seamlessly
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from config.settings import SUBG_DIR
from src.generation.pipeline import EnhancedKGRAGPipeline, PipelineResult


def main():
    parser = argparse.ArgumentParser(description="Run Enhanced-KG-LLM End-to-End Pipeline with Interactive Progress")
    parser.add_argument("--query", type=str, default=None,
                        help="Natural language query (prompts interactively when omitted)")
    parser.add_argument("--seed", type=str, default=None,
                        help="Optional explicit seed entity ID (default: auto-detected via EntitySeedResolver)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of candidate subgraphs to retrieve")
    parser.add_argument("--subg-dir", type=str, default=SUBG_DIR,
                        help="Path to subgraph partition artifacts directory (loose coupling)")
    parser.add_argument("--filter-mode", type=str, default="auto", choices=["auto", "partition_id", "entity_id"],
                        help="Traversal Cypher filter mode: 'partition_id', 'entity_id', or 'auto'")
    parser.add_argument("--model", type=str, default=None,
                        help="Ollama LLM model name (default: from $OLLAMA_MODEL or llama3.2)")
    parser.add_argument("--ollama-url", type=str, default=None,
                        help="Ollama base URL (default: http://localhost:11434)")
    args = parser.parse_args()

    if args.query is None:
        try:
            args.query = input("\nEnter your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nNo query provided. Exiting.")
            return
        if not args.query:
            print("No query provided. Exiting.")
            return

    console = Console()

    # 1. Header Panel
    header_content = (
        f"[bold cyan]Query:[/bold cyan] [white]'{args.query}'[/white]\n"
        f"[bold cyan]Seed Mode:[/bold cyan] [yellow]{args.seed if args.seed else 'AUTO-RESOLVE (from query text)'}[/yellow]   "
        f"[bold cyan]Top-K Partitions:[/bold cyan] [green]{args.top_k}[/green]   "
        f"[bold cyan]Filter Mode:[/bold cyan] [magenta]{args.filter_mode}[/magenta]\n"
        f"[bold cyan]Partition Artifacts:[/bold cyan] [dim]{args.subg_dir}[/dim]\n"
        f"[bold cyan]LLM Engine:[/bold cyan] [blue]{args.model or os.getenv('OLLAMA_MODEL', 'llama3.2')}[/blue] @ [dim]{args.ollama_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}[/dim]"
    )
    console.print(Panel(header_content, title="[bold green]🚀 ENHANCED-KG-LLM : MASTER RAG PIPELINE[/bold green]", box=box.ROUNDED, expand=False))

    pipeline = EnhancedKGRAGPipeline(
        subg_dir=args.subg_dir,
        filter_mode=args.filter_mode,
    )
    if args.model or args.ollama_url:
        pipeline.ollama.model_name = args.model or pipeline.ollama.model_name
        pipeline.ollama.base_url = args.ollama_url or pipeline.ollama.base_url

    # Progress tracking callback
    def progress_callback(event: str, data: dict):
        if event == "retrieval_start":
            console.rule("[bold blue]Stage 1: Parallel Hybrid Partition Retrieval[/bold blue]")
            console.print("  [cyan]• Routing query across 3,625 partition summaries (Dense Centroids + BM25Okapi)...[/cyan]")

        elif event == "retrieval_done":
            hits = data.get("hits", [])
            elapsed = data.get("elapsed_ms", 0.0)
            console.print(f"  [green]✔ Retrieved Top-{len(hits)} Subgraphs in {elapsed:.2f} ms[/green]")

            table = Table(title="Top Retrieved Partitions", box=box.SIMPLE_HEAVY, show_lines=False)
            table.add_column("Rank", justify="center", style="bold yellow")
            table.add_column("Partition ID", justify="center", style="bold cyan")
            table.add_column("Fused Score", justify="right", style="green")
            table.add_column("Dense Sim", justify="right", style="dim")
            table.add_column("BM25 Score", justify="right", style="dim")
            table.add_column("Nodes in Partition", justify="right", style="magenta")
            table.add_column("Sample Entities", style="dim")

            for rank, hit in enumerate(hits, 1):
                pid = str(hit.get("partition_id", hit.get("community_id")))
                fused = f"{hit.get('score', 0.0):.4f}"
                dense = f"{hit.get('vector_similarity', 0.0):.4f}"
                bm25 = f"{hit.get('bm25_score', 0.0):.4f}"
                size = f"{hit.get('size', 0):,}"
                seeds = ", ".join(hit.get("seed_nodes", hit.get("top_entities", []))[:3])
                table.add_row(str(rank), pid, fused, dense, bm25, size, seeds)

            console.print(table)

        elif event == "seed_resolution_start":
            console.rule("[bold purple]Stage 2: Query Entity & Multi-Seed Resolution[/bold purple]")
            console.print("  [cyan]• Extracting candidate mentions and cross-referencing with partition scope...[/cyan]")

        elif event == "seed_resolution_done":
            resolved = data.get("resolved_seeds", [])
            seeds = data.get("seeds", [])
            elapsed = data.get("elapsed_ms", 0.0)
            console.print(f"  [green]✔ Resolved {len(seeds)} Traversal Seed(s) in {elapsed:.2f} ms[/green]")

            if resolved:
                table = Table(title="Resolved Query Seed Entities", box=box.SIMPLE_HEAVY)
                table.add_column("Entity Name", style="bold white")
                table.add_column("Wikidata ID", style="bold yellow", justify="center")
                table.add_column("Confidence", justify="right", style="green")
                table.add_column("Matched Mention", style="cyan")
                table.add_column("Partition ID", justify="center", style="dim")
                table.add_column("Description", style="dim")

                for s in resolved:
                    pid_str = str(s.partition_id) if s.partition_id is not None else "All"
                    table.add_row(
                        s.entity_name,
                        s.entity_id,
                        f"{s.confidence:.2f}",
                        f"'{s.matched_term}'",
                        pid_str,
                        s.description[:60] + ("..." if len(s.description) > 60 else "")
                    )
                console.print(table)
            else:
                console.print(f"  [yellow]• Explicit / Fallback Seed(s): {list(seeds)}[/yellow]")

        elif event == "traversal_start":
            console.rule("[bold gold1]Stage 3: Partition-Bounded Multi-Seed Traversal[/bold gold1]")
            seeds = list(data.get("seeds", []))
            console.print(f"  [cyan]• Traversing Neo4j starting from seeds: {seeds} (max hops=2, beam width=3)...[/cyan]")

        elif event == "traversal_done":
            resp = data.get("response")
            elapsed = data.get("elapsed_ms", 0.0)
            paths = resp.bundle.paths if resp and resp.bundle else ()
            console.print(f"  [green]✔ Found {len(paths)} High-Scoring Evidence Paths in {elapsed:.2f} ms[/green]")

            table = Table(title="Extracted Knowledge Graph Evidence Paths", box=box.SIMPLE_HEAVY)
            table.add_column("Path #", justify="center", style="bold yellow")
            table.add_column("Score", justify="right", style="green")
            table.add_column("Hops", justify="center", style="cyan")
            table.add_column("Traversed Graph Path (Source ──[relation]──► Target)", style="bold white")

            for p_idx, path in enumerate(paths, 1):
                edge_reprs = []
                for e_idx, edge in enumerate(path.edges, 1):
                    edge_reprs.append(
                        f"[yellow]{edge.source_name}[/yellow] ──[bold cyan]{edge.relation_name}[/bold cyan]──► [yellow]{edge.target_name}[/yellow]"
                    )
                chain = "   ".join(edge_reprs)
                table.add_row(f"P{p_idx}", f"{path.score:.3f}", str(len(path.edges)), chain)

            console.print(table)

        elif event == "context_done":
            console.rule("[bold magenta]Stage 4: Structured Context & Citation Assembly[/bold magenta]")
            bundle = data.get("bundle")
            console.print("  [cyan]• Assembled compact evidence context with citation tags [P1.E1]...[/cyan]")

        elif event == "llm_start":
            console.rule("[bold green]Stage 5: Grounded LLM Generation (Ollama)[/bold green]")
            model = data.get("model", "llama3.2")
            console.print(f"  [cyan]• Feeding structured evidence bundle to {model}...[/cyan]")

        elif event == "llm_done":
            answer_obj = data.get("grounded_answer")
            elapsed = data.get("elapsed_ms", 0.0)
            status_text = "[yellow](Offline Grounded Fallback)[/yellow]" if answer_obj.is_mock else "[green](Live Ollama Response)[/green]"
            console.print(f"  [green]✔ LLM Generation Complete in {elapsed:.2f} ms {status_text}[/green]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("[bold cyan]Executing Enhanced-KG-LLM Pipeline...", total=None)
        
        result = pipeline.run(
            query=args.query,
            seed_entity_ids=[args.seed] if args.seed else None,
            top_k_partitions=args.top_k,
            on_progress=progress_callback,
        )

    # Final Formatted Output Panel
    console.print()
    answer_text = result.grounded_answer.answer
    citations = ", ".join(result.grounded_answer.citations_used) if result.grounded_answer.citations_used else "None cited"
    panel_body = (
        f"[bold white]{answer_text}[/bold white]\n\n"
        f"[bold cyan]Citations Detected:[/bold cyan] [yellow]{citations}[/yellow]\n"
        f"[bold cyan]Inference Engine:[/bold cyan] [dim]{result.grounded_answer.model_name}[/dim]"
    )
    console.print(Panel(panel_body, title="[bold green]★ FINAL GROUNDED ANSWER[/bold green]", box=box.DOUBLE))

    # Latency Breakdown Table
    timing_table = Table(title="Pipeline Latency Breakdown", box=box.ROUNDED)
    timing_table.add_column("Pipeline Stage", style="bold cyan")
    timing_table.add_column("Duration (ms)", justify="right", style="bold yellow")
    timing_table.add_column("% of Total", justify="right", style="green")

    total_time = result.total_latency_ms
    stage_names = {
        "retrieval_ms": "1. Subgraph Partition Retrieval",
        "seed_resolution_ms": "2. Entity Seed Resolution",
        "traversal_ms": "3. Partition-Bounded Traversal",
        "llm_ms": "4. Grounded LLM Generation",
    }
    for key, name in stage_names.items():
        dur = result.timings_ms.get(key, 0.0)
        pct = (dur / total_time * 100) if total_time > 0 else 0
        timing_table.add_row(name, f"{dur:.2f} ms", f"{pct:.1f}%")

    timing_table.add_section()
    timing_table.add_row("[bold white]Total Round-Trip[/bold white]", f"[bold green]{total_time:.2f} ms[/bold green]", "100.0%")
    console.print(timing_table)
    console.print("\n[bold green]Pipeline finished successfully.[/bold green]\n")


if __name__ == "__main__":
    main()
