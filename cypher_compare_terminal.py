#!/usr/bin/env python3
"""
Pull query-relevant edges from Neo4j using Cypher only, compare with the latest
terminal output edges, and print relevance metrics in table form.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable

from neo4j import GraphDatabase


STOP_WORDS = {
    "who", "what", "where", "when", "how", "why",
    "is", "are", "was", "were", "am",
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "from", "by", "with",
    "and", "or", "but", "not", "no", "nor",
    "that", "this", "it", "its", "has", "have", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "shall",
    "be", "been", "being", "about", "which", "there",
    "tell", "me", "give", "list", "find", "show", "get", "all",
}

CYPHER_RELEVANT_EDGES = """
UNWIND $terms AS term
MATCH (a)-[r]->(b)
WITH term, a, b, r,
     toLower(coalesce(a.Entity_Name, "")) AS a_name,
     toLower(coalesce(b.Entity_Name, "")) AS b_name,
     [x IN coalesce(a.Entity_Aliases, []) | toLower(toString(x))] AS a_aliases,
     [x IN coalesce(b.Entity_Aliases, []) | toLower(toString(x))] AS b_aliases
WHERE a_name CONTAINS term
   OR b_name CONTAINS term
   OR ANY(alias IN a_aliases WHERE alias CONTAINS term)
   OR ANY(alias IN b_aliases WHERE alias CONTAINS term)
WITH a, b, r, collect(DISTINCT term) AS matched_terms
WITH
  coalesce(a.Entity_Name, a.name, a.label, a.Label, a.title, toString(elementId(a))) AS src,
  coalesce(r.Relation_Name, type(r)) AS rel,
  coalesce(b.Entity_Name, b.name, b.label, b.Label, b.title, toString(elementId(b))) AS dst,
  matched_terms
RETURN src, rel, dst, matched_terms
ORDER BY size(matched_terms) DESC, src ASC, rel ASC, dst ASC
LIMIT $limit
"""

CYPHER_RELATION_COUNTS = """
UNWIND $terms AS term
MATCH (a)-[r]->(b)
WITH term, a, b, r,
     toLower(coalesce(a.Entity_Name, "")) AS a_name,
     toLower(coalesce(b.Entity_Name, "")) AS b_name,
     [x IN coalesce(a.Entity_Aliases, []) | toLower(toString(x))] AS a_aliases,
     [x IN coalesce(b.Entity_Aliases, []) | toLower(toString(x))] AS b_aliases
WHERE a_name CONTAINS term
   OR b_name CONTAINS term
   OR ANY(alias IN a_aliases WHERE alias CONTAINS term)
   OR ANY(alias IN b_aliases WHERE alias CONTAINS term)
RETURN coalesce(r.Relation_Name, type(r)) AS rel, count(*) AS cnt
ORDER BY cnt DESC
LIMIT 20
"""

EDGE_PATTERN = re.compile(r"^(?P<src>.+?) -\[(?P<rel>.+?)\]-> (?P<dst>.+)$")


def extract_terms(query: str) -> list[str]:
    words = [w.strip().lower() for w in re.split(r"\s+", query.strip().rstrip("?!.,")) if w.strip()]
    words = [w for w in words if w not in STOP_WORDS and len(w) > 1]

    terms: list[str] = []
    n = len(words)
    for size in range(min(4, n), 0, -1):
        for i in range(n - size + 1):
            term = " ".join(words[i : i + size])
            if term not in terms:
                terms.append(term)
    return terms


def normalize_edge(src: str, rel: str, dst: str) -> tuple[str, str, str]:
    return (src.strip().lower(), rel.strip().lower(), dst.strip().lower())


def parse_terminal_edges(terminal_path: Path) -> list[tuple[str, str, str]]:
    edges: list[tuple[str, str, str]] = []
    for raw in terminal_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if "|" in line and re.match(r"^L\d+:", line):
            line = line.split(":", 1)[1].strip()
        m = EDGE_PATTERN.match(line)
        if m:
            edges.append((m.group("src"), m.group("rel"), m.group("dst")))
    return edges


def latest_terminal_file(terminals_dir: Path) -> Path:
    files = list(terminals_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No terminal files found in {terminals_dir}")
    return max(files, key=lambda p: p.stat().st_mtime)


def fetch_relevant_edges(uri: str, user: str, password: str, terms: list[str], limit: int) -> tuple[list[tuple[str, str, str]], list[tuple[str, int]]]:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            edge_rows = session.run(CYPHER_RELEVANT_EDGES, terms=terms, limit=limit)
            edges = [(r["src"], r["rel"], r["dst"]) for r in edge_rows]

            rel_rows = session.run(CYPHER_RELATION_COUNTS, terms=terms)
            relation_counts = [(r["rel"], int(r["cnt"])) for r in rel_rows]
    finally:
        driver.close()
    return edges, relation_counts


def format_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    rows_list = [list(map(str, row)) for row in rows]
    cols = len(headers)
    widths = [len(h) for h in headers]
    for row in rows_list:
        for i in range(cols):
            widths[i] = max(widths[i], len(row[i]) if i < len(row) else 0)

    def fmt_row(items: list[str]) -> str:
        return "| " + " | ".join(items[i].ljust(widths[i]) for i in range(cols)) + " |"

    sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    out = [fmt_row(headers), sep]
    for row in rows_list:
        out.append(fmt_row(row + [""] * (cols - len(row))))
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Neo4j-relevant edges vs latest terminal output.")
    parser.add_argument("--query", default="List all dog breeds", help="Natural-language query.")
    parser.add_argument("--limit", type=int, default=5000, help="Max Neo4j relevant edges to pull.")
    parser.add_argument("--terminal-file", default="", help="Specific terminal output file path.")
    parser.add_argument(
        "--terminals-dir",
        default=os.getenv(
            "CURSOR_TERMINALS_DIR",
            str(Path.home() / ".cursor" / "projects" / "Users-ashish-pavan-Projects-PES-Capstone" / "terminals"),
        ),
        help="Directory containing terminal .txt files (used if --terminal-file is omitted).",
    )
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "Capstone_Data"))
    args = parser.parse_args()

    terms = extract_terms(args.query)
    if not terms:
        raise ValueError("No searchable terms extracted from query. Try a more specific query.")

    terminal_path = Path(args.terminal_file) if args.terminal_file else latest_terminal_file(Path(args.terminals_dir))
    terminal_edges = parse_terminal_edges(terminal_path)

    neo4j_edges, relation_counts = fetch_relevant_edges(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
        terms=terms,
        limit=args.limit,
    )

    neo4j_norm = {normalize_edge(*e) for e in neo4j_edges}
    terminal_norm = {normalize_edge(*e) for e in terminal_edges}

    overlap = neo4j_norm & terminal_norm
    precision = (len(overlap) / len(terminal_norm)) if terminal_norm else 0.0
    recall = (len(overlap) / len(neo4j_norm)) if neo4j_norm else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print("\n=== Inputs ===")
    print(format_table(
        ["Field", "Value"],
        [
            ["Query", args.query],
            ["Extracted Terms", ", ".join(terms)],
            ["Terminal File", str(terminal_path)],
            ["Neo4j URI", args.neo4j_uri],
        ],
    ))

    print("\n=== Relevance Summary ===")
    print(format_table(
        ["Metric", "Value"],
        [
            ["Neo4j Relevant Edges", len(neo4j_norm)],
            ["Terminal Output Edges", len(terminal_norm)],
            ["Overlap Edges", len(overlap)],
            ["Precision (terminal vs Neo4j)", f"{precision:.4f}"],
            ["Recall (terminal vs Neo4j)", f"{recall:.4f}"],
            ["F1", f"{f1:.4f}"],
        ],
    ))

    print("\n=== Top Relation Types In Neo4j Relevant Set ===")
    print(format_table(["Relation", "Count"], relation_counts))

    only_terminal = sorted(terminal_norm - neo4j_norm)[:20]
    only_neo4j = sorted(neo4j_norm - terminal_norm)[:20]

    print("\n=== In Terminal Only (Sample) ===")
    print(format_table(["src", "rel", "dst"], only_terminal if only_terminal else [["-", "-", "-"]]))
    print("\n=== In Neo4j Relevant Set Only (Sample) ===")
    print(format_table(["src", "rel", "dst"], only_neo4j if only_neo4j else [["-", "-", "-"]]))


if __name__ == "__main__":
    main()
