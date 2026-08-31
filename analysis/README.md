# 📊 Knowledge Graph Data Engineering & Quality Analysis

> **Dataset**: Wikidata5M Knowledge Graph (Property Graph in Neo4j)  
> **Milestone**: 50% Milestone Review 1

This directory contains the complete data engineering, graph structural sanitization, semantic NLP cleaning, and cross-file validation pipeline for the Wikidata5M graph.

---

## 🏛️ Pipeline Architecture & Execution Sequence

```text
analysis/
│
├── 📁 01_structural_pipeline/                  # STEP 1: Graph topology sanitization
│   ├── scripts/
│   │   ├── clean_dataset.py                    # Removes dangling edges & missing entity/relation IDs
│   │   ├── final_structural_check.py           # Validates graph topology integrity (0 orphaned edges)
│   │   ├── analyze_self_loops.py               # Detects & audits self-referential edges (u -> u)
│   │   └── analyze_invalid_triples.py          # Scans for corrupted triple rows
│   └── reports/
│       ├── cleaning_report.txt                 # Initial structural removal of 15,001 invalid triples
│       ├── self_loops.txt                      # 758 self-loops audited & eliminated
│       └── final_triple_quality_report.txt     # 100% referential integrity proof (STATUS: PASS)
│
├── 📁 02_semantic_pipeline/                    # STEP 2: Semantic NLP cleaning & enrichment
│   ├── scripts/
│   │   ├── audit_aliases.py                    # Audits alias distributions & outlier noise
│   │   ├── audit_descriptions.py               # Audits description lengths, coverage & empty entities
│   │   ├── audit_relations.py                  # Audits 825 relations & hub usage frequencies
│   │   └── semantic_cleaner.py                 # Strips wiki noise & normalizes descriptions (LLM/Rules)
│   └── reports/
│       ├── alias_audit.txt                     # Raw alias distribution audit (outliers with 48k aliases)
│       ├── alias_cleaning_report.txt           # Cleaned result (7.17M duplicates removed, 31.1k archives stripped)
│       ├── description_audit.txt               # Analysis of empty descriptions & bot signatures
│       ├── final_entity_description_report.txt # Final coverage report (95.31% non-empty descriptions)
│       └── relation_quality_report.txt         # Full audit of 825 relations (P31 frequency = 3.83M triples)
│
├── 📁 03_combined_final_kg/                    # STEP 3: Master compilation
│   ├── build_final_kg.py                       # ★ Master Script: Merges structural triples + semantic entities
│   └── final_dataset_sanity_check.py           # Deep cross-file referential integrity verifier
│
└── 📁 04_validation_and_comparison/            # STEP 4: Before-vs-After comparative validation
    ├── scripts/
    │   └── final_dataset_comparison.py         # Script comparing Raw Original vs. Cleaned Final KG
    └── reports/
        ├── original_vs_final_report.txt        # ★ Master Output: The Before/After comparative metrics table
        ├── full_kg_sanity_report.txt           # Final sanity check output (STATUS: PASS)
        └── fullkg_baseline_results.txt         # Baseline retrieval performance benchmarks
```

---

## 📈 Master Before-vs-After Comparison Table (Raw vs. Cleaned KG)

| Metric / Dimension | Raw Original Wikidata5M | Cleaned Final KG (in Neo4j) | Changes Made & Impact |
| :--- | :--- | :--- | :--- |
| **Total Entity Nodes** | `4,813,491` | **`4,813,491`** | **0 lost, 0 duplicates**. 100% entity preservation. |
| **Total Graph Triples** | `20,614,279` | **`20,598,520`** | **`15,759` corrupted triples removed** (`99.92%` preservation). |
| **Self-Loops $(u \rightarrow u)$** | `758` loops | **`0` (Completely Eliminated)** | Removed cycles like `Q17 | P17 | Q17` (*Japan country Japan*). |
| **Missing / Dangling Edges** | `14,179` missing endpoints | **`0` (100% Referential Integrity)** | Every edge connects existing source and target entities. |
| **Cleaned Entity Aliases** | Raw Wikipedia dumps | **`4,800,419` aliases cleaned** | Stripped talk pages, archive keywords, wiki help-desks, and symbol noise. |
| **Alias Duplicates Removed** | High redundancy | **`7,175,819` duplicates removed** | Cleaned alias index for high-precision BM25 token matching. |
| **Canonical Name Aliases** | Repeated entity names | **`5,091,677` removed** | Prevented BM25 term frequency distortion. |
| **Junk Descriptions Stripped** | Bot & placeholder text | **`159,206` stripped** | Removed vandalized and bot-generated placeholder descriptions. |
| **Empty Descriptions Filled** | `129,351` empty | **`62,830` filled/enriched** | Final description coverage reached **`95.31%`** (`4,587,764` valid entities). |
| **Relations Available** | `825` | **`825`** | `816` active relations in triples, 0 duplicate IDs. |
| **Non-Topological Hub Filter** | Unfiltered | **`8` hub relations filtered** | Filtered `P31, P17, P27, P131, P19, P20, P21, P106` for 11.08M structural partitioning. |

---

## 🔬 Key Review 1 Takeaways:
1. **Zero Referential Integrity Errors**: Every node and edge loaded into Neo4j has valid foreign-key pointers.
2. **Deterministic & LLM Semantic Sanitization**: Noise was pruned without losing factual domain knowledge.
3. **Optimized for Partitioning & Hybrid Search**: Removed hub relation explosion so Leiden partitioning could discover clean, modular communities.
