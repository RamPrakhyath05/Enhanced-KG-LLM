# Development Log

## Project: Enhanced-KG-LLM

This document records major development milestones, experiments, benchmarks, and decisions.

---

## 2025-12-19 — Initial Project Setup

### Objective

Initialize the project repository.

### Changes

- Created the initial project repository.
- Added the initial `README.md`.

---

## 2026-02-17 — Project Documentation Update

### Changes

- Updated `README.md`.

---

## 2026-07-27 — Neo4j Project Foundation

### Objective

Establish the core project structure and Neo4j connection.

### Changes

- Set up the project skeleton.
- Added Neo4j connection support.

---

## 2026-08-01 — Graph Query Layer

### Objective

Add the initial graph query functionality.

### Changes

- Added entity lookup.
- Added neighbor traversal.
- Established the initial graph query layer.

---

## 2026-08-26 — Retrieval Foundation

### Objective

Establish the initial retrieval layer for the Hybrid Graph RAG system.

The initial implementation operates at the entity level. Graph partitioning and query-aware traversal will be introduced later.

### Dataset

The Wikidata5M graph is represented through:

- `entities.csv`
- `relations.csv`
- `triples.csv`

Entity records contain IDs, names, aliases, and descriptions. Relations contain IDs, names, and aliases. Triples represent:

```text
Entity → Relation → Entity
```

### Entity Extraction

- Added streaming entity retrieval from Neo4j.
- Converted entities into searchable documents using name, aliases, and description.
- Extracted an initial corpus of **10,000 entities** for retrieval experiments.

---

## 2026-08-26 — BM25 Sparse Retrieval

### Objective

Implement lexical retrieval over the entity corpus.

### Changes

- Added BM25 retriever.
- Added BM25 index generation and testing.
- Built an index for **10,000 entities**.

Index:

```text
indexes/bm25/entities_10k.pkl
```

### Test Queries

- `American rock band`
- `investment bank Iceland`
- `Italian Renaissance painter`
- `Hong Kong actress fashion designer`
- `airport Sochi Russia`

BM25 successfully retrieved relevant entities for several queries, including:

- `investment bank Iceland` → Straumur-Burðarás
- `Hong Kong actress fashion designer` → Flora Zeta
- `airport Sochi Russia` → URSS

---

## 2026-08-26 — Dense Semantic Retrieval

### Objective

Add semantic retrieval to complement BM25.

### Changes

- Added dense retriever.
- Used `all-MiniLM-L6-v2`.
- Embedding dimension: **384**.
- Built a dense index for **10,000 entities**.
- Added dense retrieval testing.

Index:

```text
indexes/dense/entities_10k.pkl
```

Dense retrieval produced results that overlapped with BM25 for some queries and differed for others, confirming that the two methods provide complementary retrieval signals.

---

## 2026-08-26 — Hybrid RRF Retrieval

### Objective

Combine BM25 and dense retrieval into a single hybrid retriever.

### Changes

- Added `HybridRetriever`.
- Added RRF-based rank fusion.
- Added hybrid retrieval testing.

RRF configuration:

```text
k = 60
top-k = 5
```

RRF combines rankings rather than raw scores:

```text
RRF(entity) = Σ 1 / (k + rank(entity))
```

When both retrieval methods rank the same entity highly, the entity receives a stronger fused score.

Example:

```text
investment bank Iceland
→ Straumur-Burðarás
→ RRF = 0.032787
```

---

## 2026-08-26 — Initial Hybrid Benchmark

### Objective

Measure the initial retrieval performance before optimizing the hybrid execution.

### Configuration

- Corpus: **10,000 entities**
- Model: `all-MiniLM-L6-v2`
- Embedding dimension: **384**
- RRF `k`: **60**
- Top-k: **5**
- Queries: **5**

### Results

| Metric | Average |
|---|---:|
| BM25 | 36.690 ms |
| Dense | 77.279 ms |
| Hybrid | 117.297 ms |

Model/index loading time:

**7.2936 s**

### Observation

The initial hybrid implementation executes BM25 and dense retrieval sequentially.

Therefore, the measured 117.297 ms represents a sequential hybrid baseline and is not the intended final architecture.

The intended architecture runs BM25 and dense retrieval concurrently and applies RRF after both complete.

---

## 2026-08-26 — Concurrent Hybrid Retrieval

### Objective

Optimize hybrid retrieval by running BM25 and dense retrieval concurrently instead of sequentially.

### Changes

- Updated `HybridRetriever` to use `ThreadPoolExecutor`.
- BM25 and dense retrieval now execute concurrently.
- RRF fusion is performed after both retrieval processes complete.
- Preserved the existing RRF configuration:
  - `k = 60`
  - `top-k = 5`

### Benchmark

Using **10,000 entities** and the same 5 test queries:

| Metric | Average |
|---|---:|
| BM25 | 23.592 ms |
| Dense | 83.094 ms |
| Hybrid | 107.800 ms |

Model/index loading time:

**6.2951 s**

### Observation

Concurrent execution reduced hybrid latency compared with the previous sequential baseline:

```text
Sequential hybrid: 117.297 ms
Concurrent hybrid: 107.800 ms
```

---

## 2026-08-26 — Concurrent Hybrid Retrieval with Weighted Score Fusion (Instead of RRF)

### Objective

Improve hybrid retrieval by executing BM25 and dense retrieval concurrently and replacing RRF with weighted score fusion.

### Changes

- Updated `HybridRetriever` to use `ThreadPoolExecutor`.
- BM25 and dense retrieval now execute concurrently.
- Replaced RRF rank fusion with normalized weighted score fusion.
- BM25 and dense retrieval use equal weights:
  - BM25: `0.5`
  - Dense: `0.5`
- Hybrid results are ranked using the combined normalized score.

### Benchmark

Using **10,000 entities** and the same 5 test queries:

| Metric | Average |
|---|---:|
| BM25 | 22.464 ms |
| Dense | 84.900 ms |
| Hybrid | 106.177 ms |

Model/index loading time:

**6.4953 s**

### Observation

Weighted score fusion produced a similar hybrid latency to the previous concurrent RRF implementation while providing a simpler score-based combination of BM25 and dense retrieval.
The retrieval layer currently serves as an entity-level baseline. The next major stage is knowledge graph cleaning and subsequent graph partitioning.

---

## Current Status

Completed:

- Neo4j integration
- Entity extraction
- Entity document construction
- BM25 retrieval
- Dense retrieval
- Concurrent hybrid retrieval
- Weighted score fusion
- Initial retrieval benchmarks

Next:

- Knowledge graph cleaning
- Graph partitioning
- Partition-level retrieval
- Query-aware traversal
- Context construction
- LLM generation

---
