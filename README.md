# Enhanced-KG-LLM (50% Implementation Milestone)

> **Milestone Objective**: Finding the Correct Knowledge Graph Partition Based on User Query Using Parallel Hybrid Retrieval.

## Overview
This repository contains the complete 50% implementation milestone for Review 1 (September 2, 2026). It covers:
1. **Neo4j Property Graph Database**: Full Wikidata5M ingestion (4.81M entities, 20.6M relationships).
2. **Leiden Graph Partitioning**: Continuous cosine edge weighting & CPM Leiden decomposition with post-processing (oversized splitting + semantic KNN merging $\rightarrow$ 3,625 balanced communities, `v6` partition).
3. **Parallel Hybrid Partition Retrieval**: Parallel Dense (Centroid Embeddings) + Sparse (BM25 lexical) search fused via **Reciprocal Rank Fusion (RRF $k=60$)** to route user queries directly to the target partition in $< 90\text{ ms}$.

---

## Quick Start

### 1. Requirements & Setup
```bash
pip install -r requirements.txt
```
Ensure your `.env` contains your Neo4j credentials:
```env
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

### 2. Verify Database (Prem)
```bash
python -m scripts.test_connection
```

### 3. Run Graph Partitioning (Rahul)
```bash
python -m scripts.run_partitioning --limit 10000
```

### 4. Test Parallel Hybrid Latency (Ram)
```bash
python -m scripts.test_hybrid
```

### 5. 50% Milestone Flagship Demo: Query-to-Partition Routing (Ashish & Team)
```bash
python -m scripts.test_partition_routing
```
Or test a custom query:
```bash
python -m scripts.test_partition_routing --query "Australian wildlife fauna and gecko lizards"
```

---

## Guide
For the complete presentation breakdown and team roles, see [`REVIEW_1_GUIDE.md`](./REVIEW_1_GUIDE.md).
