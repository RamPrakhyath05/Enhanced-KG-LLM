# Enhanced KG-RAG for Efficient LLM Reasoning

> **Efficient Knowledge Graph Retrieval for LLMs via Graph Partitioning and Hybrid Semantic Search**

## Overview

Large Language Models (LLMs) benefit significantly from Retrieval-Augmented Generation (RAG), but existing Knowledge Graph-based RAG systems often suffer from high retrieval latency and inefficient evidence extraction over large-scale graphs. Most approaches rely on either dense semantic retrieval or sparse keyword retrieval independently and often perform generic graph traversal after retrieval, leading to unnecessary computation and noisy context.

This project proposes an efficient KG-RAG pipeline that partitions a large knowledge graph into semantically coherent subgraphs, performs parallel hybrid retrieval using dense vector search and BM25 to identify the most relevant partitions, and then applies an efficient query-aware traversal strategy to retrieve only the evidence required for answering the query.

The retrieved evidence is provided as structured context to an LLM, improving response quality while reducing latency and computational overhead.

The result is a faster, more efficient, and more accurate KG-RAG system with improved multi-hop reasoning and reduced hallucination.

---

## Problem Statement

Applying GNN-based retrieval globally on large-scale knowledge graphs such as Wikidata is computationally expensive, slow at inference time, and prone to semantic noise due to unrestricted message passing.

This project addresses these issues by reducing the search and reasoning space through graph partitioning and hybrid semantic retrieval.

---

## Key Idea

Instead of running GNNs over the entire knowledge graph:

1. Partition the knowledge graph using **Leiden community detection**.
2. Build both dense and sparse indexes for each partition.
3. Execute dense semantic retrieval and BM25 retrieval in parallel.
4. Fuse the retrieval results to identify the most relevant partition(s).
5. Efficiently traverse only the selected partition(s) using query-aware traversal.
6. Retrieve the supporting triples required to answer the query.
7. Convert the retrieved triples into structured context for the LLM.

---

## Pipeline

```text
                     OFFLINE
────────────────────────────────────────────

Knowledge Graph
        │
        ▼
Leiden Community Detection
        │
        ▼
Partitioned Graph
        │
        ├──────────────┐
        ▼              ▼
   Dense Index     BM25 Index


                     ONLINE
────────────────────────────────────────────

User Query
        │
        ├──────────────┐
        ▼              ▼
Dense Retrieval   Sparse Retrieval
        │              │
        └──────┬───────┘
               ▼
          Rank Fusion
               ▼
      Selected Partition(s)
               ▼
      Query-Aware Traversal
               ▼
       Supporting Triples
               ▼
         Context Builder
               ▼
      Large Language Model
               ▼
             Answer
````

---

## Core Components

### Knowledge Graph

* Property Graph representation
* Wikidata5M dataset

### Graph Partitioning

* Leiden community detection
* Offline partition generation

### Hybrid Retrieval

* Dense retrieval using semantic embeddings
* Sparse retrieval using BM25
* Parallel execution
* Score/rank fusion for partition selection

### Graph Traversal

* Efficient traversal within selected partitions
* Query-guided evidence extraction
* Minimal context generation

### LLM Integration

* Structured graph context
* Grounded response generation
* Reduced hallucinations

---

## Scope

### In Scope

* Graph partitioning and subgraph-based retrieval
* Hybrid ANN and GNN pipeline
* Integration of existing pre-trained GNNs and LLMs
* Benchmarking against global-GNN baselines
* Performance evaluation including latency, accuracy, and resource usage

### Out of Scope

* Designing new GNN or LLM architectures
* Creating new large-scale knowledge graphs from scratch

---

## Dataset

### Wikidata5M

* Approximately 5 million entities
* Multi-relational and cross-domain knowledge
* Rich textual metadata including labels, descriptions, and aliases
* Suitable for large-scale semantic embedding and ANN search
* Easily convertible to a property graph format

Other datasets such as DBpedia, ConceptNet, and FB15k-237 were evaluated but found less suitable for large-scale semantic KG-RAG.

---

## Research Gap Addressed

Current KG-RAG systems generally rely on similarity-based retrieval to identify relevant graph partitions and often perform generic graph traversal within the selected subgraphs.

These approaches do not fully exploit the complementary strengths of dense semantic retrieval and sparse keyword retrieval, nor do they optimize traversal for efficient evidence extraction.

This project addresses these gaps by combining **hybrid partition retrieval** with **efficient graph traversal** to reduce latency while maintaining retrieval quality.

---

## Capstone Timeline

| Phase   | Semester     | Description                                                          |
| ------- | ------------ | -------------------------------------------------------------------- |
| Phase-1 | 5th Semester | Background study, literature survey, dataset exploration (Completed) |
| Phase-2 | 6th Semester | High-level system design and partial implementation                  |
| Phase-3 | 7th Semester | Full implementation, experimentation, and paper draft                |
| Phase-4 | 8th Semester | Journal publication (Scopus / Web of Science / UGC indexed)          |

---

## References

Key references include:

* Angles, R. — *The Property Graph Database Model*
* Li et al. — *Simple Is Effective: The Roles of Graphs and LLMs in KG-RAG*
* Pan et al. — *GNN-RAG: Graph Neural Retrieval for LLM Reasoning*
* Guo et al. — *LightRAG*
* Bai, S. — *Semantic Partitioning for Large-Scale Knowledge Graph Embeddings*
* *SPLIT-RAG*
* *Tree-of-Traversals*
* R-GCN-related works

---

## Project Status

**Phase-1 and Phase-2 completed.**

Currently entering **Phase-3: Full implementation, experimentation, and paper draft.**

---

## Note

The repository name `Enhanced-KG-LLM` is provisional and may be updated as the project evolves.

---

## Team

* **Rahul Senthil Kumar** — [Rahul6700](https://github.com/Rahul6700)
* **Ram Prakhyath Annamareddy** — [RamPrakhyath05](https://github.com/RamPrakhyath05)
* **Renikuntla Ashish Pavan** — [ashishpavan1819](https://github.com/ashishpavan1819)
* **Suryavanshi Prem Pandurang** — [prem-2411](https://github.com/prem-2411)

### Project Mentor / Guide

**Dr. Sandesh B J**
Chairperson, Department of Computer Science and Engineering
PES University
