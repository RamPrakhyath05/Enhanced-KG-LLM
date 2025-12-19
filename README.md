# Enhanced KG-RAG for Efficient LLM Reasoning

> **Efficient Knowledge Graph Retrieval for LLMs via Graph Partitioning and Hybrid Semantic Search**

## Overview

Large Language Models (LLMs) perform significantly better when augmented with external knowledge using Retrieval-Augmented Generation (RAG). However, existing Knowledge Graph (KG) based RAG systems often suffer from high latency, heavy computation, and noisy retrieval, especially when Graph Neural Networks (GNNs) are applied globally over massive knowledge graphs.

This project proposes an efficient KG-RAG pipeline that:
- Partitions large knowledge graphs into semantically coherent subgraphs
- Uses semantic embeddings and ANN search to identify relevant subgraphs
- Applies localized GNN-based reasoning instead of global graph traversal
- Supplies structured and minimal context to the LLM

The result is a faster, more efficient, and more accurate KG-RAG system with improved multi-hop reasoning and reduced hallucination.

---

## Problem Statement

Applying GNN-based retrieval globally on large-scale knowledge graphs such as Wikidata is computationally expensive, slow at inference time, and prone to semantic noise due to unrestricted message passing. This project addresses these issues by reducing the search and reasoning space through graph partitioning and hybrid semantic retrieval.

---

## Key Idea

Instead of running GNNs over the entire knowledge graph:
1. Partition the KG using community detection algorithms such as Louvain or Leiden
2. Embed subgraphs semantically using textual metadata
3. Retrieve relevant subgraphs using ANN-based semantic search
4. Run GNN reasoning locally on the selected subgraphs
5. Feed structured evidence to the LLM for grounded generation

---

## Core Components

- **Knowledge Graph**
  - Property Graph model (not RDF)
  - Primary dataset: Wikidata-5M

- **Graph Partitioning**
  - Louvain and Leiden community detection
  - Semantically coherent subgraph generation

- **Semantic Retrieval**
  - Text-based embeddings from entity labels, descriptions, and aliases
  - ANN-based subgraph selection

- **Graph Neural Networks**
  - Pre-trained GNNs such as R-GCN variants
  - Localized message passing on selected subgraphs

- **LLM Integration**
  - RAG pipeline using graph-derived structured context
  - Reduced hallucination and improved multi-hop reasoning

---

## Scope

### In Scope
- Graph partitioning and subgraph-based retrieval
- Hybrid ANN and GNN pipeline
- Integration of existing pre-trained GNNs and LLMs
- Benchmarking against global-GNN baselines
- Performance evaluation including latency, accuracy, and resource usage

### Out of Scope
- Designing new GNN or LLM architectures
- Creating new large-scale knowledge graphs from scratch

---

## Dataset

### Wikidata-5M
- Approximately 5 million entities
- Multi-relational and cross-domain knowledge
- Rich textual metadata including labels, descriptions, and aliases
- Suitable for large-scale semantic embedding and ANN search
- Easily convertible to property graph format

Other datasets such as DBpedia, ConceptNet, and FB15k-237 were evaluated but found less suitable for large-scale semantic KG-RAG.

---

## Research Gap Addressed

- Existing KG-RAG systems rarely leverage community-level subgraphs
- GNNs are typically trained and applied globally
- Structural partitioning is seldom combined with semantic retrieval

This project bridges these gaps by combining graph structure, semantic embeddings, and localized reasoning.

---

## Capstone Timeline

| Phase | Semester | Description |
|------|---------|-------------|
| Phase-1 | 5th Semester | Background study, literature survey, dataset exploration (Completed) |
| Phase-2 | 6th Semester | High-level system design and partial implementation |
| Phase-3 | 7th Semester | Full implementation, experimentation, and paper draft |
| Phase-4 | 8th Semester | Journal publication (Scopus / Web of Science / UGC indexed) |

---

## References

Key references include:
- Angles, R. — The Property Graph Database Model
- Li et al. — Simple Is Effective: The Roles of Graphs and LLMs in KG-RAG
- Pan et al. — GNN-RAG: Graph Neural Retrieval for LLM Reasoning
- Guo et al. — LightRAG
- Bai, S. — Semantic Partitioning for Large-Scale Knowledge Graph Embeddings
- SPLIT-RAG, Tree-of-Traversals, and R-GCN related works

---

## Project Status

Phase-1 completed.  
Currently entering Phase-2: System Design and Partial Implementation.

---

## Note

The repository name `enhanced-kg-llm` is provisional and may be updated as the project evolves.

---

## Team
**Rahul Senthil Kumar** ([Rahul6700](https://github.com/Rahul6700))<br>
**Ram Prakhyath Annamareddy** ([RamPrakhyath05](https://github.com/RamPrakhyath05))<br>
**Renikuntla Ashish Pavan** ([ashishpavan1819](https://github.com/ashishpavan1819))<br>
**Suryavanshi Prem Pandurang**

Project Mentor / Guide:<br>
**Dr. Sandesh B J**<br>
**Chairperson, Department of Computer Science and Engineering**<br>
**PES University** 

---

