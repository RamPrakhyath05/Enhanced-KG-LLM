## 🧠 Ashish Branch – GNN-Based Graph Traversal Module

This branch contains my individual contribution to the capstone project, focusing on **Graph Neural Network (GNN)–based traversal and reasoning over a property graph**.

---

### 🔹 Overview

The goal of this module is to enable **query-driven semantic traversal of a knowledge graph** using **Relational Graph Convolutional Networks (R-GCNs)**.
Given a user query, the system retrieves a relevant subgraph and applies a GNN to aggregate multi-hop relational information, producing structured knowledge that can be consumed by downstream components (e.g., an LLM).

---

### 🔹 Key Responsibilities Covered in This Branch

* Property graph–based graph representation
* Query-driven subgraph extraction using embeddings
* Relation-aware graph traversal using **R-GCN**
* Local execution (no cloud or external GNN APIs)
* Conversion of graph output into a readable string format

---

### 🔹 Pipeline Implemented

```
User Query
   ↓
Query Embedding (SentenceTransformers)
   ↓
Relevant Subgraph Extraction
   ↓
Relational GNN Traversal (R-GCN)
   ↓
Structured Text Output
```

---

### 🔹 Technologies Used

* **Python**
* **PyTorch**
* **PyTorch Geometric** (RGCNConv)
* **SentenceTransformers** (semantic query embeddings)
* **Property Graph model**
* Local execution with Apple Silicon (MPS) support

---

### 🔹 Repository Structure (Relevant to This Branch)

```
data/
  └── graph_data.py          # Static property graph data

models/
  └── rgcn.py                # R-GCN implementation

retrieval/
  └── query_to_subgraph.py   # Query → subgraph logic

utils/
  └── graph_utils.py         # Graph tensor conversion & output

main.py                      # End-to-end execution
```

---

### 🔹 Execution Instructions

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the pipeline:

   ```bash
   python main.py
   ```

3. Output:

   * A structured, human-readable string representing the query-relevant graph facts.

---

### 🔹 Scope & Design Decisions

* The GNN is **run locally**, not accessed via external APIs, to maintain transparency and experimental control.
* The model used is **Relational Graph Convolutional Network (R-GCN)**, chosen for its suitability to multi-relational property graphs.
* Attention mechanisms and advanced extensions are intentionally left as **future work**.

---

### 🔹 Future Extensions

* Attention-augmented R-GCN
* Neo4j-backed dynamic graph loading
* Query-conditioned message passing
* Integration with an LLM for natural language reasoning

---

### 🔹 Contribution Summary

> This branch represents the complete implementation of the **GNN-based reasoning and traversal component** of the capstone project and is intended to be merged into the main branch after review.

---
