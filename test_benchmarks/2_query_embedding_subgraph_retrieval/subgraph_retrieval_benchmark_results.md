# Academic Benchmark Report: Module 2 (Query Embedding, Tokenization & Subgraph Retrieval)

> [!NOTE]
> **Evaluation Protocol**: Evaluates query-to-subgraph routing over **331 multi-hop questions** from the **BRINK Benchmark** ([Ding et al., 2025](https://arxiv.org/abs/2508.08344)) across **3,625 precomputed semantic subgraphs** (4.81M nodes).
> Compares:
> 1. **Dense Bi-Encoder Centroid Routing** (`all-MiniLM-L6-v2`)
> 2. **Sparse Lexical BM25 Routing** (Okapi BM25 on Entity Vocabularies)
> 3. **Hybrid Normalized Convex Fusion** (50% Dense + 50% BM25)
> 4. **Reciprocal Rank Fusion (RRF, $k=60$)**
> 5. **Random Partition Baseline**

---

## 1. Executive Summary & Key Results

| Retrieval Paradigm | Topic Hits@1 | Topic Hits@3 | Topic Hits@5 | Answer Hits@3 | Answer Recall@5 | MRR (Topic) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Baseline** | 0.30% | 0.30% | 0.30% | 0.30% | 0.15% | 0.0035 | 0.05 ms |
| **Sparse BM25** | 38.67% | 57.10% | 61.93% | 45.02% | 48.46% | 0.4987 | 8.63 ms |
| **Dense Centroids** | 24.47% | 37.16% | 42.60% | 34.44% | 36.01% | 0.3305 | 2.14 ms |
| **RRF (Rank Fusion)** | 39.58% | 57.70% | 63.75% | 46.83% | 48.77% | 0.5075 | 10.79 ms |
| **Hybrid (Dense + BM25)** | **45.62%** | **66.16%** | **72.51%** | **52.27%** | **54.74%** | **0.5761** | **10.79 ms** |

---

## 2. Experimental Methodology & Implementation Details

To ensure rigorous reproducibility and alignment with Information Retrieval (IR) and Knowledge Graph standards, this benchmark evaluates **five distinct routing paradigms**:

### Test 1: Dense Semantic Centroid Routing (Bi-Encoder)
* **Implemented in**: `evaluate_retrieval()` using `SentenceTransformer('all-MiniLM-L6-v2')`.
* **Academic Reference**: *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks* (Reimers & Gurevych, EMNLP 2019).
* **Objective**: Measure semantic conceptual alignment between natural language query intent and partition topic centroids.
* **Implementation Algorithm**:
  1. Natural language query string $q$ is embedded into dense representation vector $q_e = \text{Encoder}(q) \in \mathbb{R}^{384}$.
  2. Query vector is $L_2$-normalized: $\hat{q}_e = \frac{q_e}{\|q_e\|_2}$.
  3. Inner dot product is evaluated against the $3,625 \times 384$ pre-normalized partition centroid matrix $C$:
     $$S_{\text{dense}}(q, p_i) = \hat{q}_e \cdot \hat{c}_i^T$$
  4. Partitions are ranked descending: $\mathcal{R}_{\text{dense}} = \text{argsort}(-S_{\text{dense}})$.

---

### Test 2: Sparse Lexical BM25 Routing (Inverted Index)
* **Implemented in**: `evaluate_retrieval()` using `rank_bm25.BM25Okapi`.
* **Academic Reference**: *The Probabilistic Relevance Framework: BM25 and Beyond* (Robertson & Zaragoza, FnTIR 2009).
* **Objective**: Measure exact keyword and named entity lexical overlap between query tokens and partition entity vocabulary.
* **Implementation Algorithm**:
  1. Natural language query is tokenized via regular expression word extraction: $\mathcal{T}_q = \text{tokenize}(q)$.
  2. For each partition document $D_i$ (containing the concatenated entity surface labels of partition $i$), the Okapi BM25 score is computed:
     $$S_{\text{BM25}}(q, D_i) = \sum_{t \in \mathcal{T}_q} \text{IDF}(t) \cdot \frac{f(t, D_i) \cdot (k_1 + 1)}{f(t, D_i) + k_1 \cdot \left(1 - b + b \cdot \frac{|D_i|}{\text{avgdl}}\right)}$$
     where standard hyperparameter calibrations $k_1 = 1.5, b = 0.75$ are applied.
  3. Partitions are ranked descending: $\mathcal{R}_{\text{BM25}} = \text{argsort}(-S_{\text{BM25}})$.

---

### Test 3: Hybrid Score Fusion (Dense + Lexical Min-Max Blending)
* **Implemented in**: `evaluate_retrieval()`
* **Academic Reference**: *Hybrid Search Architecture for Vector Databases* (Karpukhin et al., EMNLP 2020).
* **Objective**: Evaluate whether combining dense semantic semantics with sparse exact lexical matching resolves out-of-vocabulary entities and semantic drift.
* **Implementation Algorithm**:
  1. Raw dense scores and lexical BM25 scores are normalized to the unit interval $[0, 1]$ via min-max scaling:
     $$S_{\text{dense}}^{\text{norm}} = \frac{S_{\text{dense}} - \min(S_{\text{dense}})}{\max(S_{\text{dense}}) - \min(S_{\text{dense}}) + \epsilon}, \quad S_{\text{BM25}}^{\text{norm}} = \frac{S_{\text{BM25}} - \min(S_{\text{BM25}})}{\max(S_{\text{BM25}}) - \min(S_{\text{BM25}}) + \epsilon}$$
  2. A linear convex combination with equal weighting ($\alpha = 0.50$) is computed:
     $$S_{\text{hybrid}}(q, p_i) = \alpha \cdot S_{\text{dense}}^{\text{norm}}(q, p_i) + (1 - \alpha) \cdot S_{\text{BM25}}^{\text{norm}}(q, p_i)$$
  3. Partitions are sorted descending by combined score.

---

### Test 4: Reciprocal Rank Fusion (RRF)
* **Implemented in**: `evaluate_retrieval()`
* **Academic Reference**: *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods* (Cormack, Clarke & Buettcher, SIGIR 2009).
* **Objective**: Provide a parameter-free, scale-invariant rank-based fusion baseline.
* **Implementation Algorithm**:
  1. Let $r_{\text{dense}}(p_i)$ and $r_{\text{BM25}}(p_i)$ be the ordinal ranks (1-indexed) of partition $p_i$ in the dense and lexical lists.
  2. The fused score is calculated with constant smoothing factor $k = 60$:
     $$S_{\text{RRF}}(p_i) = \frac{1}{60 + r_{\text{dense}}(p_i)} + \frac{1}{60 + r_{\text{BM25}}(p_i)}$$
  3. Partitions are ranked descending by $S_{\text{RRF}}$.

---

### Test 5: Standard Academic IR Metrics Formulation
For each method $\mathcal{M}$ and evaluation rank cutoffs $k \in \{1, 3, 5, 10\}$:
1. **Hits@k (Success Rate)**:
   $$\text{Hits}@k = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}\left( \text{Gold Partition} \cap \text{Top-}k(q, \mathcal{M}) \neq \emptyset \right)$$
2. **Mean Reciprocal Rank (MRR@100)**:
   $$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\min \{ \text{rank}(p) : p \in \text{Gold Partitions} \}}$$
3. **Answer Entity Recall@k**:
   $$\text{Recall}@k = \frac{1}{|Q|} \sum_{q \in Q} \frac{|\mathcal{E}_{\text{gold}}(q) \cap \bigcup_{p \in \text{Top-}k} \text{Entities}(p)|}{|\mathcal{E}_{\text{gold}}(q)|}$$
4. **Candidate Reduction Ratio ($CRR@k$)**:
   $$CRR@k = \left( 1.0 - \frac{\text{Average Nodes Retrieved in Top-}k}{\text{Total Graph Nodes (4.81M)}} \right) \times 100\%$$


---

## 3. Detailed Retrieval Accuracy Breakdown

### A. Topic Entity Partition Retrieval (Anchor Localization)
Measuring whether the router successfully directs the agent to the partition where the seed query entity resides:

| Method | Hits@1 | Hits@3 | Hits@5 | Hits@10 | MRR@100 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Baseline** | 0.30% | 0.30% | 0.30% | 0.60% | 0.0035 |
| **Sparse BM25** | 38.67% | 57.10% | 61.93% | 70.69% | 0.4987 |
| **Dense Centroids** | 24.47% | 37.16% | 42.60% | 50.15% | 0.3305 |
| **RRF (Rank Fusion)** | 39.58% | 57.70% | 63.75% | 73.41% | 0.5075 |
| **Hybrid (Dense + BM25)** | **45.62%** | **66.16%** | **72.51%** | **79.76%** | **0.5761** |

---

### B. Answer Partition Retrieval & Multi-Hop Answer Recall
Measuring whether the target answers are captured within the retrieved subgraphs:

| Method | Answer Hits@1 | Answer Hits@3 | Answer Hits@5 | Answer Recall@3 | Answer Recall@5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Baseline** | 0.30% | 0.30% | 0.60% | 0.15% | 0.15% |
| **Sparse BM25** | 31.12% | 45.02% | 51.36% | 42.62% | 48.46% |
| **Dense Centroids** | 22.05% | 34.44% | 39.27% | 31.77% | 36.01% |
| **RRF (Rank Fusion)** | 33.53% | 46.83% | 51.36% | 44.59% | 48.77% |
| **Hybrid (Dense + BM25)** | **35.35%** | **52.27%** | **57.70%** | **49.76%** | **54.74%** |

---

## 4. Scalability & Graph Search Reduction Efficiency

A fundamental paper claim of partitioned KG-RAG is that it filters out **>99% of irrelevant graph entities** before traversal:

| Retrieval Cutoff | Average Nodes Returned | Percent of Total 4.81M KG | Candidate Reduction Ratio ($CRR$) |
| :--- | :---: | :---: | :---: |
| **Top-1 Subgraph** | **2,163 nodes** | **0.045%** | **99.96% PRUNED** |
| **Top-3 Subgraphs** | **6,184 nodes** | **0.128%** | **99.87% PRUNED** |
| **Top-5 Subgraphs** | **10,049 nodes** | **0.209%** | **99.79% PRUNED** |
| **Top-10 Subgraphs** | **19,089 nodes** | **0.397%** | **99.60% PRUNED** |

> [!TIP]
> **Paper Claim**: Retrieving Top-3 subgraphs isolates **6,184 nodes** from the entire 4.81M graph, achieving a **99.87% candidate reduction ratio**. This reduces Neo4j traversal search spaces by over 3 orders of magnitude while preserving **66.2% topic hit rate**.

---

## 5. System Latency & Operational Throughput

* **Dense Query Encoding (all-MiniLM-L6-v2)**: `0.58 ms / query`
* **Dense Centroid Vector Search (3,625 partitions)**: `1.55 ms / query`
* **Sparse BM25 Index Search**: `8.63 ms / query`
* **End-to-End Hybrid Routing Latency**: **`10.79 ms / query`**
* **System Retrieval Throughput**: **`92.7 queries / second (QPS)`**

---
*Report generated by `test_benchmarks/2_query_embedding_subgraph_retrieval/benchmark_subgraph_retrieval.py`.*
