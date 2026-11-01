# Enhanced-KG-LLM benchmark results and status

**Updated:** 2026-10-28 (Asia/Kolkata)  
**Purpose:** Record executed checks and results without conflating component metrics with end-to-end QA.

## Executive summary

The metrics harness and reproducibility scaffolding are implemented. The BRINK validation split now has evaluations for QID-free string routing, an offline lexical linker, the saved hybrid entity linker, linked-query partition routing with base and fine-tuned MiniLM, and a paired uncertainty analysis. These remain validation results; several hyperparameters were selected on this same split.

No aggregate traversal or generated-answer result is available for the live Enhanced-KG-LLM BRINK pipeline. The live Neo4j and Ollama services were unavailable during the latest full-pipeline attempt; the older one-question smoke record is diagnostic only. A separate GTSQA/WikiKG2 evidence-retrieval study is now available in `../new/partition_research` and is summarized in [PARTITION_RETRIEVAL_PAPER.md](PARTITION_RETRIEVAL_PAPER.md); it measures partition-guided traversal, not BRINK partition ranking. Existing README latency / “zero hallucination” figures are not independently validated here.

## Executed results

### String-query BRINK validation routing (no QIDs in the retrieved query)

Run: `brink_val_string_queries`; **3,464** of 3,465 validation questions were scored (one topic label was missing). For each question, the BRINK `q_entity` field was used only to find that entity's canonical `Entity_Name`; the literal QID was replaced before retrieval. Gold answers were used only to map answer entities to their existing partitions for scoring. All evaluated retrieval inputs are free of QIDs. This isolates query-to-partition routing when the topic is expressed as a string.

This is **label-resolved**, not an automatic entity-linking result: the benchmark metadata supplies the correct topic ID to construct the string query. A production system must identify the entity from the user's text itself. The result therefore tests retrieval given the right human-readable topic string, not the full RAG pipeline or linker quality.

| Query form / router | Any-answer H@5 | Hard-answer H@5 | MRR@100 |
|---|---:|---:|---:|
| Raw BRINK query containing QID, dense MiniLM | 0.1132 | 0.1028 | 0.0772 |
| Raw QID, BM25 | 0.0456 | 0.0375 | 0.0305 |
| Raw QID, 0.70 dense / 0.30 BM25 (validation-selected) | 0.1253 | 0.1163 | 0.0800 |
| String topic name, dense MiniLM | 0.4018 | 0.3736 | 0.3164 |
| String topic name, BM25 | 0.5508 | 0.5127 | 0.4374 |
| String topic name, 0.50 dense / 0.50 BM25 (current code) | 0.6383 | 0.5993 | 0.5174 |
| String topic name, 0.40 dense / 0.60 BM25 (best validation H@5) | **0.6400** | **0.6039** | 0.5228 |
| String topic name, RRF k=60 | 0.5857 | 0.5473 | 0.4546 |

Replacing QIDs with topic names raised H@5 by **+0.5225** for the current 0.50/0.50 hybrid (paired question bootstrap 95% CI [+0.5049, +0.5401]). The string-query weight sweep selected 0.40 dense / 0.60 BM25, with H@5 **+0.5147** over the raw-query sweep's best 0.70/0.30 hybrid. That cross-weight contrast is descriptive; the paired fixed-weight comparison is the 0.50/0.50 result above. Selection and reporting use the same validation split, so the selected score is descriptive and likely optimistic; do not treat it as a test result.

Median router scoring time for a raw/string query pair was **12.14 ms** and p95 **16.46 ms** on CPU, excluding loading and model/index startup. The benchmark reused the current `data/subg` index (3,625 partitions) and all-MiniLM-L6-v2 encoder. It does **not** measure answer generation, traversal, evidence recall, or total RAG latency. Because the BRINK source queries contain QIDs, this benchmark creates string-form inputs by canonical-label replacement; it is a controlled proxy for human-readable questions, not independently authored natural-language evaluation data.

Artifacts: [aggregate string-query report](results/raw/brink_val_string_queries.json) · [per-question rankings and transformed queries](results/raw/brink_val_string_queries.jsonl) · [benchmark script](scripts/benchmark_brink_string_queries.py).

### Entity-linking component evaluation

Run: `brink_val_entity_linking`; **3,464** validation examples were evaluated. The linker received QID-free questions formed by substituting each topic with its canonical name. Gold topic IDs were used only to score candidates. The baseline reuses `EntitySeedResolver.extract_mentions`, then does exact normalized lookup against all canonical names and aliases in the local entity table. It does not use answer information, partition membership, or Neo4j.

| Linker variant | Candidate generation recall | Link Hit@1 | Link Hit@3 | Link Hit@5 | MRR |
|---|---:|---:|---:|---:|---:|
| Earliest extracted mention first | 0.9053 | 0.6801 | 0.7708 | 0.7979 | 0.7343 |
| Latest extracted mention first | 0.9053 | **0.7477** | **0.8571** | **0.8741** | **0.8026** |

Mention extraction itself surfaced the gold entity string in **88.42%** of examples. When a candidate existed, preferring later query mentions improved Hit@1 by **+0.0676** over earliest-first ranking (paired question bootstrap 95% CI [+0.0537, +0.0814]). That ordering is a validation-tuned heuristic: BRINK often places its topic near the end of the sentence, so this gain may not transfer to arbitrary user questions. Candidate generation still missed 9.47% of gold topics, while ambiguous lexical matches lower precision at larger K (latest-first P@3: **0.2912**, recall@3: **0.8571**). A practical system should retain confidence thresholds/NIL handling rather than always choosing a candidate.

**Implementation limitation:** the actual `EntitySeedResolver.resolve()` requires a reachable Neo4j database and its full-text alias index. Neo4j at `127.0.0.1:7687` was unavailable during this run, so these scores are for an offline lexical baseline using the resolver's mention-extraction rules, not the production resolver. This isolates linker behavior and identifies a useful ranking change, but the production resolver should be benchmarked when its database is running.

Artifacts: [aggregate linker report](results/raw/brink_val_entity_linking.json) · [per-question candidate rankings](results/raw/brink_val_entity_linking.jsonl) · [linker benchmark script](scripts/benchmark_brink_entity_linking.py).

### Hybrid entity linker and linked partition routing (validation only)

A separate offline run benchmarked the actual hybrid linker implementation from `partition/subg/entity_linker.py`: exact lexical name/alias candidates plus dense top-50 search over the existing entity embeddings. Inputs are QID-free BRINK validation strings constructed by replacing the benchmark topic QID with its canonical name; the topic ID is used only as the scoring label. No answers or partition membership are passed to the linker. The gold topic label was available for 3,464/3,465 rows.

At the validation-selected lexical weight 0.70, linker Hit@1/3/5/10 was **0.9284 / 0.9902 / 0.9954 / 0.9968**, with MRR **0.9583**. The weight-0.50 setting achieved higher Hit@1 (0.9385) but lower Hit@5 (0.9838), illustrating that the operating point depends on the metric. On the RTX 4500 Ada, query-time median was **21.74 ms**, p95 **22.42 ms**, p99 **23.09 ms**; the separate startup was 8.66 s. Latency includes per-query encoding and candidate search but excludes model/index loading. These are link ranking metrics, not answer accuracy. There is no NIL/unknown-topic test in this BRINK subset.

The linked routing experiment used the linker’s top-1 entity name/alias expansion, the same 3,625-partition index, and the original `w_cosine` centroids. SHA-256 of the current centroids matches the partition v2 centroids, so the saved fine-tuned MiniLM encoder is evaluated in its trained embedding space. The fine-tuned checkpoint was trained on BRINK training data; the reported values are from validation and the 0.60 dense weight was selected using validation.

| Query encoder and text | Dense weight | Any-answer H@5 | Hard-answer H@5 | MRR@100 |
|---|---:|---:|---:|---:|
| Base MiniLM, canonical topic name only | 0.40 | 0.6400 | 0.6039 | 0.5228 |
| Base MiniLM, linker expansion | 0.30 | 0.7255 | 0.6905 | — |
| Fine-tuned MiniLM, canonical topic name only | 0.50 | 0.8259 | 0.7806 | — |
| Fine-tuned MiniLM, linker expansion | 0.60 | **0.8785** | **0.8401** | **0.7742** |

The fine-tuned linked-query value is **answer-containing partition Hits@5**: at least one partition containing an annotated answer entity appears in the first five. It is not answer correctness, evidence recall, or complete-path coverage. The best configuration is validation-selected, so the value is not an unbiased test estimate.

Paired bootstrap results (10,000 resamples; same 3,464 questions; rule-cluster resampling over 173 BRINK rules) quantify the major changes. Intervals are descriptive because configuration selection occurred on validation.

| Comparison (B−A) | Metric | Difference | Question bootstrap 95% CI | Rule-cluster 95% CI |
|---|---|---:|---:|---:|
| Base, linker expansion vs canonical string, fixed dense 0.50 | Any H@5 | +0.0782 | [+0.0670, +0.0898] | [+0.0604, +0.0958] |
| Fine-tuned linked vs base linked, fixed dense 0.60 | Any H@5 | +0.1758 | [+0.1619, +0.1894] | [+0.1322, +0.2237] |
| Fine-tuned linked vs base linked, fixed dense 0.60 | Hard H@5 | +0.1706 | [+0.1568, +0.1839] | [+0.1300, +0.2168] |
| Fine-tuned linked vs fine-tuned string, fixed dense 0.60 | Any H@5 | +0.0531 | [+0.0436, +0.0629] | [+0.0388, +0.0681] |
| Fine-tuned linked vs fine-tuned string, fixed dense 0.60 | Hard H@5 | +0.0586 | [+0.0488, +0.0684] | [+0.0442, +0.0736] |

The rule-cluster intervals are wider because questions from a shared BRINK rule are correlated. Report them alongside question-level intervals in the paper. The gains support the useful roles of name/alias expansion and encoder alignment on this benchmark, but do not establish generalization beyond BRINK.

Artifacts: [hybrid linker aggregate](results/raw/brink_val_hybrid_entity_linker.json) · [hybrid linker per-question candidates](results/raw/brink_val_hybrid_entity_linker.jsonl) · [linked routing, base encoder](results/raw/brink_val_linked_routing.json) · [linked routing, fine-tuned encoder](results/raw/brink_val_linked_routing_ft_minilm.json) · [paired intervals](results/reports/paper_paired_intervals.json). Scripts: [hybrid linker benchmark](scripts/benchmark_brink_hybrid_entity_linker.py) · [linked routing benchmark](scripts/benchmark_brink_linked_routing.py).

### BRINK answer-partition routing on a fresh rule-grouped internal holdout

The official BRINK test split has prior reuse, so a fresh internal split was formed from official BRINK training data by grouping `rule_index`: 141 train groups (19,404 questions), 28 development groups (4,158), and 27 held-out test groups (4,157). A new MiniLM encoder was trained only on train groups with multi-positive softmax cross-entropy over frozen `w_cosine` centroids. The previous fine-tuned checkpoint was not reused because it had trained on the full official train split. Fusion weights were selected on dev (0.45 dense for base MiniLM; 0.55 for the fresh fine-tuned encoder), then frozen before one test run.

Questions replace `q_entity` QIDs with canonical topic names. This evaluates routing conditional on the correct topic label and does not evaluate a live entity linker. All 4,157 test questions had any-answer and hard-answer partitions mapped across a 3,625-partition index.

| Test router | Any-answer H@5 | Hard-answer H@5 | H@10 | H@50 | MRR@100 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.5376 | 0.4980 | 0.6038 | 0.6921 | 0.4335 |
| Base dense RRF k=60 | 0.5874 | 0.5425 | 0.6497 | 0.7556 | 0.4649 |
| Fine-tuned dense RRF k=60 | 0.6863 | 0.6392 | 0.7924 | 0.9031 | 0.5622 |
| Base MiniLM hybrid, dense weight 0.45 | 0.6288 | 0.5850 | 0.6832 | 0.7693 | 0.5198 |
| **Fresh fine-tuned MiniLM hybrid, dense weight 0.55** | **0.7859** | **0.7436** | **0.8282** | **0.9071** | **0.6874** |

The fine-tuned hybrid improves H@5 over base hybrid by **15.71 points**. The paired 95% CI is [+14.51, +16.96] by question and [+4.36, +29.57] by rule-cluster bootstrap (27 clusters). These are answer-partition retrieval scores, not answer accuracy or evidence recall. The comparison and intervals are descriptive; only the fusion weights and test-use restriction were recorded in the frozen configuration, not a multiplicity-adjusted hypothesis family.

Details, dense-only baselines, candidate volume, full confidence intervals, and figure are in [PARTITION_RETRIEVAL_PAPER.md](PARTITION_RETRIEVAL_PAPER.md). The frozen split, model, configuration, and per-question test ranking artifacts are saved under `metrics/configs/` and `metrics/results/`.

### Metric implementation tests

Command: `.venv/bin/python -m unittest metrics.tests.test_score_metrics metrics.tests.test_traversal_counters`  
Result: **12 tests passed**. Coverage includes multi-answer normalized EM/F1, ranking and MRR, evidence recall, alternative complete paths, citation scores, latency percentiles, paired bootstrap, BRINK-style answer-set metrics, and aggregation.

### Existing data-quality audit reports (not rerun by this harness)

The repository already contains cleaning and validation scripts under `analysis/` and reports under `analysis/01_structural_pipeline/reports/`, `analysis/02_semantic_pipeline/reports/`, and `analysis/04_validation_and_comparison/reports/`. The comparison report records 4,813,491 entities preserved; triples reduced from 20,614,279 to 20,598,520 (99.9236% preserved); 758 self-loops removed; and 14,179 dangling references reduced to zero. These are previously saved audit results, not new runs in `metrics/`.

**Do not cite the description-cleaning counts until reconciled.** `original_vs_final_report.txt` reports 62,830 descriptions added and final coverage 95.31%; `final_entity_description_report.txt` reports 63,524 additions and only 65,827 empty descriptions. Those totals do not describe the same final state. The paper should identify the exact entity file/version and rerun one final audit against it.

### Partition index integrity and size

Command: `metrics/scripts/audit_partition_index.py --index-dir data/subg`  
Status: **PASS**.

| Check | Result |
|---|---:|
| Partitions | 3,625 |
| Centroid dimensions | 384 |
| Assigned entities | 4,813,491 |
| Partition size min / median / p95 / max | 150 / 1,049 / 3,653 / 8,000 |
| Centroid rows match partition IDs | Yes |
| BM25 document rows match partition IDs | Yes |
| Partition IDs unique | Yes |
| Entity map complete; centroids finite | Yes / Yes |
| Overflow recorded in index metadata | 0 |

This validates the saved index layout, not the semantic quality of its clusters or retrieval performance. The input partition metadata identifies it as `w_cosine`; artifacts are stored under ignored local `data/` and are not included in version control.

### BRINK-Wikidata5m partition routing

Run: `brink_val_routing_20261026T191944Z`; **3,465/3,465** validation questions had mapped any-answer and hard-answer partitions. The query was the raw BRINK question, including QIDs. No entity linker, QID-to-name substitution, or fine-tuned query encoder was used. The index was the local 3,625-community `data/subg` index. The development sweep selected dense weight 0.70 (BM25 weight 0.30) by any-answer H@5.

| Router | Any-answer H@1 | Any-answer H@5 | Hard-answer H@5 | MRR@100 |
|---|---:|---:|---:|---:|
| Dense only | 0.0447 | 0.1131 | 0.1027 | 0.0772 |
| BM25 only | 0.0124 | 0.0456 | 0.0375 | 0.0305 |
| Current code weight 0.50 dense / 0.50 BM25 | 0.0297 | 0.1157 | 0.1062 | 0.0713 |
| Validation-selected 0.70 dense / 0.30 BM25 | 0.0349 | **0.1253** | **0.1163** | 0.0799 |
| RRF, k=60 | **0.0569** | 0.1195 | 0.1007 | **0.0881** |

The router scoring latency was median **14.56 ms**, p95 **19.68 ms**, and p99 **22.07 ms** per question on the recorded local CUDA run. This excludes model/index loading, graph traversal, context construction, LLM generation, and service startup. It is not full-pipeline latency.

Paired question bootstrap for validation H@5 (10,000 resamples, seed 42) gave selected hybrid minus dense **+0.0121** (95% CI +0.0063 to +0.0182), minus BM25 **+0.0797** (+0.0684 to +0.0909), and minus RRF **+0.0058** (−0.0012 to +0.0127). Rule-cluster bootstrap over 173 `rule_index` groups widened the intervals: hybrid minus dense [+0.0009, +0.0257], minus BM25 [+0.0350, +0.1271], and minus RRF [−0.0212, +0.0331]. The hybrid weight was selected on this same validation set, so both interval families are descriptive and selection-optimistic; they are not confirmatory significance tests. RRF remains statistically unresolved against the selected hybrid in the rule-cluster comparison.

These are **answer-containing partition retrieval** results. They do not measure whether evidence facts or a complete reasoning path were retrieved, and they are not answer accuracy. They also are not comparable to the earlier ~0.90 answer-partition results, which used a linked-query/fine-tuned or MPNet-adapter protocol. The low raw-QID score shows that query preprocessing/encoder alignment is material and should be benchmarked as its own controlled factor.

Candidate-size cost at top 5 was substantial: mean entities returned were 8,626 for the implemented 0.50 hybrid, 9,094 for validation-selected 0.70, and 8,703 for RRF (p95: 15,492, 16,084, and 15,088 respectively). Each query also scored approximately 7,250 modality-partition pairs (3,625 dense and 3,625 BM25). The final JSON records SHA-256 hashes for both BRINK TSVs, all index files, and key source files, plus the model, Python, OS, CUDA device, and code commit.

Artifacts:

- [Aggregate validation report](results/raw/brink_val_routing_20261026T191944Z.json)
- [Per-question rankings](results/raw/brink_val_routing_20261026T191944Z.jsonl)
- [Index audit JSON](results/reports/partition_index_audit.json)

### Full-pipeline smoke check (one validation question)

A live one-question run succeeded through Neo4j and Ollama with `llama3.2:1b`, offline fallback disabled, and no skipped/mock answer. Model/index startup plus warm-up took **9.25 s**. The warmed question took **401.38 ms** end-to-end: routing 23.61 ms, seed resolution 92.22 ms, traversal 64.39 ms, and generation 220.98 ms. It fetched 13 adjacency entries, scored 11 candidate edges (33 feature-scoring operations), and formatted 4 evidence triples in 302 context tokens. Peak process RSS was about **4.09 GB**. For this single example, the resolved seeds missed the gold topic entity and none of the one annotated evidence triples was retrieved. These are diagnostics for one question only, not an aggregate quality or latency estimate.

Artifacts: [per-question pipeline smoke record](results/raw/pipeline_val_final_v2.jsonl) · [run manifest](results/raw/pipeline_val_final_v2.jsonl.manifest.json) · [one-example metric aggregation](results/reports/pipeline_val_final_v2_metrics.json).

## Benchmark scope implemented

`metrics/score_metrics.py` supplies reusable scoring functions for answer EM/F1 over multiple references, answer-set Hits/precision/recall/F1, hard-answer hit rate, partition Hits@K/MRR, evidence recall, complete annotated-path coverage, citation precision/recall/F1, paired bootstrap intervals, and median/p95/p99 latency summaries.

Runnable tools currently include:

- `scripts/audit_partition_index.py` — artifact integrity, partition sizes and storage footprint.
- `scripts/benchmark_brink_routing.py` — dense, BM25, dense/BM25 weight sweep, and RRF on BRINK question TSVs; emits per-question rankings and aggregate metrics. Test split is guarded and labeled exploratory if explicitly enabled.
- `scripts/score_traversal_runs.py` — scores per-question traversal outputs at shared budgets, including evidence recall, complete-path coverage, latency, adjacency entries, scoring operations, and paired intervals.
- `scripts/benchmark_pipeline.py` — captures full pipeline outputs, evidence, context size, stage timing, answer output, and failure/skip flags. It requires live Neo4j and Ollama; mock fallback is disabled.
- `scripts/prepare_brink_gold.py` and `scripts/score_brink_generations.py` — map BRINK answer IDs to labels and compute official-style normalized answer-set scores.
- `scripts/score_predictions.py` — generic aggregate scoring for saved JSONL outputs. `metrics/data_utils.py` parses BRINK grounding annotations into premise paths while excluding the held-out conclusion.

## What remains before paper claims

1. Extend the linker evaluation with genuine free-form questions and NIL/ambiguous-topic examples; this BRINK validation proxy substitutes the canonical topic name and cannot test whether a mention was correctly detected in independently authored text.
2. Implement and compare BFS, relation-guided, semantic-guided, and path-history traversal under identical adjacency-entry budgets. The traversal scorer is ready for these outputs, but this repository currently contains only the path-scoring traverser, not these baselines.
3. Run answer generation against the same evidence sets and fixed Ollama model/prompt. Score answer quality using labels prepared from the matching graph metadata; report skipped/direct-evidence/mock status separately.
4. Run end-to-end validation, tune only on validation, freeze all settings, and use a genuinely untouched test set. The existing BRINK test has been used during earlier project iterations, so a test run should be called exploratory unless an independent test benchmark is chosen.
5. Repeat stochastic training/partition construction with multiple seeds where practical. Question- and rule-cluster bootstrap do not measure training or partition seed variance.

## Citation and benchmark suitability

BRINK is the best primary benchmark match for the paper’s incomplete-knowledge claim: it provides complete/incomplete graphs, answer labels, and reasoning questions designed so direct supporting facts are removed. Cite the benchmark paper and identify the exact release/snapshot. [BRINK paper](https://aclanthology.org/2026.eacl-long.114/) · [official repository and dataset links](https://github.com/boschresearch/brink).

GTSQA is a suitable **separate** evidence-retrieval generalization benchmark because it provides ground-truth evidence subgraphs and unseen relation/graph-structure test labels. It is based on a different Wikidata/WikiKG2 graph substrate, and its scores must not be merged with BRINK. Cite [the GTSQA paper](https://openreview.net/forum?id=N3WOfE4bvw) and [dataset card](https://huggingface.co/datasets/Graphcore/GTSQA) if used. Cite [KEPLER](https://aclanthology.org/2021.tacl-1.11/) when describing Wikidata5M.
