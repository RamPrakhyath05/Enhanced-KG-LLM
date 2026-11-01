# Partition retrieval paper: current evidence and paper plan

Updated 2026-10-28. This is the consolidated evaluation note for a paper focused on retrieving relevant KG partitions/evidence. It keeps two related retrieval tasks separate because their units and metrics differ.

## Recommended paper focus

A defensible paper can center on **budgeted evidence retrieval using graph partitioning**, supported by the frozen GTSQA/WikiKG2 experiment, and present BRINK as a separate **question-to-answer-partition routing** experiment. Do not combine their scores or call both “Hits@5”: GTSQA tests whether search recovers annotated evidence triples under an edge-inspection cap; BRINK tests whether ranked partitions contain annotated answer entities.

A safe provisional title is **“Partition-Aware Retrieval for Knowledge-Graph Question Answering: Evidence Coverage and Answer-Partition Routing.”** Do not call the GTSQA refinement semantic-aware: it uses graph structure only. The project’s BRINK partition router uses dense entity/partition representations and BM25; that is a separate component.

## A. BRINK: answer-containing partition routing

### Fresh grouped holdout

The official BRINK test split was used in prior project experiments, so it is not the confirmatory set here. A new internal split was made from the official BRINK training questions, grouping by `rule_index` before training or retrieval evaluation. It has 141 train rules (19,404 question rows), 28 development rules (4,158 rows), and 27 held-out test rules (4,157 rows). The model is freshly initialized; the previous fine-tuned checkpoint, which saw the full official training split, was not reused.

The query text is made QID-free by replacing `q_entity` IDs with the canonical topic name in the entity table. This uses the benchmark’s correct topic label. It measures partition routing conditional on a resolved topic name, **not automatic entity linking on arbitrary user-written questions**. Gold answers are used only to identify relevant partitions for scoring.

The model trains for five epochs on 19,396 usable training questions with a multi-positive softmax loss over the frozen `w_cosine` partition centroids. Every mapped gold answer partition is a positive, avoiding false negatives from alternative annotated answers. On development rules, dense weights 0.45 for the base encoder and 0.55 for the fine-tuned encoder were selected by answer-containing partition Hits@5. The weights and index hashes were frozen before the held-out test evaluation.

### Held-out test results

All 4,157 held-out questions had a mapped any-answer and hard-answer partition. The index contains 3,625 partitions. “Any H@k” is the fraction of questions with at least one partition containing an annotated answer in the top k. It is not answer accuracy or evidence recall.

| Router | H@1 | H@3 | H@5 | Hard H@5 | H@10 | H@50 | MRR@100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base MiniLM dense only | 0.2300 | 0.3705 | 0.4313 | 0.3926 | 0.5090 | 0.6406 | 0.3210 |
| BM25 only | 0.3450 | 0.4857 | 0.5376 | 0.4980 | 0.6038 | 0.6921 | 0.4335 |
| RRF, k=60 (base dense ranking) | 0.3618 | 0.5328 | 0.5874 | 0.5425 | 0.6497 | 0.7556 | 0.4649 |
| RRF, k=60 (fine-tuned dense ranking) | 0.4489 | 0.6257 | 0.6863 | 0.6392 | 0.7924 | 0.9031 | 0.5622 |
| Base MiniLM + hybrid (dev-selected 0.45 dense) | 0.4289 | 0.5834 | 0.6288 | 0.5850 | 0.6832 | 0.7693 | 0.5198 |
| Fine-tuned MiniLM dense only | 0.4458 | 0.5836 | 0.6497 | 0.6173 | 0.7291 | 0.8306 | 0.5377 |
| **Fresh fine-tuned MiniLM + hybrid (dev-selected 0.55 dense)** | **0.6038** | **0.7460** | **0.7859** | **0.7436** | **0.8282** | **0.9071** | **0.6874** |

The dense-only rows, RRF, and selected hybrids are separate conditions. Each RRF row combines its named dense ranking with BM25.
The main paired comparison, fine-tuned hybrid vs base hybrid using their separately development-selected weights, improves H@5 by **15.71 percentage points** (0.7859 vs 0.6288). The paired 95% bootstrap CI is **[+14.51, +16.96] points** by question and **[+4.36, +29.57] points** by `rule_index` cluster. There are 4,157 questions but only 27 independent held-out rule groups; the wider clustered interval is the more conservative uncertainty estimate. The frozen config specifies dev-only weight selection and forbids test tuning, but does not declare a multiplicity-adjusted comparison family; present these intervals descriptively and avoid treating secondary comparisons as confirmatory significance tests.

Additional test comparisons: fine-tuned hybrid improves H@5 over fine-tuned RRF by **+9.96 points** (question CI [+8.95, +11.02]; rule-cluster CI [+2.63, +19.93]), over base RRF by **+19.85 points** (question CI [+18.55, +21.22]; rule-cluster CI [+9.10, +33.16]), and over BM25 by **+24.83 points** (question CI [+23.29, +26.34]; rule-cluster CI [+13.65, +38.19]). These additional comparisons are unadjusted; treat them as descriptive unless the paper predefines and adjusts multiple comparisons.

At top five, the fine-tuned router returns partitions containing a mean of **11,391 entities** (median 10,508; p95 18,436), compared with **10,313** for the base hybrid (median 9,524; p95 17,846). Thus the H@5 gain has a modestly larger candidate volume. Warm batch query encoding averaged about 0.04 ms/question and BM25 scoring about 5.2 ms/question on an RTX 4500 Ada; these exclude model/index loading and ranking/materialization, so they are component timings rather than end-to-end latency.

### Interpretation and limitations

This grouped holdout supports the claim that a query encoder trained on answer-partition supervision improves answer-partition ranking on unseen BRINK rules. It does not show correct answer generation, retrieval of supporting facts, faithfulness, or automatic topic linking. The test is an internal holdout from BRINK train, not BRINK’s official test. The query uses the correct canonical topic name, and only one graph partition/index configuration (`w_cosine`) was used. The rule-cluster interval is wide because 27 groups remain.

Artifacts:

- [Frozen rule split manifest](configs/brink_rule_holdout_20261028.json)
- [Frozen dev-selected test config](configs/brink_rule_holdout_frozen.json)
- [Fresh encoder training manifest/checkpoint](results/models/brink_rule_holdout_minilm_20261028/training_manifest.json)
- [Development aggregate and per-question records](results/raw/brink_rule_holdout_20261028/brink_dev_rule_holdout.json)
- [Held-out test aggregate](results/raw/brink_rule_holdout_20261028/brink_test_rule_holdout.json)
- [Held-out test per-question rankings](results/raw/brink_rule_holdout_20261028/brink_test_rule_holdout.jsonl)
- [Post-hoc metrics and candidate volume](results/reports/brink_rule_holdout_paper_analysis.json)
- [Integrity and file-hash provenance](results/reports/brink_rule_holdout_provenance.json)
- [H@k figure (PNG)](results/reports/brink_rule_holdout_h_at_k.png) · [PDF](results/reports/brink_rule_holdout_h_at_k.pdf)

Reproduction scripts: [grouped training](scripts/train_brink_rule_holdout_router.py), [dev/test evaluation](scripts/eval_brink_rule_holdout_router.py), and [paper analysis](scripts/analyze_brink_rule_holdout.py). Example commands are in [scripts/README.md](scripts/README.md).

## B. GTSQA: evidence retrieval with partition-guided graph search

This is a separate, frozen official-test study in [`new/partition_research`](../../new/partition_research). It compares multi-source BFS, structural Leiden, METIS-256, and a project-defined Leiden + constrained refinement traversal on the OGB WikiKG2 standard training graph. All methods share the same graph, seed entities, traversal, and exact edge-inspection budgets. GTSQA seeds are supplied by the benchmark, so the experiment is conditional on correct entity linking. No semantic text features are used.

The official GTSQA test has 1,622 questions. In the standard OGB training graph, 1,121 have all annotated evidence available, 470 have partial support, and 31 have none. The primary complete-support cohort is 1,121 questions. At the prespecified 16,109-edge budget (0.1% of the graph), results are:

| Method | Evidence recall | Complete annotated-evidence rate | Median latency |
|---|---:|---:|---:|
| Unpartitioned BFS | 0.7602 | 0.4969 | 6.30 ms |
| Leiden CPM, 2× weighted density | 0.8018 | 0.5843 | 11.89 ms |
| METIS-256 | 0.8268 | 0.6334 | 12.17 ms |
| **Leiden + constrained refinement** | **0.8299** | **0.6378** | 12.14 ms |

Refined Leiden improves evidence recall over BFS by **+6.97 points** (paired 95% CI [+5.67, +8.28]) and complete-evidence rate by **+14.09 points** (CI [+11.51, +16.77]). Its recall difference from METIS-256 is only **+0.31 points** (CI [−0.71, +1.33]); this does not establish superiority. At the 1% budget, METIS is numerically better. Partition-aware search therefore raises evidence coverage at fixed edge inspections but increases latency versus BFS, and refinement is statistically unresolved against METIS at the primary budget.

A new analysis reports the normalized area under the recall-versus-log-budget curve (three prespecified budgets, 1,611 / 16,109 / 161,092 edges; paired 20,000-resample bootstrap):

| Method | Normalized log-budget AUC | 95% question bootstrap CI |
|---|---:|---:|
| Unpartitioned BFS | 0.7559 | [0.7422, 0.7697] |
| Leiden CPM | 0.7937 | [0.7799, 0.8071] |
| METIS-256 | 0.8185 | [0.8055, 0.8315] |
| Leiden + constrained refinement | **0.8201** | [0.8071, 0.8329] |

Refined Leiden minus BFS AUC is +0.0641 (95% paired CI [+0.0535, +0.0749]); refined Leiden minus METIS AUC is +0.0016 (CI [−0.0063, +0.0097]). Do not call the latter methods different on this measure.

![GTSQA evidence recall and complete-evidence coverage by budget](../../new/partition_research/artifacts/paper_analysis/gtsqa_test_budget_quality.png)

The figure and JSON/CSV analysis are generated by [`make_paper_analysis.py`](../../new/partition_research/scripts/make_paper_analysis.py). Full dataset scope, test history, construction details, artifact integrity results, and secondary available-evidence results are in [`TEST_RESULTS_REPORT.md`](../../new/partition_research/TEST_RESULTS_REPORT.md) and [`EVALUATION_PROTOCOL.md`](../../new/partition_research/EVALUATION_PROTOCOL.md). The test eligibility audit inspected only support counts before the final retrieval run; the frozen configuration and artifact audit record hashes. The complete GTSQA result is an external benchmark result, but it is **graph traversal/evidence retrieval**, not BRINK partition ranking.

## What can be claimed now

A safe results statement is:

> On a rule-disjoint BRINK internal holdout, a fresh multi-positive fine-tuned MiniLM router increased answer-containing partition Hits@5 from 62.88% to 78.59% relative to the base MiniLM hybrid router (paired rule-cluster bootstrap 95% CI for the difference: +4.36 to +29.57 percentage points). In a separate frozen GTSQA/WikiKG2 test, graph partition-guided search increased evidence recall over unpartitioned BFS at equal edge-inspection budgets; constrained Leiden refinement reached 82.99% recall at 16,109 inspected edges, with no resolved advantage over METIS-256.

Keep the two clauses in separate result subsections and explain that they measure distinct units.

## Remaining work before submission

1. **Choose one central contribution.** The cleanest primary story is partition-guided evidence retrieval (GTSQA); use BRINK answer-partition routing as a separate complementary experiment. A single paper that combines them needs a clear shared research question and method diagram.
2. **Add stratified GTSQA results.** Report the frozen test method by hop count, unseen relation/generalization type, and evidence-subgraph topology, using the official GTSQA test metadata. Do not select methods from these strata.
3. **Strengthen efficiency reporting.** Repeated, controlled latency runs and peak memory/VRAM were not collected for the GTSQA test. Current single-pass latency is descriptive. Report construction time, peak memory, index size, edge cuts, size distribution, and repeated warm p50/p95 for each method, ideally using the training/validation questions for timing.
4. **Add structural controls.** On a development-only split, compare random partition labels and additional METIS/Leiden resolutions or seeds. Keep the frozen test out of this selection. Report community balance and cut with retrieval quality.
5. **Address annotation limits.** GTSQA annotations are not guaranteed exhaustive; report recall and complete annotated-subgraph coverage, not precision against unannotated edges. BRINK H@k is an answer-partition proxy, not evidence.
6. **If claiming system quality**, run the same generator and prompt over retrieved evidence, score official BRINK answer outputs, and audit faithfulness. Those overall answer metrics are not currently available.
7. **Freeze final manuscript claims and code.** Archive the exact data/index/model/config hashes and preserve raw per-question predictions. Cite the benchmark sources and the METIS/Leiden methods; describe the constrained refinement as a project-defined heuristic, not an established algorithm.

## Benchmark references

- BRINK benchmark: [ACL Anthology paper](https://aclanthology.org/2026.eacl-long.114/) and [official repository/scorer](https://github.com/boschresearch/brink).
- GTSQA ground-truth subgraphs: [paper](https://arxiv.org/abs/2511.04473) and [official dataset card](https://huggingface.co/datasets/Graphcore/GTSQA).
- OGB WikiKG2 graph: [official documentation](https://ogb.stanford.edu/docs/linkprop/#ogbl-wikikg2).
- METIS multilevel partitioning: Karypis and Kumar, 1998, [DOI](https://doi.org/10.1137/S1064827595287997).
- Leiden community detection: Traag, Waltman, and van Eck, 2019, [DOI](https://doi.org/10.1038/s41598-019-41695-z).
