# Paper benchmark protocol for Enhanced-KG-LLM

Updated 2026-10-27. This document separates established benchmark practice from the measurements currently available in this repository.

## 1. Claim boundaries

Use a metric only for the system stage it actually scores:

- **Entity linking:** rank the correct topic entity; report Hit/Recall@k, MRR, candidate-generation recall, ambiguity, and NIL handling. This does not measure answer retrieval.
- **Partition routing:** report whether a partition containing an annotated answer or evidence item is retrieved. Call this answer-containing partition Hits@k. It is not answer accuracy, evidence recall, or complete reasoning-path coverage.
- **Graph traversal:** report evidence-edge recall/precision at the same adjacency-entry budget and complete annotated-path coverage. A path is complete only if every required edge in at least one gold path was inspected/retrieved; state treatment of alternate paths.
- **Generation:** score the generated answer against references (normalized EM/F1 or benchmark-specific answer-set metrics) separately from retrieval. Measure whether answer claims are supported by retrieved evidence using evidence/citation precision and recall plus a documented human review or validated judge.
- **Overall system:** report end-to-end answer metrics and stage-wise latency/cost on the same questions, prompts, generator, evidence limits, and failure policy.

BRINK has its own answer-set evaluation. Preserve the raw answer output and report the benchmark's answer-level Hit/precision/recall/F1 and hard-answer metrics according to the official implementation. Do not replace these with partition Hits@k. Also disclose whether the graph contains the answer, whether hard answers require multi-hop reasoning, and the BRINK release/snapshot. See the [BRINK paper](https://aclanthology.org/2026.eacl-long.114/) and [official repository](https://github.com/boschresearch/brink).

## 2. Component protocols

### Dataset and split discipline

1. Hash and record every question, graph, answer, entity table, partition map, index, model checkpoint, prompt, and code revision.
2. Train learned encoders/rankers on train only. Select thresholds, fusion weights, traversal heuristics, prompts, and model checkpoints on validation only.
3. Freeze the complete configuration before one final test evaluation. The project test split has previously been inspected during iteration; treat any existing test result as exploratory. For a confirmatory paper claim, use a newly untouched split or independent benchmark.
4. Keep all candidates for a question together across splits. Where benchmark questions share templates/rules/entities, use grouped splits or report uncertainty clustered by the shared source group.
5. Report sample counts, exclusions, missing labels, and whether queries are original user strings, benchmark text, or metadata-normalized surrogates.

### Entity linker

For each question, save the full ranked candidate list, candidate-generation source, score, selected candidate, and NIL decision. Report Hit@1/3/5/10 and MRR; candidate-generation recall; coverage and accuracy conditional on non-NIL; ambiguity (for example, number of tied/near-tied candidates); and performance by mention form/entity popularity. Include NIL/unknown-topic examples to measure false-link rate. Evaluate the production graph-backed resolver as well as lexical/dense ablations when services are available.

The current BRINK run is an offline test of the hybrid lexical+dense linker with canonical topic name substituted into the original text. It lacks independent mention wording, NIL examples, and production Neo4j lookup. Label it accordingly.

### Partition retrieval

Use answer-containing partition Hits@1/5/10/50, hard-answer Hits@k, MRR, and (if partitions have graded relevance) nDCG@k, MAP, precision@k and recall@k. Report the number and size distribution of relevant partitions per question and candidate entities returned. Compare dense, lexical, fusion, and rank fusion on identical examples and the same frozen partition index. Give paired confidence intervals and disclose validation-based weight selection. BEIR provides a widely used retrieval evaluation precedent for nDCG, MAP, recall, and precision: [BEIR paper](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/65b9eea6e1cc6bb9f0cd2a47751a186f-Paper-round2.pdf).

A high answer-partition hit rate only says that the correct answer's partition was ranked. It does not show that traversal can reach the answer, that relevant evidence was returned, or that the generator used it faithfully.

### Traversal

Compare BFS, relation-guided, semantic best-first, and beam/path-history methods at identical adjacency-entry budgets (e.g., 25, 50, 100, 200). Freeze all seeds and query inputs. Report evidence-edge recall and complete-path coverage at each budget, plus area under the recall-versus-budget curve. Include ablations for relation similarity, entity similarity, structural/degree features, and path-history weight, and beam-width sensitivity selected on validation. Record adjacency entries, unique nodes/edges, scoring operations, peak memory, median/p95 latency, and timeout/failure counts. Never compare methods at equal wall-clock alone if their inspected-edge budgets differ.

### Answer generation and overall RAG

Pass each method's evidence to the same fixed generator and prompt, with equal context/token limits and decoding settings. Preserve raw outputs. Score BRINK answer-set metrics as the primary task score, and normalized EM/F1 only when text references support it. Separately measure evidence support: citation/evidence precision and recall, faithfulness, and answer relevance. Blind a human audit sample to system condition; if using an LLM judge, publish the rubric, judge model/version, prompts, calibration against human labels, and disagreement. ARES recommends evaluating context relevance, answer faithfulness, and answer relevance with lightweight human supervision and prediction-powered inference ([ARES](https://aclanthology.org/2024.naacl-long.20/)); RAGAS provides complementary reference-free RAG metrics ([RAGAS](https://aclanthology.org/2024.eacl-demo.16/)). These are diagnostics, not substitutes for task correctness.

Report end-to-end and stage-wise cold-start/warm latency (median, p95, p99), throughput, peak RAM/VRAM, token counts, and model/API cost. State hardware, concurrency, batching, cache policy, and whether graph/model loading is included. Report all failed, skipped, and fallback examples in denominators.

## 3. Statistical reporting

Use paired comparisons on the same questions. For binary Hits@k, report the absolute percentage-point difference and a 95% paired bootstrap interval. Use at least 10,000 resamples and a fixed disclosed seed. Because BRINK questions share rules, also bootstrap rule clusters (sample rule groups, then include their questions) and report the number of independent clusters. For several simultaneous comparisons, predefine primary comparisons or adjust the inferential procedure; do not present a validation-selected winner's interval as confirmatory evidence.

For stochastic generation, use paired question-level scores and repeat seeds or decoding runs where practical. Separate question sampling uncertainty from training seed, graph construction, and generation randomness. Report the selected metric and all tried configurations to avoid winner-only reporting.

## 4. Current status and reproducible commands

The current complete component results are recorded in [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md), with raw per-question artifacts in `results/raw/`. The current linked fine-tuned result is validation-only and selection-optimistic. No aggregate traversal or full generation benchmark is yet available because the live Neo4j and Ollama services were unavailable at the latest run. Do not invent or extrapolate those metrics from a one-question smoke check.

Example commands (run from project root, with the repository's Python environment and local data paths configured):

```bash
python metrics/scripts/benchmark_brink_hybrid_entity_linker.py \
  --questions /path/to/val_cleaned_final_complete.tsv \
  --hard-answers /path/to/val_cleaned_final.tsv --split val \
  --device cuda --out metrics/results/raw/brink_val_hybrid_entity_linker.json

python metrics/scripts/benchmark_brink_linked_routing.py \
  --string-run metrics/results/raw/brink_val_string_queries.jsonl \
  --linker-run metrics/results/raw/brink_val_hybrid_entity_linker.jsonl \
  --index-dir data/subg \
  --model /path/to/frozen-finetuned-minilm \
  --out metrics/results/raw/brink_val_linked_routing_ft_minilm.json

python metrics/scripts/benchmark_pipeline.py \
  --questions /path/to/val_cleaned_final_complete.tsv \
  --hard-answers /path/to/val_cleaned_final.tsv --split val \
  --out metrics/results/raw/pipeline_val.jsonl
```

The pipeline command requires reachable Neo4j and Ollama. Keep outputs in a new uniquely named run directory and preserve the emitted manifest. Do not run test tuning through these commands.

## 5. Appropriate external benchmarks and citations

- **BRINK** is the primary task match for incomplete-knowledge reasoning and answer-set evaluation. Follow its official annotation and scoring protocol, cite the paper and code, and identify the exact data snapshot.
- **GTSQA** can provide a separate evidence-subgraph retrieval/generalization evaluation. It uses a distinct graph/dataset substrate; report its results in a separate table and do not combine them with BRINK. Cite the [GTSQA paper](https://openreview.net/forum?id=N3WOfE4bvw) and [dataset card](https://huggingface.co/datasets/Graphcore/GTSQA).
- **BEIR** is a retrieval-evaluation precedent, not a replacement for BRINK's task-specific answer metrics.
- **KILT** is relevant when reporting knowledge-intensive task outputs with provenance/evidence. Cite [KILT](https://aclanthology.org/2021.naacl-main.200.pdf) if using its metrics or framing.
- **ARES** and **RAGAS** are evaluation frameworks for RAG quality dimensions; disclose their exact metric implementation and do not imply their scores are ground truth.
