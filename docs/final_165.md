# Capstone Team 165 — Complete Research and Experiment Record

**Purpose.** This is a standalone context document for a reader or LLM that has not seen the project or its conversation history. It combines the project’s main research record, later encoder experiments, controlled partition rebuild, a separate BRINK evidence-retrieval study, and the separate GTSQA/OGB study. It explains which results can be compared and which must remain separate.

**Workspace:** `/home/pesu-rf/Documents/Capstone_Team165/new`  
**Research scope:** partition-based retrieval over Wikidata knowledge graphs, especially BRINK-Wikidata5m under incomplete knowledge. A second study evaluates bounded evidence retrieval on GTSQA over OGB WikiKG2.  
**Most recent results included:** 2026-10-23.  
**Status:** multiple retrieval experiments are complete; no valid end-to-end LLM answer-quality comparison is complete.

---

## 1. Executive overview

This project has developed along four connected but distinct experimental tracks:

1. **Original BRINK partition benchmark.** Compare semantic edge-weighted Leiden partitions, unweighted partitions, raw versus cleaned communities, random partitions, and a one-community control using centroid plus BM25 retrieval. The best original result is cleaned linear-cosine `w_cosine`, answer-partition Hits@5 = **0.5845**, hard-answer Hits@5 = **0.5322**. Cleanup substantially improves raw Leiden; graph modularity does not select the best retrieval partition. The original weighting comparison is confounded by different Leiden objectives/resolutions and different embedding builds for some variants.
2. **BRINK query-encoder extension.** Keep the original 3,625-community `w_cosine` index fixed and adapt query encoders. An exploratory MPNet query encoder with a learned 768→384 projection reports official-test answer-partition Hits@5 = **0.9033** and hard Hits@5 = **0.8536**. The test set had already been used during iteration, so this is not a clean confirmatory estimate. A new rule-grouped internal split produces lower but cleaner results: fine-tuned MiniLM reaches **0.7804** answer H@5 and **0.7529** hard H@5 on the fixed original index.
3. **Controlled BRINK partition × encoder comparison.** Rebuild `w`, `w²`, `w³`, and unweighted partitions from one shared entity-embedding matrix, common graph topology, common Leiden objective/settings, and common cleanup. Benchmark zero-shot MiniLM, fine-tuned MiniLM, and the MPNet adapter on all four. On the internal rule-grouped test, best answer H@5 is **0.7196** (unweighted + fine-tuned MiniLM), and best hard H@5 is **0.6864** (`w³` + fine-tuned MiniLM). Partition differences are small and statistically unresolved. These numbers supersede any claim that the earlier weighting variants were fully controlled, but they do not reproduce the exploratory 90% result because the index and partition setup differ.
4. **BRINK evidence retrieval and GTSQA budgeted retrieval.** A separate, stricter BRINK experiment compares full-graph retrieval with structural and semantic Leiden-local evidence indexes on annotated reasoning facts. On held-out BRINK test, full graph has evidence recall@100 **0.64300**, structural **0.63299**, semantic **0.63252**; semantic versus structural is statistically unresolved. Separately, a GTSQA study on OGB WikiKG2 finds that partition-aware traversal can recover more annotated evidence than unpartitioned BFS at the same hard edge-inspection budget, but query latency is higher. Refined Leiden is competitive with METIS, not demonstrably better.

These are different tasks and metrics. **Never combine answer-partition Hits@K, evidence recall, GTSQA retrieval recall, and generated-answer quality in one leaderboard.** A partition hit means an answer entity belongs to a retrieved community; it does not prove the reasoning path was retrieved or an LLM answered correctly.

### Most defensible findings at present

- On the original BRINK partition benchmark, the cleanup procedure is associated with much higher answer-partition Hits@5 than raw Leiden outputs.
- On the original BRINK encoder setup, learned query-to-centroid alignment improves retrieval; fine-tuned MiniLM significantly beats zero-shot MiniLM on the grouped internal split.
- In the controlled partition × encoder grid, `w`, `w²`, `w³`, and unweighted partitions have similar fine-tuned MiniLM H@5 (0.712–0.720); the current experiment does not establish a winner among weight transforms.
- In the separate BRINK evidence study, partition-local retrieval loses about one percentage point of evidence recall@100 versus full-graph retrieval. Semantic weighting does not beat structural partitioning on held-out test.
- In GTSQA, constrained-refined Leiden improves evidence retrieval versus unpartitioned BFS and unrefined Leiden at the primary 0.1% edge-inspection budget; it is statistically unresolved against METIS-256 and is slower than BFS in the measured implementation.
- There is no completed LLM generation experiment with scored outputs. Therefore there is no result yet for end-to-end answer quality, factuality, or GraphRAG performance.

---

## 2. How to interpret the measures

### 2.1 BRINK answer-partition retrieval

For a question, the system ranks communities/partitions. Gold answer entity IDs are mapped to their community IDs. **Any-answer partition H@K** is 1 when at least one valid answer’s community is among the top K communities. **Hard-answer partition H@K** uses the one “hard” answer whose directly supporting triple was removed in BRINK. “Hits@5” in this project generally refers to the any-answer partition metric unless explicitly marked hard.

This metric does not check whether the query’s topic entity and answer are co-located, whether the removed triple’s alternative evidence path is inside the selected communities, or whether an LLM can produce the right answer. It can also improve when larger communities return more entities. Pair it with hard H@K, MRR, answer candidate recall, number of entities/facts returned, and evidence retrieval when making claims.

BRINK questions can have **multiple valid answer entities**. The complete answer file supplies the any-answer positives; the cleaned answer file supplies the single hard answer. The hard answer is one designated answer, not the only correct answer.

### 2.2 Evidence retrieval

Evidence recall measures how many annotated premise facts (or facts on the best annotated path) the retriever returns. Complete-path coverage measures whether all premises of at least one annotated path are retrieved. These are distinct from answer-partition Hits@K. BRINK path annotations may not enumerate all valid evidence, so a missed annotation cannot automatically be called a false positive/negative in the general KG sense.

### 2.3 GTSQA budgeted retrieval

GTSQA results report macro recall over released evidence triples under a strict cap on the number of graph edges inspected. “Complete evidence” is the share of questions where all provided gold evidence is recovered. This is not the same as BRINK’s returned-fact K curve: one caps graph inspections; the other caps the evidence list returned after traversal.

### 2.4 Statistical uncertainty

Paired question or rule-cluster bootstrap intervals quantify sampling uncertainty for the saved evaluation questions. They do not quantify random variation from retraining a model or rerunning Leiden with other seeds. A single model seed and single partition seed remain limitations even when bootstrap intervals are available.

---

## 3. Data, graph snapshots, and leakage controls

### 3.1 Wikidata5m and BRINK

BRINK-Wikidata5m is designed for reasoning under missing knowledge: a directly supporting head triple is removed, while annotated alternative premises remain available for reasoning. The local pinned release used by the project has:

| Split | Questions |
|---|---:|
| Train | 27,719 |
| Validation | 3,465 |
| Test | 3,464 |

Each split has a hard-answer file and a complete-answer file. Use the local TSVs once per question; do not count both file variants as separate examples. The Hugging Face release README gives counts three higher in total than the pinned downloaded local files; reports should identify the exact local snapshot.

The local metadata graph contains 4,813,491 entity IDs and 825 relation IDs. The original partition construction used a local structural triple file with 20,598,521 rows and hub relations removed; the graph has 3,529,659 structural nodes and 10,454,598 simplified edges. The newer controlled BRINK evidence study instead uses the official incomplete file with 20,478,006 facts and 20,477,248 non-loop projected facts.

**Leakage warning:** local complete triples contain 32,067 of the 32,101 facts that BRINK withheld from its complete graph. Do not use the local complete graph as the BRINK incomplete-knowledge search graph. Use the pinned official `incomplete_facts.tsv` for evidence-retrieval experiments.

Compatibility audit findings:

- 4,844 graph endpoint QIDs are not present in local entity metadata (the incomplete graph endpoints otherwise resolve at 99.895%). Keep graph-only nodes; give them empty text/neutral semantic weights where needed.
- Three relation IDs (`P1962`, `P2439`, `P3484`) used by 877 graph facts are missing from local relation tables.
- For BRINK test, 3,148/3,464 questions have complete annotated premise paths in the official incomplete graph; 316 do not. Evidence recall uses the supported subset and reports this coverage. Official answer scoring, if later done, should still cover all benchmark questions.

### 3.2 GTSQA and OGB WikiKG2 (separate study)

GTSQA is a different QA/evidence benchmark. It is paired here with the standard OGB `ogbl-wikikg2` training graph of 16,109,182 triples. The official GTSQA test has 1,622 questions:

- 1,121 have all annotated gold evidence facts in the search graph (primary full-evidence evaluation).
- 470 have partial evidence support.
- 31 have no supported gold facts.
- The secondary available-evidence analysis uses 1,591 questions with at least one supported fact.

This benchmark includes seed entities and tests retrieval conditional on correct entity linking. The study is structural; no semantic metadata weighting was used because metadata coverage/provenance was not sufficient for that graph.

---

## 4. Original BRINK partition study

### 4.1 Entity and graph construction

Entity vectors use `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions and L2 normalization. Two embedding builds exist:

| Build | Text template | Description | Aliases |
|---|---|---|---|
| Earlier `partition/build_embeddings.py` | `name + ': ' + description[:200]` | Up to 200 characters | No |
| `new_new/step1_embeddings.py` | name, “Also known as” aliases, description | Up to 300 characters | Up to 8 aliases |

The full vector matrix has shape `(4,813,491, 384)` and occupies about 7.4 GB in float32. The structural graph filters hub relations `{P31, P17, P27, P131, P19, P20, P21, P106}` before semantic partitioning, leaving 11,087,274 structural triple rows; duplicate/reciprocal rows are simplified into 10,454,598 undirected graph edges.

For normalized vectors `e_u`, `e_v`, the old weighting pipeline computes:

`cos(u,v) = e_u · e_v`  
`w_p(u,v) = (cos(u,v) + shift)^p`, where `shift = abs(min cosine) + 1e-4`.

The old stored transforms are `p=1` (`w`), `p=2` (`w²`), and `p=3` (`w³`). The original file medians were approximately 0.5376, 0.2884, and 0.1544 respectively. Squaring/cubing sharpens weight contrast. Unweighted sets every simplified edge to weight 1.

### 4.2 Leiden and cleanup variants

The historical runs did not fully control the Leiden objective/resolution:

| Variant | Leiden objective/settings | Raw communities | Notes |
|---|---|---:|---|
| `w` | CPM, resolution 0.00003 | 215,474 | 181,531 singletons; median size 1 |
| `w²` | RBConfiguration, resolution 3.0 | 21,807 | Median size 2 |
| `w³` | RBConfiguration, resolution 3.0 | 23,923 | Median size 2 |
| Unweighted | RBConfiguration, resolution 1.0 | 17,940 | Median size 2 |

All used seed 42 and two iterations. The four-phase cleanup is: (A) split oversized communities using MiniBatchKMeans embeddings; (B) merge communities smaller than 150 by semantic centroid similarity with increasing size caps; (C) merge isolated communities to nearest centroids subject to caps; (D) split any remaining oversized communities. Final splits target approximately 1,200–1,500 graph nodes; final max size is usually around 5,000. Non-graph/hub-only entities (1,283,832) are later assigned to nearest centroids under size cap 8,000.

### 4.3 Original answer-partition benchmark

The original 10-condition benchmark used centroid + BM25, with equal-weight Reciprocal Rank Fusion (RRF, `k=60`). Query QIDs were replaced by entity names; the partition benchmark itself did not use the later linked-query fine-tuned encoders. Results on 3,464 BRINK test and 3,465 validation questions:

| Variant | Communities | Test any H@5 | Val any H@5 | Test MRR@5 | Test hard H@5 | Test nDCG@5 | nDCG@10 | Weighted modularity | Mean conductance | Query latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Clean `w_cosine` | 3,625 | **.5845** | .5856 | **.4265** | **.5322** | .437 | .469 | .453 | .208 | 6.1 ms |
| Clean `w³` | 2,837 | .560 | .574 | .4052 | .498 | .400 | .433 | .456 | .279 | 4.8 ms |
| Clean `w²` | 2,851 | .553 | .550 | .4006 | .481 | .389 | .424 | .348 | .313 | 4.8 ms |
| Clean unweighted | 2,447 | .526 | .531 | .3693 | .433 | .351 | .387 | .249 | .339 | 4.1 ms |
| Raw Leiden `w³` | 23,923 | .453 | .461 | .3453 | .445 | .366 | .385 | .880 | .005 | 36.6 ms |
| Raw Leiden `w²` | 21,807 | .443 | .447 | .3251 | .434 | .349 | .370 | .842 | .006 | 36.8 ms |
| Raw Leiden `w` | 215,474 | .303 | .315 | .2029 | .294 | .223 | .243 | .721 | .351 | 257.7 ms |
| Raw Leiden unweighted | 17,940 | .292 | .291 | .1788 | .287 | .204 | .233 | .832 | .001 | 27.1 ms |
| Random, 2,851 groups | 2,851 | .033 | .032 | .0173 | .010 | .012 | .019 | ≈0 | .493 | 5.0 ms |
| Single community | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 | 0.04 ms |

The single-community control gets perfect Hits because it returns the whole graph and is not useful retrieval. Precision when treating every returned entity as a prediction is effectively zero. The random condition is near chance. Do not present `single` as a method.

Original cleaned partition size/quality details:

| Variant | Min / median / max graph community size | Size entropy | Gini | Cosine silhouette | CH | DB |
|---|---|---:|---:|---:|---:|---:|
| Clean `w` | 150 / 751 / 4,955 | .963 | .428 | −.104 | 3.06 | 2.83 |
| Clean `w²` | 150 / 1,182 / 4,756 | .984 | .271 | −.077 | 3.92 | 3.08 |
| Clean `w³` | 150 / 1,171 / 5,000 | .982 | .293 | −.089 | 3.88 | 3.09 |
| Clean unweighted | 1 / 1,305 / 4,967 | .978 | .320 | −.065 | 4.71 | 3.10 |

Raw Leiden produced extremely skewed sizes (raw unweighted maximum 385,499; raw `w` had many singletons). The single-community baseline illustrates why H@K alone can be gamed. For the best historical RRF partition `w`, treating retrieved entities as predictions gave Hits@Any .584, Hits@Hard .532, HHR .911 at K=5; precision was .0005 because communities return thousands of entities.

### 4.4 Interpretation and limitations of original partition results

- Cleanup changed H@5 by approximately +28.1 pp for `w`, +23.4 pp unweighted, +11.0 pp `w²`, +10.7 pp `w³` relative to raw memberships. The cleanup package is therefore central to the result.
- Raw partitions had higher modularity and lower conductance in some cases but worse retrieval. Modularity/conductance are graph objectives, not task metrics.
- Clean `w` numerically outperformed `w³`, `w²`, and unweighted; however, historical runs confound weighting with Leiden objective/resolution, community count, cleanup implementation details, and for some assets embedding build.
- `w` and `w²` assets were built from different entity-embedding construction pipelines. This concern motivated the controlled rebuild in §7.
- There were no seed repeats in the original table. Structural quality metrics such as silhouette/CH/DB were sampled and are indicative only.
- A later compatibility audit found three missing graph relations and graph-only QIDs; benchmark metrics and local graph source must be stated explicitly.

### 4.5 Difference between original fusion protocols

The apparent `w_cosine` mismatch is from different scripts, not a single changed metric:

| Protocol | Fusion | Any H@5 | Hard H@5 |
|---|---|---:|---:|
| Original partition benchmark | Equal-weight RRF, `k=60` | .58446 | .53220 |
| Realistic weighted-fusion run | Per-query min-max normalized dense/BM25; BM25 weight .55 | .58995 | .51718 |

These are separate configurations. Preserve protocol names when citing them.

---

## 5. BRINK query encoder experiments

### 5.1 Candidate index and query protocol

The 90% line of work kept the original cleaned `w_cosine` candidate partition index fixed (3,625 communities, original MiniLM entity centroids). Query QIDs were name-substituted and the existing entity linker generated query augmentation; the gold topic entity was not injected in the best setup. The dense query score and BM25 score were normalized per query to `[0,1]`, then fused:

`score = (1 − w_bm25) × dense_norm + w_bm25 × BM25_norm`.

The answer candidate set is ranked partitions. The model does not embed/retrieve individual answer nodes directly. A partition is relevant when it contains any valid answer entity or, separately, the designated hard answer.

### 5.2 Historical sequence (mixed provenance)

| Stage | Test H@5 | Validation H@5 | Status / interpretation |
|---|---:|---:|---|
| Base MiniLM, raw query + RRF | .5845 | .5856 | Historical starting point; same headline as original benchmark |
| Entity-linked query + weighted fusion (`w_bm25=.8`) | .7182 | .7255 | Reported in prior handoff; no independent raw JSON/predictions found |
| Fine-tuned MiniLM, linked + weighted fusion | .8885 | .8802 | Prior handoff, 5 epochs; exact run provenance incomplete |
| Re-embed entities with fine-tuned encoder | .8458 | — | Rejected; query encoder was trained to original centroid space |
| Oracle topic instead of linker | .8516 | — | Rejected as reported; linker format performed better |
| Multi-positive hard-negative MiniLM | .8943 | .8880 | Recorded v2 run; hard-answer H@5 not in short summary |
| Base/fine-tuned dense score ensemble | .8943 | .8880 | Dev selected no base contribution; no gain |
| MPNet adapter | **.9033** | **.9019** | Recorded v2 run; test result exploratory, detailed below |

The historical handoff also included “H@50 = .865” alongside H@5 = .8885. Those figures cannot be the same monotone Hits@K measure on the same denominator; do not cite the H@50 claim without a corrected source. The `.7182`, `.8885`, `.8458`, and `.8516` scores should be labeled handoff-reported unless original predictions/configs are recovered.

### 5.3 Fine-tuning and MPNet adapter

The earlier MiniLM fine-tuning selected one gold answer partition as a class. If a question has multiple valid answer partitions, this can accidentally treat other correct answers as negatives. A subsequent multi-positive objective marked every mapped complete-answer partition positive and contrasted their combined probability mass against online hard negatives.

**MPNet adapter method:**

- Base query encoder: `sentence-transformers/all-mpnet-base-v2` (768 dimensions).
- A trainable bias-free linear projection maps 768 to 384 dimensions, matching frozen original MiniLM centroids.
- The 4.8M entity vectors, partition assignments, and retrieval index are not recomputed.
- BRINK train questions train the query model; complete-answer partition labels are multi-positive.
- 64 highest-scoring non-gold partitions are online hard negatives; candidate-set multi-positive log-softmax, temperature .05.
- AdamW; learning rate `1e-5`, weight decay .01, batch 32, gradient clip 1.0, six epochs, linear LR decay to 20%, seed 20261022, BF16 autocast.
- Linked questions are used. Validation selects checkpoint and then fusion weight; BM25 weight .42 was selected from a .30–.50 sweep by .01 increments.

| Split | Scored | BM25 weight | H@1 | H@3 | H@5 | H@10 | Hard H@5 | MRR@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 3,465/3,465 | .42 | — | — | .9019 | — | .8589 | .7981 |
| Test | 3,463/3,464 | .42 | .7320 | .8651 | **.9033** | .9344 | .8536 | .8014 |

Candidate recall at top K for this saved test evaluation:

| Metric | H@1 | H@3 | H@5 | H@10 | H@20 | H@50 | H@100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Any answer | .7320 | .8651 | .9033 | .9344 | .9555 | .9755 | .9847 |
| Hard answer | .6624 | .8097 | .8536 | .8934 | .9180 | .9443 | .9590 |

One test question had no mapped answer partition; the model scored 3,463 questions. Approximate Wilson 95% intervals computed from the aggregate counts are .8930–.9127 for any H@5 and .8414–.8650 for hard H@5; these are descriptive binomial intervals and do not correct for test-set selection.

### 5.4 Follow-up attempts around the 90% result

- A validation fusion comparison of min-max fusion, RRF, rank-percentile/Borda, and z-score fusion did not improve on min-max weighted fusion at `w_bm25=.42` (H@5 .9019). Strongest listed z-score result was .9004.
- A dense-score ensemble of base MiniLM and multi-positive fine-tuned MiniLM selected zero base-model contribution on validation; no gain.
- A numeric MLP listwise reranker used ten features per top-100 candidate: dense/BM25 min-max scores, rank percentiles, log community size, interaction, combined score, rank gap, and margins from query maxima. Positive coverage was 99.25% train and 98.47% validation. Best validation any H@5 was **.8941** (epoch 1), below the first-stage .9019; it was not evaluated on test. This is a negative result for this numeric feature reranker, not evidence that all reranking fails.
- Re-embedding entities with the fine-tuned query encoder lowered reported test H@5 to .8458. It changes the centroid target space after query training, so it is not a fair replacement without retraining/alignment.
- Oracle topic injection yielded a reported .8516. It was not beneficial in this configuration; this is a reported exploratory result, with no retained run artifact in the current audit.

### 5.5 Limits of the 90.33% result

The supplied official test was used to compare models, fusion settings, and query variants. Therefore .9033 is a strong **exploratory observed retrieval score**, not an unbiased final generalization estimate. It is not 90.33% answer accuracy. Hard H@5 is lower (.8536). There is no saved per-question matched baseline prediction set for a paired CI, no training-seed variance, and no generated-answer score. The full improvement from .5845 to .9033 changes query text/linking, fusion, and model training, so it is not a single-factor causal effect.

---

## 6. Rule-grouped internal encoder benchmark on the original `w_cosine` index

To reduce question-rule leakage and avoid further tuning the already-used official test, a split was built from BRINK training rows and grouped by `rule_index`:

| Split | Questions | Rule groups |
|---|---:|---:|
| Train | 19,363 | 137 |
| Development | 4,208 | 30 |
| Internal test | 4,148 | 29 |

No rule group crosses splits. All systems used the same 3,625-community original `w_cosine` index, raw questions with QIDs replaced by entity names, no linker/oracle, and dev-selected fusion. Three encoders were compared: zero-shot MiniLM, fine-tuned MiniLM, and MPNet adapter. Fine-tuned models used one seed. The raw answer/H@K is grouped by answer-containing partitions; one hard-answer entity was unmapped in internal test and is counted as a miss.

| Model | Dev BM25 w | Dev any H@5 | Test any H@5 | Test hard H@5 | Test MRR any@10 | Any H@50 | Any H@100 | Mean entities in top 5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot MiniLM | .70* | .6528 | .6873 | .6497 | .5557 | .8799 | .9202 | 9,210 |
| Fine-tuned MiniLM | .52 | .7322 | **.7804** | **.7529** | **.6550** | .9055 | .9257 | 9,884 |
| MPNet adapter | .48 | **.7505** | .7399 | .7124 | .6399 | **.9118** | **.9540** | 10,114 |

`*` Zero-shot chose the maximum tested BM25 weight (.70); a wider dev-only sweep could change this baseline. Internal-test answer coverage was 4,148/4,148 and hard-answer coverage 4,147/4,148. Fine-tuned MiniLM’s paired improvement over zero-shot was +9.31 pp any H@5 (rule-cluster bootstrap 95% CI [+5.77, +13.08]) and +10.32 pp hard H@5 ([+6.75, +13.99]). MPNet versus fine-tuned MiniLM was −4.05 pp any H@5 ([-9.29, +0.51]); this interval overlaps zero. MPNet has higher top-100 recall but weaker top-5 than fine-tuned MiniLM in this split.

Full H@K curve (same internal test):

| Model / measure | @1 | @3 | @5 | @10 | @20 | @50 | @100 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MiniLM zero-shot any | .4561 | .6261 | .6873 | .7628 | .8247 | .8799 | .9202 | .5557 |
| Fine-tuned MiniLM any | .5641 | .7201 | .7804 | .8344 | .8727 | .9055 | .9257 | .6550 |
| MPNet adapter any | .5591 | .6996 | .7399 | .7973 | .8587 | .9118 | .9540 | .6399 |
| MiniLM zero-shot hard | .4120 | .5817 | .6497 | .7312 | .7968 | .8575 | .9014 | .5135* |
| Fine-tuned MiniLM hard | .5101 | .6847 | .7529 | .8151 | .8573 | .8951 | .9183 | .6118* |
| MPNet adapter hard | .5031 | .6632 | .7124 | .7775 | .8435 | .9012 | .9474 | .5956* |

`*` Hard MRR@10.

Answer cardinality matters: 2,425/4,148 questions have one valid answer; fine-tuned MiniLM gets any H@5 .6804 and hard H@5 .6726 on that subset. Of 191 questions with four valid answers, any H@5 is .9372. These are descriptive, confounded by rule composition, and should be reported with support counts.

---

## 7. Controlled partition rebuild and 12-condition encoder grid

### 7.1 Why it was run

The initial paper record states that the historical `w` and `w²` graph builds used different entity embedding construction pipelines and different Leiden settings. That weakens causal interpretation of the weight comparison. The controlled rebuild uses one shared embedding matrix, graph topology, Leiden objective, seed/iterations, cleanup implementation, index builder, query split, and encoder checkpoints.

### 7.2 Shared partition construction

- Embeddings: root `/partition/entity_embeddings.npy`, shape `(4,813,491,384)`, memory-mapped; no re-embedding.
- Topology: common simplified graph of 3,529,659 nodes and 10,454,598 unique undirected edges. Structural triple rows were mapped to the same simplified topology.
- Shared cosine shift: `0.2806964946746826`.
- Raw weight variants: `(cosine + shift)^p`, with p = 1, 2, or 3; duplicate raw rows are aggregated on simplified edges. Unweighted is one per simplified edge.
- Leiden: RBConfigurationVertexPartition, resolution 3.0, seed 42, two iterations for all four variants.
- Cleanup: same A/B/C/D methods as original `step4_cleanup.py`.
- Entity/index construction: same shared root embedding file, entity ID map, names/aliases, nearest-centroid assignment of all non-graph entities, size cap 8,000, centroids over all assigned entities, BM25 docs from names/aliases.

| Variant | Raw communities | Clean communities | Min / median / max graph-node community size |
|---|---:|---:|---|
| `w` | 18,730 | 2,824 | 152 / 1,180.5 / 4,970 |
| `w²` | 21,807 | 2,863 | 150 / 1,172 / 4,759 |
| `w³` | 23,923 | 2,834 | 150 / 1,163 / 5,000 |
| Unweighted | 17,881 | 2,795 | 150 / 1,173 / 4,861 |

The results are nearly matched in final community count, but not exactly. Weighting changes the community structures, and cleanup may create slightly different counts. This is not a matched-final-granularity experiment and has one Leiden seed.

### 7.3 Evaluation protocol

Use the same 19,363/4,208/4,148 rule-grouped split from §6; same raw name-substituted queries; same three encoder checkpoints; no linker or oracle. Build one temporary complete index per partition to fit available disk. For each of 12 model × partition cells, score dense centroids and BM25, min-max normalize per query, select BM25 weight from `{0,.05,...,1}` by development any H@5, and score test once. Each cell has its own dev-selected BM25 weight. The test set contains 29 rule groups.

### 7.4 Main test results

| Partition | Communities | Encoder | Dev-selected BM25 weight | Any H@5 | Hard H@5 | Any H@10 | Any MRR@10 | Any H@100 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `w` | 2,824 | Zero-shot MiniLM | .50 | .6459 | .5899 | .7249 | .4861 | .9024 |
| `w` | 2,824 | Fine-tuned MiniLM | .55 | .7155 | .6690 | .7825 | .5510 | .9535 |
| `w` | 2,824 | MPNet adapter | .50 | .6866 | .6418 | .7642 | .5361 | .9581 |
| `w²` | 2,863 | Zero-shot MiniLM | .45 | .6418 | .5829 | .7155 | .4899 | .8843 |
| `w²` | 2,863 | Fine-tuned MiniLM | .50 | .7122 | .6647 | .7828 | .5590 | .9405 |
| `w²` | 2,863 | MPNet adapter | .55 | .6591 | .6114 | .7466 | .5046 | .9332 |
| `w³` | 2,834 | Zero-shot MiniLM | .65 | .6553 | .6087 | .7449 | .5109 | .9053 |
| `w³` | 2,834 | Fine-tuned MiniLM | .50 | .7187 | **.6864** | .8026 | **.5835** | .9549 |
| `w³` | 2,834 | MPNet adapter | .55 | .6912 | .6586 | .7674 | .5483 | .9585 |
| Unweighted | 2,795 | Zero-shot MiniLM | .55 | .6454 | .5841 | .7235 | .4878 | .8836 |
| Unweighted | 2,795 | Fine-tuned MiniLM | .30 | **.7196** | .6757 | **.8067** | .5614 | .9433 |
| Unweighted | 2,795 | MPNet adapter | .50 | .6721 | .6227 | .7568 | .5100 | .9458 |

The unweighted/fine-tuned MiniLM system has the highest any-answer H@5; `w³`/fine-tuned MiniLM has the highest hard H@5 and any MRR@10. Their differences are very small and not decisive.

| Encoder | Partition | Any H@1 | Any H@3 | Any H@5 | Any H@10 | Any H@20 | Any H@50 | Any H@100 | Hard H@5 | Hard H@100 | Hard MRR@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MiniLM zero-shot | `w` | .3725 | .5699 | .6459 | .7249 | .7944 | .8587 | .9024 | .5899 | .8751 | .4263 |
| MiniLM fine-tuned | `w` | .4332 | .6514 | .7155 | .7825 | .8368 | .9084 | .9535 | .6690 | .9405 | .4907 |
| MPNet adapter | `w` | .4238 | .6273 | .6866 | .7642 | .8416 | .9166 | .9581 | .6418 | .9467 | .4731 |
| MiniLM zero-shot | `w²` | .3778 | .5634 | .6418 | .7155 | .7751 | .8385 | .8843 | .5829 | .8592 | .4298 |
| MiniLM fine-tuned | `w²` | .4407 | .6492 | .7122 | .7828 | .8435 | .9024 | .9405 | .6647 | .9284 | .4965 |
| MPNet adapter | `w²` | .3826 | .5931 | .6591 | .7466 | .8394 | .8922 | .9332 | .6114 | .9192 | .4425 |
| MiniLM zero-shot | `w³` | .4084 | .5735 | .6553 | .7449 | .8177 | .8744 | .9053 | .6087 | .8855 | .4593 |
| MiniLM fine-tuned | `w³` | .4834 | .6507 | .7187 | .8026 | .8575 | .9185 | .9549 | .6864 | .9438 | .5332 |
| MPNet adapter | `w³` | .4455 | .6186 | .6912 | .7674 | .8457 | .9190 | .9585 | .6586 | .9479 | .4960 |
| MiniLM zero-shot | Unweighted | .3809 | .5586 | .6454 | .7235 | .7878 | .8469 | .8836 | .5841 | .8563 | .4215 |
| MiniLM fine-tuned | Unweighted | .4436 | .6555 | .7196 | .8067 | .8623 | .9151 | .9433 | .6757 | .9311 | .4992 |
| MPNet adapter | Unweighted | .3898 | .5988 | .6721 | .7568 | .8423 | .9094 | .9458 | .6227 | .9308 | .4421 |

### 7.5 Paired rule-cluster uncertainty

10,000 paired bootstrap draws resampled the 29 held-out `rule_index` clusters. The test set is 4,148 questions. Selected contrasts:

| Contrast | Metric | Difference | 95% interval |
|---|---|---:|---:|
| `w³ − w`, fine-tuned MiniLM | Any H@5 | +0.0031 (+0.31 pp) | [−0.0358, +0.0391] |
| `w³ − w`, fine-tuned MiniLM | Hard H@5 | +0.0174 (+1.74 pp) | [−0.0249, +0.0557] |
| Fine-tuned − zero-shot, `w³` | Any H@5 | +0.0634 (+6.34 pp) | [+0.0180, +0.1057] |
| Fine-tuned − zero-shot, `w³` | Hard H@5 | +0.0776 (+7.76 pp) | [+0.0314, +0.1204] |
| MPNet − zero-shot, `w³` | Any H@5 | +0.0359 (+3.59 pp) | [−0.0235, +0.0963] |
| MPNet − zero-shot, `w³` | Hard H@5 | +0.0499 (+4.99 pp) | [−0.0122, +0.1109] |

Fine-tuning MiniLM has a positive paired interval versus zero-shot on `w³`; partition-weight contrasts overlap zero. One checkpoint/seed per model means these intervals exclude training seed variation.

### 7.6 Conclusions and limitations

- Shared embeddings/objective/settings address the largest known old `w` vs `w²` confound.
- The partition H@5 range for fine-tuned MiniLM is only .7122–.7196. This run does not establish that a particular exponent is better.
- The 90.33% MPNet result was not reproduced here because this experiment has a different partition/index and an unseen-rule internal test protocol. The two scores do not contradict one another; they measure different systems on different experimental setups.
- Final communities are similar in count, but not exact matched granularity. No multi-seed robustness estimate.
- Fusion weight selection is independent for every model/partition, which is flexible; report full dev selection and consider a frozen common-weight comparison in a confirmatory study.
- These are answer-partition metrics, not evidence path or generated QA scores.

---

## 8. Separate BRINK study: structural versus semantic evidence retrieval

This track uses the official incomplete BRINK graph and explicit annotated premise paths. It is distinct from centroid/BM25 answer-partition retrieval. Its goal is to test whether communities improve a deterministic evidence retriever, not whether the answer-containing partition ranks in top K.

### 8.1 Graph and partition method

- Pinned graph: BRINK `incomplete_facts.tsv`, 20,478,006 source triples; 758 self-loops are excluded from the undirected projection, leaving 20,477,248 edges.
- Shared graph nodes: 4,818,335 (includes graph-only QIDs); all source triples/relation IDs are retained except projection self-loops.
- Structural condition: every projected edge weight 1.
- Semantic condition: pinned MiniLM entity metadata embeddings; `1 + lambda * max(0, cosine)` with lambda .5, normalized so mean edge weight is 1. Missing embeddings use neutral weight 1.
- Leiden CPM, same seed 42, two iterations, same resolution within each pair. Five predeclared CPM resolution candidates are fractions `{.25,.5,1,2,4}` times global graph density.
- Validation chose shared resolution `4.4100925283e-07` (0.25× graph density) by maximizing mean structural/semantic evidence recall@100. This is a validation-only choice.
- The deterministic retrieval anchors at BRINK’s provided topic entity, searches only facts internal to that entity’s assigned community (or all graph for full baseline), ranks by minimum undirected hop distance to either endpoint, breaks ties by sorted triple IDs, and returns K facts for K ∈ {10,25,50,100,200}.

Validation resolution selection across the complete predeclared grid (annotated evidence recall@100 on 3,144 supported validation questions):

| CPM density multiplier | Resolution | Structural recall | Semantic recall | Mean used for selection |
|---:|---:|---:|---:|---:|
| .25× | 4.4101e−7 | .63088 | .63200 | **.63144** |
| .5× | 8.8202e−7 | .62150 | .62792 | .62471 |
| 1× | 1.7640e−6 | .59542 | .60586 | .60064 |
| 2× | 3.5281e−6 | .60189 | .60512 | .60350 |
| 4× | 7.0561e−6 | .60597 | .59887 | .60242 |

The selected value is not a semantic-only optimum: the rule maximized the mean of structural and semantic validation recall, then froze that shared setting for the held-out system comparison.

The dataset’s local metadata descriptions cover only part of the graph; this was addressed with an explicit neutral-weight policy. The partition graph is transductive over the released incomplete graph, not a claim of inductive generalization.

### 8.2 Held-out test evidence outcomes

There are 3,464 test questions; 3,148 have a complete annotated premise path in the graph and are the evidence metric denominator. The remaining 316 are unsupported and excluded from evidence quality metrics, but included in latency count. No topic entities were absent.

| System | Evidence recall@10 | @25 | @50 | @100 | @200 | Complete-path coverage@100 |
|---|---:|---:|---:|---:|---:|---:|
| Full graph | .45415 | .57396 | .62161 | **.64300** | **.65523** | **.36944** |
| Structural partition | .44812 | .56650 | .61256 | .63299 | .64522 | .36245 |
| Semantic partition | .45002 | .56713 | .61176 | .63252 | .64554 | .36277 |

At K=100, paired bootstrap (10,000 replicates; 3,148 supported questions):

| Difference | Recall delta (95% CI) | Complete-path delta (95% CI) |
|---|---|---|
| Structural − full | −.01001 [−.01382, −.00619] | −.00699 [−.01144, −.00254] |
| Semantic − full | −.01048 [−.01461, −.00667] | −.00667 [−.01112, −.00222] |
| Semantic − structural | −.00048 [−.00461, +.00365] | +.00032 [−.00445, +.00508] |

Partition-local retrieval has about one point lower annotated evidence recall than full graph. Semantic versus structural partitioning is unresolved; the study does not support that semantic weighting improves held-out evidence retrieval. At the selected resolution, validation showed a small positive semantic-minus-structural difference of +.00111 with interval [−.00239,+.00461], also inconclusive.

### 8.3 Runtime and storage

Single pass per question after five warmups; descriptive, not repeated-run timing. Test latency and index component:

| System | Median / mean / p95 latency | Indexed edges | Index build | Index + membership bytes |
|---|---|---:|---:|---:|
| Full graph | 114.97 / 397.62 / 1,264.65 ms | 20,477,248 | 9.84 s | 202,364,672 |
| Structural partition | 106.44 / 380.14 / 1,201.35 ms | 19,766,197 | 10.01 s | 215,949,604 |
| Semantic partition | 79.17 / 347.45 / 1,142.68 ms | 19,245,388 | 9.96 s | 211,783,132 |

The one-pass timings suggest lower latency in this implementation for the local semantic index, but the experiment did not repeat/randomize system order, so it does not establish stable speedup. This is retrieval-only, not end-to-end latency; no GPU was used for these timings. Partition indexes use fewer edges but have a larger reported index+membership size due to the additional membership array.

### 8.4 Boundary audit and follow-up screening

Among 5,238 question–gold-fact incidences over 3,148 supported questions, partition boundaries cut 86 structural gold facts (1.64%) and 114 semantic facts (2.18%). Those cut facts account for only 3.79% and 5.02% of missed gold facts at K=100. Most missing evidence is internal to the chosen community, so simply replicating boundary facts cannot explain or fix most misses; any boundary-index strategy needs a fixed budget and a new evaluation.

Other proposed directions have no result yet:

- Query-adaptive routing by gold path length was only an oracle subgroup analysis; true path length is unavailable at query time. No deployable router was trained/evaluated.
- Relation-aware/multiplex partitioning has not been implemented.
- Metadata quality perturbation (names-only, description dropout/noise) has not been tested.
- Answer-generation with Ollama is pending. `gpt-oss:20b` returned an empty final response with `done_reason=length` on the first test question under several attempted response modes/token limits; no predictions or answer scores were saved. Do not cite this as a model outcome.
- An implementation equivalence check compared optimized bounded top-K retrieval with prior retrieval on validation: 6,930 method-question rankings, 0 mismatches.

---

## 9. Separate GTSQA study: fixed graph-inspection budgets

This is a structural partition study on GTSQA + OGB WikiKG2, not BRINK. It asks whether a partition-aware traversal retrieves more annotated evidence under a hard per-question edge-inspection cap. All methods use the same seed entities, graph, evidence, and traversal. Partitioning affects priority/search locality. The official GTSQA test was frozen before its one-time evaluation.

### 9.1 Pilot and validation

**128-question training pilot** compared unpartitioned BFS, Leiden resolution candidates, and METIS. This was a development screen only. At 1% (161,092 inspections), recall / complete evidence / mean ms were:

| Method | Recall / complete / latency |
|---|---|
| BFS | .9246 / .8359 / 83.76 |
| Leiden 2× density | .9432 / .8906 / 151.06 |
| METIS-128 | .9520 / .9141 / 158.53 |
| METIS-256 | .9617 / .9297 / 153.16 |

The initial 128 questions were excluded from a disjoint 512-question training validation sample. At the primary 0.1% budget (16,109 triples), recall / complete / latency were: BFS `.8297/.6777/6.52ms`; Leiden 2× `.8537/.7227/11.85ms`; METIS-128 `.8763/.7637/11.92ms`; METIS-256 `.8813/.7793/12.16ms`. METIS-256 led, but its recall difference versus METIS-128 was only +.0050, CI [−.0079,+.0180]. These development samples were used to select follow-up settings and are not confirmatory results.

**Constrained structural refinement:** begin from Leiden CPM at 2× weighted density; split communities above the METIS-256 observed maximum 10,061 using METIS; merge small communities into adjacent communities by maximum aggregate edge weight subject to max 10,061; minimum 2,442; stable tie breaking. Target split sizes 1,221, 2,442, 4,884 were screened on train validation. This is a project-defined heuristic, not a standard named algorithm. It uses no semantic embedding features.

On the 512-question validation sample at 0.1%, BFS recall .8297, raw Leiden .8537, METIS-256 .8813, and refined Leiden target 1,221 `.8829` (complete .7852; 13.00 ms). Refined-vs-BFS recall gain +.0532 [+.0346,+.0718], vs raw Leiden +.0291 [+.0157,+.0432], vs METIS-256 +.0016 [−.0121,+.0151]. Validation selected target 1,221 and then froze the test configuration.

Partition sizes before test: raw Leiden 138,447 communities, median size 1, max 500,656; METIS-256 has 256 parts, median 9,641, max 10,061; refined Leiden has 379 communities, median 8,606, max 10,060. Refined Leiden weighted cut = 10,361,204, higher than METIS-256 cut = 9,875,502. Refinement improves balance versus raw Leiden but does not match METIS’s exact balance. The broader five-point GTSQA CPM resolution sweep (0.125×–2× weighted density) had median community size 1 at every setting and maximum size 500,656–2,186,527; resolution tuning alone did not yield balanced raw Leiden communities, motivating METIS and the constrained refinement follow-up.

### 9.2 Frozen GTSQA test

Of 1,622 test questions, 1,121 had complete support and form the primary set; 1,591 had at least one supported fact and form secondary available-evidence set. Budgets are 1,611, 16,109, and 161,092 inspected triples (0.01%, 0.1%, 1%). 20,000 paired bootstrap resamples, seed 20261022.

Primary full-support results, cells are macro recall / complete evidence / mean latency:

| Method | 0.01% / 1,611 | 0.1% / 16,109 | 1% / 161,092 |
|---|---|---|---|
| Unpartitioned BFS | .6334 / .3372 / .54 ms | .7602 / .4969 / 6.29 ms | .8699 / .6940 / 75.95 ms |
| Leiden CPM 2× density | .6723 / .4130 / 1.10 ms | .8018 / .5843 / 12.21 ms | .8988 / .7618 / 141.71 ms |
| METIS-256 | .6912 / .4460 / 1.11 ms | .8268 / .6334 / 12.50 ms | **.9291 / .8216 / 147.18 ms** |
| Refined Leiden | **.6985 / .4585 / 1.11 ms** | **.8299 / .6378 / 12.53 ms** | .9220 / .7984 / 146.89 ms |

At primary 0.1% budget, refined Leiden versus BFS gained +.0697 recall (95% paired CI [+.0567,+.0828]) and +.1409 complete-evidence rate ([+.1151,+.1677]). Versus raw Leiden, recall gain +.0281 ([+.0189,+.0375]). Versus METIS-256, +.0031 ([−.0070,+.0133]), unresolved. At 1%, METIS-256 is numerically higher; refined minus METIS recall −.0070 ([−.0151,+.0011]), unresolved.

Secondary available-evidence recall (1,591 questions):

| Method | 0.01% | 0.1% | 1% |
|---|---:|---:|---:|
| BFS | .6058 | .7366 | .8438 |
| Leiden CPM 2× | .6375 | .7721 | .8712 |
| METIS-256 | .6572 | .7978 | **.9044** |
| Refined Leiden | **.6617** | **.8023** | .8983 |

Refined Leiden’s latency at the primary budget is about twice BFS (12.53 ms vs 6.29 ms). The evidence gain is at equal edge-inspection budget, not a latency reduction. Latencies are single-pass measurements and need repeated controlled runs before systems-performance claims.

---

## 10. Additional audit and experiments without positive/current results

### 10.1 Compatibility audit

The local full Wikidata5m CSV and benchmark graph are not identical. Local triples have 20,598,520 unique directed triples. BRINK complete graph has 20,510,107; BRINK incomplete has 20,478,006. Local full triples contain 32,067 withheld BRINK facts. GTSQA and KGQAGen evidence overlap with local Wikidata5m is poor: GTSQA test has only 795/1,622 `answer_subgraph` examples with full evidence in local CSV; KGQAGen-10k test has 59/1,074 with a nonempty proof fully present. GTSQA’s primary study therefore uses its compatible OGB WikiKG2 training graph and explicit support-filtered denominators.

### 10.2 Rejected or negative retrieval ideas

- **Re-embed after query finetuning:** the reported score declined; query encoder targets were original centroids. Rebuild both entity centroids and train targets together if revisiting.
- **Oracle topic entity:** reported lower score than linker-format query in the v2 pipeline; exploratory and lacks full saved artifacts.
- **Dense ensemble of base/fine-tuned MiniLM:** validation chose fine-tuned alone.
- **Numeric MLP reranker:** best dev .8941, below .9019 first stage; no test run.
- **Fusion alternatives:** Borda/RRF/z-score did not beat validation min-max weighted sum in the v2 run.
- **Single-community baseline:** perfect Hits is trivial because every answer is returned; invalid as useful retrieval performance.
- **BRINK local full graph:** forbidden for incomplete-KG claims because it contains almost all withheld target facts.
- **Initial full-convergence structural Leiden run:** took ~20,829.73 seconds at first validation resolution and had no paired semantic result under same run budget; retain as runtime pilot, exclude from primary comparison.
- **One-hour repeated retrieval attempt:** interrupted before writing any report/score; contributes no result.
- **GPT-OSS answer run:** empty response on first question under attempted configurations; no predictions/answer metrics; not a result.

### 10.3 Screening-only angles

- **BRINK by annotated path length:** exploratory post hoc test subgroup. For 1-triple paths (n=1,111), Recall@100: full graph .9586, structural .9307, semantic .9298. For 2-triple paths (n=1,984), about .474 for all. For 3-triple paths (n=53), .3333 for each. Semantic-minus-structural differences are near zero; do not make strong subgroup claims.
- **Boundary edge recovery:** cut facts explain only ~4–5% of missed evidence. Border-edge replication is a follow-up hypothesis, not a demonstrated improvement.
- **Query-adaptive routing:** no learned router exists; gold path length is unavailable to a real query-time router.
- **Multiplex/relation-aware graph partitioning:** proposed only; not implemented.
- **Metadata robustness:** no controlled names-only/description-dropout/noise tests.
- **Latency amortization:** partitioned GTSQA traversal is slower per query in measured implementation; no positive partition-construction break-even claim.

---

## 11. Experimental evidence hierarchy and claims

| Result | Status | Safe interpretation |
|---|---|---|
| Original cleaned `w` H@5 .5845 | Reported/evaluable historical result | Best of original 10-way partition benchmark; weighting/objective/build confounds remain |
| Original raw-to-clean gains | Descriptive | Cleanup is strongly associated with higher retrieval; phase ablation/seeds needed for causal generality |
| Official-test MPNet H@5 .9033 | Exploratory test result; prior test use | High observed answer-partition retrieval on original index; not QA accuracy or unbiased final estimate |
| Grouped internal fine-tuned MiniLM H@5 .7804 | Controlled internal evaluation, one seed | Fine-tuning improves on zero-shot on unseen rule groups on fixed original index |
| Rebuilt partition grid H@5 .712–.720 FT MiniLM | Controlled across weight variants, one partition seed | Current weight variants are close; no established winner |
| BRINK evidence retrieval full vs partitions | Frozen held-out test, paired CIs | Full graph retains ~1 pp more annotated evidence; semantic vs structural unresolved |
| GTSQA refined Leiden vs BFS/METIS | Frozen held-out test, strict graph-edge budget | Partitioned retrieval improves recall over BFS; refined Leiden unresolved vs METIS; slower latency |
| Any generated BRINK answer quality | Not available | No generator output was successfully scored |
| Generalization to other KG/domain | Not established | Requires compatible benchmark and fresh design |

### Claims not supported

- “The 90% system answers 90% of questions correctly.” It only retrieves an answer-containing partition in top five in that exploratory setup.
- “Semantic partitioning improves evidence retrieval.” Held-out BRINK semantic and structural results are near-identical.
- “Leiden refinement is better than METIS.” GTSQA paired intervals include zero at the main budget; METIS is numerically better at 1%.
- “Partitions reduce latency.” In GTSQA the current traversal is slower than BFS; BRINK one-pass local-index timings are exploratory only.
- “High modularity gives better KG-RAG retrieval.” Original raw partitions contradict this empirically in this setup.
- “The original `w` weighting beats `w²` under controlled conditions.” Historical builds/settings differ; the controlled grid finds only small unresolved differences.
- “Answers/evidence are generated faithfully.” No scored LLM answer generation exists.

---

## 12. Next work ranked by research value

1. **Preserve evidence and answer retrieval as separate outcomes.** Use current controlled 12-condition matrix as descriptive partition/encoder analysis; report BRINK evidence study separately.
2. **Repeat seeds.** At least 3–5 query-encoder training seeds for MiniLM/MPNet and multiple Leiden seeds for partition variants; report seed variability separately from question bootstrap uncertainty.
3. **Confirm partition conclusions on a fresh untouched split or another compatible dataset.** The official BRINK test has been inspected in multiple earlier experiments. The grouped internal test is better controlled but is still drawn from train and has only 29 held-out rule clusters.
4. **Matched granularity/resolution sweep.** Compare exponent variants at matched community counts or report full size/quality-response curves; include one common Leiden objective and cleanup. Consider cleanup phase ablations.
5. **Query encoder follow-up on a new grouped split.** Keep all answer partitions as positives, compare dense-only/BM25-only/fixed fusion and wider dev-only fusion sweep; save matched per-question scores for paired tests.
6. **Evidence-aware reranking or boundary retrieval.** Only after candidate recall and size budget are measured on dev. Numeric MLP is a negative result. A text-aware cross-encoder or relation/path feature reranker is untested.
7. **End-to-end BRINK QA.** Select a standard instruction model that returns answers reliably; freeze its exact tag, prompt, decoding, evidence input, and answer budget. Score official BRINK answer metrics over all test questions and report evidence retrieval separately. Do not select on test.
8. **Repeat latency measurements.** Randomize system order, warm consistently, report median/mean/p95 across repeated passes, CPU/GPU status, memory, and include linking/index/generation separately. Current timing data are retrieval-component one-pass only.
9. **Cross-dataset generalization.** GTSQA and BRINK must remain separately analyzed. Establish source graph/benchmark compatibility before using other QA datasets.

---

## 13. Artifact map and reproduction entry points

Paths below are relative to the workspace unless explicitly marked. Original project assets outside this writable workspace may be read-only.

### Main BRINK methodology and historical results

- `RESEARCH_DOCUMENTATION.md` — original 2026-10-21 end-to-end partition study, dataset/method/results; historical and partly superseded for fair weight comparison.
- `PAPER_METRICS_AND_ANALYSIS.md` — metric meanings, result reconciliation, original encoder experiments, grouped evaluation summaries, paper checklist, controlled benchmark supplement.
- `/home/pesu-rf/Documents/Capstone_Team165/partition/v2/V2_EXPERIMENT_LOG.md` — v2 model history; includes handoff provenance caveats, MPNet, fusion/reranker diagnostics.
- `/home/pesu-rf/Documents/Capstone_Team165/partition/v2/MPNET_ADAPTER_RESULTS.md` — MPNet details.

### Grouped query-encoder experiment (fixed original index)

- `paper_eval/grouped_internal_split.json` — question IDs and rule grouping.
- `paper_eval/run_grouped_encoder_eval.py` — query training/evaluation on original `w_cosine` index.
- `paper_eval/analyze_grouped_results.py` — paired bootstrap/subgroup analysis.
- `paper_eval/results/grouped_encoder_results.json` — aggregate values and config.
- `paper_eval/results/grouped_encoder_analysis.md` and `.json` — full H@K, intervals, paired comparisons, subgroup analysis.
- `paper_eval/results/<model>/{dev,test}_per_question.json` — complete per-question rankings and labels.
- `paper_eval/models/minilm_finetuned/`, `paper_eval/models/mpnet_adapter/` — trained checkpoints/adapters.

### Controlled partition grid (latest answer-partition benchmark)

- `paper_eval/rebuild_controlled_partitions.py` — shared-source weighting/Leiden/cleanup rebuild.
- `paper_eval/benchmark_models_partitions.py` — all 3 encoders × 4 variants; one temporary index at a time.
- `paper_eval/analyze_controlled_partitions.py` — 10,000 paired rule-cluster bootstrap comparisons.
- `paper_eval/controlled/CONTROLLED_PARTITION_BENCHMARK.md` — detailed focused report.
- `paper_eval/controlled/build_manifest.json` — common build settings and partition sizes.
- `paper_eval/controlled/model_partition_results.json` — all models, partitions, dev/test H@K/MRR/size metrics and fusion choices.
- `paper_eval/controlled/paired_cluster_bootstrap.json` — paired intervals.
- `paper_eval/controlled/predictions/<partition>/<model>/{dev,test}.json` — per-question top-100 rankings.
- `paper_eval/controlled/{raw,clean}_*_membership.pkl` — exact partition assignments.
- `paper_eval/controlled/index_*/` — compact centroids, partition IDs, index metadata; temporary large BM25/entity maps were deleted after scoring.

### BRINK evidence retrieval study

- `partition_experiment/README.md`, `RESEARCH_PROTOCOL.md`, `STATUS.md` — source-backed method, leakage controls, frozen protocol, status.
- `partition_experiment/SYSTEM_BENCHMARK_RESULTS.md` — held-out full graph vs structural/semantic test comparison.
- `partition_experiment/ANGLE1_RECALL_SEARCH_BUDGET.md` — returned-fact K curve and query-work interpretation.
- `artifacts/evaluation/brink_val_resolution_selection.json` — validation-selected resolution.
- `artifacts/evaluation/brink_{val,test}_system_benchmark_selected-resolution-v1/system_benchmark_report.json` — aggregate results and provenance.
- Corresponding `per_question_system_benchmark.jsonl` and `run_manifest.json` preserve question-level results/hashes.
- `artifacts/runs/brink-val-cpm-grid-*/` and `brink-val-cpm-grid-*-gpu-2iter_full/` — partition experiments/configs.
- `artifacts/evaluation/brink_test_annotation_support_audit.json` — support count audit.
- `scripts/generate_ollama_answers.py` — resumable answer runner; currently no successful prediction results.

### GTSQA/OGB study

- `partition_research/RESEARCH_PLAN.md`, `EVALUATION_PROTOCOL.md`, `BENCHMARK_ASSESSMENT.md` — plan, benchmark suitability, frozen method definitions.
- `partition_research/TRAIN_PILOT_REPORT.md` — 128-question pilot.
- `partition_research/TRAIN_VALIDATION_REPORT.md` — disjoint 512-question validation.
- `partition_research/TRAIN_REFINEMENT_REPORT.md` — constrained Leiden refinement and train-only tuning.
- `partition_research/TEST_RESULTS_REPORT.md` — frozen GTSQA test results.
- `partition_research/artifacts/frozen_test_config.json`, `gtsqa_test_frozen.summary.json`, `gtsqa_test_frozen.paired.json`, `gtsqa_test_frozen.integrity.json`, `gtsqa_test_frozen.jsonl` — frozen config, metrics, paired intervals, integrity, per-query data.
- `partition_research/scripts/run_gtsqa_test_frozen.py`, `refine_leiden_partition.py`, `analyze_gtsqa_test_results.py` — implementations.

### Cross-dataset compatibility and research screening

- `benchmark_compatibility_audit.md` / `.json` — graph and benchmark ID/evidence overlap audit.
- `FOUR_FOLLOWUP_ANGLES_SCREENING.md` — boundary, routing, relation-aware, latency-amortization status.
- `SEVEN_ANGLE_REVIEW.md` — synthesis of BRINK and GTSQA research angles.
- `partition_research/artifacts/` — partition summaries, test/validation/bootstrap records.

### Reproduction commands (from workspace root)

```bash
# Original-index grouped encoder fit/evaluation (uses available GPU environment)
python paper_eval/run_grouped_encoder_eval.py

# Recreate controlled partitions; then evaluate all model × partition cells
python paper_eval/rebuild_controlled_partitions.py
python paper_eval/benchmark_models_partitions.py
python paper_eval/analyze_controlled_partitions.py

# Current GTSQA and BRINK evidence reports are already frozen in artifacts;
# use the protocol and run manifests before attempting any re-run.
```

The controlled rebuild took about 20.7 minutes on the recorded machine; building and scoring all 12 indexes took about 10–12 minutes. Query encoder checkpoints and the original 7.4 GB embedding file are required. Do not overwrite benchmark/source assets. The full result output uses several hundred MB; ensure adequate disk space before re-running.

---

## 14. Instructions for a new reader or LLM

Treat this document as the orientation and experiment index, then consult the machine-readable artifact for any number used in a paper. Keep the four study tracks in §1 separate. For each reported result, state dataset/split, denominator, query construction, index/partition version, fusion/model, metric definition, and whether the test set was previously inspected. Do not elevate handoff-only results over saved artifacts. Do not describe answer-partition Hits@K as QA accuracy or evidence retrieval. Do not claim semantic Leiden is better on the available held-out evidence study. Clearly mark single-seed and one-pass timing limitations. When in doubt, prefer the per-question JSON/JSONL and run manifest over prose summaries.
