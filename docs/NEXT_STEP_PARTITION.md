# Partition handoff for the next pipeline stage

## Recommended choice

Use the existing **cleaned `w_cosine` partition and its matching index** if the next step is to continue the current best observed retrieval pipeline (MPNet adapter, reported answer-partition H@5 = 0.9033). That encoder was trained and selected against the original `w_cosine` centroids, and the full `w_cosine` retrieval index is already built. Keeping these together avoids changing the candidate space under the trained encoder.

The 90.33% figure is an exploratory answer-partition retrieval result on the already-inspected BRINK test set; it is not generated-answer accuracy. It is the best existing complete system to carry into the next LLM/evidence-generation stage, while preserving the same retrieved partitions and recording evidence/answer-generation results separately.

The latest controlled rebuild gives a different answer depending on the objective: fine-tuned MiniLM + unweighted has the top any-answer H@5 (0.7196), while fine-tuned MiniLM + `w³` has the top hard-answer H@5 (0.6864) and MRR@10. The partition differences are small and their paired rule-group intervals include zero. Those rebuilt partitions are useful research candidates, but the prior MPNet adapter was not trained on their centroid spaces. Switching the deployed next-stage pipeline to one of them requires building its full index and reselecting/retraining the query encoder on the same partition. Do not combine the MPNet 0.9033 result with a rebuilt `w³` or unweighted index.

## Primary files: use as a matching set

### Partition membership and graph node order

- [Cleaned `w_cosine` membership](/home/pesu-rf/Documents/Capstone_Team165/partition/membership_final_v6.pkl)
- [Graph node ID order](/home/pesu-rf/Documents/Capstone_Team165/partition/graph_node_ids_w2.pkl)

The membership pickle is a Python list of **3,529,659** community labels. Entry `i` is the partition label for entry `i` in the graph node ID list. These two files must be used together. This is the **graph-node membership**; the prebuilt index additionally assigns the 1,283,832 hub-only/non-graph entities.

### Complete retrieval index for the selected partition

Use the existing directory [index_w_cosine](/home/pesu-rf/Documents/Capstone_Team165/partition/subg/index_w_cosine):

- `centroids.npy` — 3,625 normalized 384-dimensional community vectors.
- `partition_ids.pkl` — row-to-community-label mapping for `centroids.npy`.
- `entity_map.pkl` — community label → all assigned entity QIDs (includes non-graph entities).
- `bm25_docs.pkl` — tokenized entity names and aliases for each community.
- `meta.json` — confirms label `w_cosine`, source membership path, 3,529,659 graph nodes, 1,283,832 non-graph entities, 3,625 communities, size cap 8,000, and zero overflow.

The original vector space and model/checkpoint are under [partition/v2](/home/pesu-rf/Documents/Capstone_Team165/partition/v2); the MPNet adapter is in `finetuned_mpnet_adapter/`. Its projection maps MPNet’s 768-dimensional query representation to the existing 384-dimensional centroid space. The model, projection, centroids, BM25 document ordering, and `partition_ids.pkl` must stay aligned.

### Integrity checks

SHA-256 values at handoff:

| File | SHA-256 |
|---|---|
| `membership_final_v6.pkl` | `d46c607878b7514389d41b11d2b85e4f6dd3026b868c7052b07e2b98c2f0d547` |
| `graph_node_ids_w2.pkl` | `34aa902d88319a3a856174abd23e3f6516e92cc6ca7737b3eddfdeebdc0150bd` |
| `index_w_cosine/centroids.npy` | `d818a7c609221b78dca155db9a01cdeafa039282c72e02008e8295a7535b81e5` |
| `index_w_cosine/partition_ids.pkl` | `846b62753118f4d8e26cbc85a7ab5111b1a401683ce6820c042626c7ab239247` |

The index `meta.json` names `membership_final_v6.pkl` as its membership source. Do not substitute a controlled-rebuild membership file while continuing to load this existing index.

## How to use it in the next stage

### A. Retrieve the top communities

1. Prepare the question using the same preprocessing as the MPNet retrieval experiment: replace the topic QID in question text with the entity name and apply the same entity-linker augmentation. Do not put the gold topic entity into the query.
2. Load the MPNet adapter checkpoint and its `projection.pt`; encode the query into 384 dimensions.
3. Score that vector against `centroids.npy` with cosine similarity. Score the query against `bm25_docs.pkl` with BM25.
4. Apply the same per-query min-max normalization and weighted fusion used for the reported MPNet run: `score = 0.58 * dense_norm + 0.42 * bm25_norm`.
5. Sort by descending fused score. Translate score-matrix row positions through `partition_ids.pkl` to get community labels.
6. For each selected community label, use `entity_map.pkl[label]` to obtain all member entity IDs. Keep its fused score and rank as metadata for downstream evidence selection/generation.

Use the saved evaluation code and model paths as the source of truth for query-linker preprocessing and exact scoring implementation: `/home/pesu-rf/Documents/Capstone_Team165/partition/v2/partition_v2_mpnet_eval.py` and `partition_v2_mpnet_adapter.py`. Fusion weight 0.42 is validation-selected in that experiment.

### B. Form evidence for generation

Selecting a partition gives an entity set, not an answer or a complete reasoning proof. To make the context for the next stage:

1. Retrieve the facts induced by the selected entities from the **same graph snapshot used by the retrieval pipeline**. Preserve relation IDs/names and original triple direction.
2. Rank or budget those facts with a documented, deterministic rule. The earlier centroid+BM25 Hits@5 experiment did not itself define a complete LLM evidence-ranking/generation protocol.
3. Record the question, selected community IDs/scores, entity and fact counts, exact evidence facts sent to the model, prompt, model tag, decoding settings, raw output, and official BRINK answer metrics.
4. Report answer-partition retrieval separately from whether evidence contains an annotated alternative path and whether the generated answer is correct. Do not call one a substitute for the other.

The `entity_map.pkl` is useful for entity membership. If the next stage needs the induced facts, use a relation-preserving triple source/index with an explicit policy for hub relations. The centroid/BM25 index does not contain triples.

### C. Rebuild the matching index if needed

Usually no rebuild is needed; the complete `index_w_cosine` directory already exists. If it is missing, rebuild from the membership and common graph node order using the project index builder, with all arguments from the same original partition asset set:

```bash
python /home/pesu-rf/Documents/Capstone_Team165/partition/subg/build_partition_index.py \
  --membership /home/pesu-rf/Documents/Capstone_Team165/partition/membership_final_v6.pkl \
  --node-ids /home/pesu-rf/Documents/Capstone_Team165/partition/graph_node_ids_w2.pkl \
  --embeddings /home/pesu-rf/Documents/Capstone_Team165/partition/entity_embeddings.npy \
  --id-map /home/pesu-rf/Documents/Capstone_Team165/partition/entity_id_to_row.pkl \
  --entities /home/pesu-rf/Documents/Capstone_Team165/partition/entities.csv \
  --out-dir /home/pesu-rf/Documents/Capstone_Team165/partition/subg/index_w_cosine_rebuilt \
  --label w_cosine --size-cap 8000
```

Keep the `--out-dir` new so the existing retrieval index is not overwritten. Then verify the new centroids/order/index metadata and re-run retrieval validation before using it with a saved query checkpoint.

## Controlled alternatives (for experiments, not a drop-in replacement)

The common-source memberships are under [paper_eval/controlled](/home/pesu-rf/Documents/Capstone_Team165/new/paper_eval/controlled):

| Variant | Membership file | Compact index data | Internal grouped-split fine-tuned MiniLM results |
|---|---|---|---|
| `w` | `clean_w_membership.pkl` | `index_w/` | Any H@5 .7155; hard H@5 .6690 |
| `w²` | `clean_w2_membership.pkl` | `index_w2/` | Any H@5 .7122; hard H@5 .6647 |
| `w³` | `clean_w3_membership.pkl` | `index_w3/` | Any H@5 .7187; hard H@5 **.6864** |
| Unweighted | `clean_unweighted_membership.pkl` | `index_unweighted/` | Any H@5 **.7196**; hard H@5 .6757 |

The compact `index_*` folders contain centroids, partition IDs, and metadata only. Temporary `entity_map.pkl` and `bm25_docs.pkl` were deleted after benchmarking to conserve disk, so reconstruct the full index with `build_partition_index.py` before downstream use. The node order is still `/home/pesu-rf/Documents/Capstone_Team165/partition/graph_node_ids_w2.pkl`; the embeddings, entity ID map, and entity metadata are also under the original `partition` directory. For the controlled comparison’s encoder, use the `paper_eval/models` checkpoints and matching score protocol rather than assuming the original MPNet adapter is aligned to these centroid sets.

## Decision in one sentence

For the immediate next step with the best already-measured retrieval system, use **`membership_final_v6.pkl` + `graph_node_ids_w2.pkl` + the complete `subg/index_w_cosine/` index + the matching MPNet adapter**. Keep rebuilt `w³` and unweighted as controlled research alternatives, not replacements in the current MPNet pipeline.
