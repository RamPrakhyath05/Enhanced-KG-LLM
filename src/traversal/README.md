# Partition-Bounded Path Scoring for KG-RAG

This directory is an additive integration layer. It does not alter the existing partitioning, embedding, BM25, or partition-retrieval files.

## Why T10-style path scoring

The traversal experiments found that T9 and T10 tied for the highest exact accepted-path accuracy (95%). T10 is used here because the RAG objective values complete, coherent supporting paths over a raw node list. Its original 20-query benchmark is not a guarantee of universal superiority; evaluate it on this repository's own held-out queries before making a production claim.

Smaller evidence is not automatically better. The correct objective is the smallest evidence set that still contains a complete, relevant, and auditable answer path. This module therefore returns up to three complete paths, not an arbitrary top-k node list.

## Integration contract

1. Use the existing `PartitionRetriever.search(query, top_k=3)`.
2. Resolve one or more seed entity IDs from the query or application entity linker.
3. Pass the partition-hit objects to `PartitionRAGTraversalAdapter`.
4. The adapter unions `entity_ids`, calls the path scorer, and builds the evidence bundle.
5. Send `bundle.context` to the LLM and retain `bundle.paths` and `bundle.citations` for provenance and UI/API consumers.

```python
from src.retrieval.partition_retriever import PartitionRetriever
from src.traversal import PartitionRAGTraversalAdapter, PathScoringTraverser

partition_hits = partition_retriever.search(question, top_k=3, client=neo4j_client)
traverser = PathScoringTraverser(neo4j_client)
adapter = PartitionRAGTraversalAdapter(traverser)
response = adapter.retrieve(
    query=question,
    seed_entity_ids=[resolved_seed_entity_id],
    partition_hits=partition_hits,
)
bundle = response.bundle

llm_context = bundle.context          # compact, citation-labelled text
provenance = bundle.as_dict()          # paths, IDs, scores, and citations
```

## Return format

Do not give the LLM only Python node objects. The recommended output is an `EvidenceBundle`:

- `context`: concise, data-delimited text with labelled KG triples for the prompt;
- `candidate_answer_ids`: terminal entities from the selected paths;
- `paths`: typed complete evidence paths and scores;
- `citations`: source/relation/target IDs for verification and display;
- `traversal_metadata`: scope size, candidate edges examined, and selected path count.

The context builder asks the LLM to use only the supplied KG evidence and to state insufficiency rather than invent a relation. The structured fields remain available for auditing and citations.

## Scoring

For each candidate edge in the retrieved partition scope:

\[
S_{edge}=0.35\,\operatorname{norm}(S_{dense})+0.35\,\operatorname{norm}(S_{local\ BM25})+0.20\,S_{relation}
\]

The path score accumulates edge scores and adds a novelty reward for a new relation while subtracting a hop penalty:

\[
S_{path}=\sum(S_{edge}+0.10\,\mathbb{1}[relation\ is\ new]-0.05)
\]

Dense similarity uses `all-MiniLM-L6-v2`. Local BM25 is computed only over the already bounded candidate edge set because the current repository has partition-level, not global edge-level, BM25 statistics. This preserves the existing retrieval architecture and avoids a global graph scan.

## Limits and next steps

- The layer requires seed entity IDs. It intentionally does not guess entity links; the application should pass an entity-linking result.
- Neo4j traversal is strictly constrained to `allowed_entity_ids`, normally the union of the top three retrieved partitions.
- Add application-level evaluation with held-out questions before treating the T10 defaults as final production parameters.
