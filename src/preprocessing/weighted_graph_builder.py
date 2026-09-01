import os
import pickle
import numpy as np
import pandas as pd
import igraph as ig
from tqdm import tqdm


class WeightedGraphBuilder:
    """
    Builds a weighted igraph from structural triples and continuous entity embeddings.
    Computes semantic cosine similarity for each edge, shifts negative similarities,
    and simplifies duplicate edges by summing weights.
    """

    HUB_RELATIONS = {
        "P31", "P17", "P27", "P131", "P19", "P20", "P21", "P106",
    }

    def __init__(self, chunk_size: int = 500_000):
        self.chunk_size = chunk_size

    def filter_hub_relations(self, triples_df: pd.DataFrame) -> pd.DataFrame:
        """Filters out high-degree non-topological hub relations."""
        structural = triples_df[~triples_df["Relation_ID"].isin(self.HUB_RELATIONS)].copy()
        return structural

    def build_from_embeddings(
        self,
        triples_df: pd.DataFrame,
        embeddings: np.ndarray,
        entity_id_to_row: dict,
    ) -> tuple[ig.Graph, list]:
        """
        Computes cosine similarities per edge, shifts to positive range, and constructs igraph.
        """
        all_nodes = pd.unique(pd.concat([triples_df["Entity_1_ID"], triples_df["Entity_2_ID"]]))
        node_to_idx = {node: i for i, node in enumerate(all_nodes)}

        src_emb_rows = triples_df["Entity_1_ID"].map(entity_id_to_row).to_numpy()
        tgt_emb_rows = triples_df["Entity_2_ID"].map(entity_id_to_row).to_numpy()

        src_emb_rows = np.nan_to_num(src_emb_rows.astype(float), nan=0).astype(np.int64)
        tgt_emb_rows = np.nan_to_num(tgt_emb_rows.astype(float), nan=0).astype(np.int64)

        n_edges = len(triples_df)
        weights = np.empty(n_edges, dtype=np.float32)

        for start in range(0, n_edges, self.chunk_size):
            end = min(start + self.chunk_size, n_edges)
            src_vecs = embeddings[src_emb_rows[start:end]]
            tgt_vecs = embeddings[tgt_emb_rows[start:end]]
            sims = np.sum(src_vecs * tgt_vecs, axis=1)
            weights[start:end] = sims

        min_w = weights.min()
        if min_w <= 0:
            shift = abs(min_w) + 1e-4
            weights = weights + shift

        src_idx = triples_df["Entity_1_ID"].map(node_to_idx).to_numpy()
        tgt_idx = triples_df["Entity_2_ID"].map(node_to_idx).to_numpy()
        edge_list = list(zip(src_idx.tolist(), tgt_idx.tolist()))

        g = ig.Graph(n=len(all_nodes), edges=edge_list, directed=False)
        g.es["weight"] = weights.tolist()
        g.simplify(combine_edges={"weight": "sum"})

        return g, list(all_nodes)
