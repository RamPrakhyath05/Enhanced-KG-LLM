from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import igraph as ig
import leidenalg
import numpy as np


class LeidenPartitioner:
    """
    Modular Leiden Partitioner supporting both:
    1. Standard RBConfigurationVertexPartition for topological subgraphs.
    2. CPMVertexPartition (Constant Potts Model) for weighted semantic graphs with post-processing.
    """

    def __init__(self, resolution: float = 1.0, n_iterations: int = 2, seed: int = 42, method: str = "rb"):
        self.resolution = resolution
        self.n_iterations = n_iterations
        self.seed = seed
        self.method = method

    def partition(
        self,
        entities: Dict[str, dict],
        edges: List[Tuple[str, str, str]],
    ) -> Tuple[Dict[str, int], Dict[int, dict], float]:
        entity_ids = list(entities.keys())
        id_to_idx = {eid: idx for idx, eid in enumerate(entity_ids)}

        g = ig.Graph(directed=False)
        g.add_vertices(len(entity_ids))
        g.vs["name"] = entity_ids

        ig_edges = []
        for src, tgt, _ in edges:
            if src in id_to_idx and tgt in id_to_idx:
                ig_edges.append((id_to_idx[src], id_to_idx[tgt]))

        g.add_edges(ig_edges)
        g.simplify(multiple=True, loops=True)

        if self.method.lower() == "cpm":
            partition_cls = leidenalg.CPMVertexPartition
        else:
            partition_cls = leidenalg.RBConfigurationVertexPartition

        partition = leidenalg.find_partition(
            g,
            partition_cls,
            resolution_parameter=self.resolution,
            n_iterations=self.n_iterations,
            seed=self.seed,
        )

        partition_map = {}
        communities = defaultdict(list)

        for vertex_idx, community_id in enumerate(partition.membership):
            eid = entity_ids[vertex_idx]
            partition_map[eid] = int(community_id)
            communities[int(community_id)].append(eid)

        degrees = g.degree()
        community_metadata = {}

        for comm_id, members in communities.items():
            members_sorted = sorted(members, key=lambda eid: degrees[id_to_idx[eid]], reverse=True)
            top_entities = members_sorted[:10]

            entity_summaries = []
            for eid in top_entities:
                e_info = entities.get(eid, {})
                name = e_info.get("name") or ""
                desc = e_info.get("description") or ""
                if desc:
                    entity_summaries.append(f"{name}: {desc}")
                elif name:
                    entity_summaries.append(name)

            summary_text = (
                f"Community {comm_id} (Size: {len(members)} entities)\n"
                f"Core Entities: {', '.join([entities.get(e, {}).get('name', e) for e in top_entities])}\n"
                f"Descriptions:\n" + "\n".join(entity_summaries[:5])
            )

            community_metadata[comm_id] = {
                "community_id": comm_id,
                "size": len(members),
                "entity_ids": members,
                "top_entities": top_entities,
                "summary_text": summary_text,
            }

        return partition_map, community_metadata, partition.modularity

    def split_oversized_communities(
        self,
        g: ig.Graph,
        membership: list,
        max_size: int = 5000,
        sub_resolution: float = 1e-4,
    ) -> list:
        """Recursively splits communities exceeding max_size."""
        sizes = defaultdict(int)
        for c in membership:
            sizes[c] += 1

        oversized = {c for c, s in sizes.items() if s > max_size}
        if not oversized:
            return membership

        new_membership = list(membership)
        next_comm_id = max(membership) + 1

        for comm in oversized:
            node_indices = [i for i, c in enumerate(membership) if c == comm]
            subg = g.subgraph(node_indices)
            sub_part = leidenalg.find_partition(
                subg,
                leidenalg.CPMVertexPartition,
                weights="weight" if "weight" in subg.es.attributes() else None,
                resolution_parameter=sub_resolution,
                seed=self.seed,
            )
            for local_idx, sub_comm in enumerate(sub_part.membership):
                orig_idx = node_indices[local_idx]
                new_membership[orig_idx] = next_comm_id + sub_comm
            next_comm_id += len(set(sub_part.membership))

        return new_membership
