# main_graph_data.py

import warnings
warnings.filterwarnings("ignore")

import torch

from data.graph_data import EDGES
from retrieval.query_to_subgraph import extract_subgraph
from utils.graph_utils import build_graph, graph_to_string
from models.rgcn import RGCN

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def main():
    query = "famous physicists and their achievements"

    # 1. Query → relevant subgraph
    subgraph = extract_subgraph(query, EDGES, top_k=3)

    # 2. Build graph (with reverse edges + self-loops)
    nodes, relations, edge_index, edge_type = build_graph(
        subgraph,
        device,
        add_reverse=True,
        add_self_loops=True
    )

    # IMPORTANT: +1 relation for self-loops
    num_relations = len(relations) + 1

    # 3. R-GCN (dim is controlled HERE)
    model = RGCN(
        num_nodes=len(nodes),
        num_relations=num_relations,
        dim=64      # 👈 CHANGE THIS (32 / 64 / 128)
    ).to(device)

    with torch.no_grad():
        _ = model(edge_index, edge_type)

    # 4. Output
    print("\nFINAL OUTPUT:\n")
    print(graph_to_string(subgraph))

if __name__ == "__main__":
    main()
