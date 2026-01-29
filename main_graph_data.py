# main.py

import warnings
warnings.filterwarnings("ignore")

import torch

from data.graph_data import EDGES
from retrieval.query_to_subgraph import extract_subgraph
from utils.graph_utils import build_graph, graph_to_string
from models.rgcn import RGCN

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def main():
    query = "scientist who won a physics award"

    # 1. Query → subgraph
    subgraph = extract_subgraph(query, EDGES)

    # 2. Build graph tensors
    nodes, relations, edge_index, edge_type = build_graph(subgraph, device)

    # 3. Run R-GCN
    model = RGCN(
        num_nodes=len(nodes),
        num_relations=len(relations)
    ).to(device)

    with torch.no_grad():
        _ = model(edge_index, edge_type)

    # 4. Output as STRING
    output = graph_to_string(subgraph)

    print("\nFINAL OUTPUT:\n")
    print(output)

if __name__ == "__main__":
    main()
