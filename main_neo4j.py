import warnings
warnings.filterwarnings("ignore")

import torch

from data.neo4j_loader import Neo4jLoader
from retrieval.query_to_subgraph import extract_subgraph
from utils.graph_utils import build_graph, graph_to_string
from models.rgcn import RGCN

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def main():
    query = "scientist who won a physics award"

    # 1. Load graph from Neo4j
    loader = Neo4jLoader(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )

    edges = loader.fetch_edges(limit=200)
    loader.close()

    # 2. Query → relevant subgraph
    subgraph = extract_subgraph(query, edges)

    # 3. Build tensors
    nodes, relations, edge_index, edge_type = build_graph(subgraph, device)

    # 4. GNN traversal
    model = RGCN(
        num_nodes=len(nodes),
        num_relations=len(relations)
    ).to(device)

    with torch.no_grad():
        _ = model(edge_index, edge_type)

    # 5. Output STRING
    output = graph_to_string(subgraph)

    print("\nFINAL OUTPUT:\n")
    print(output)

if __name__ == "__main__":
    main()
