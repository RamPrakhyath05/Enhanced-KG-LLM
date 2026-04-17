import warnings
warnings.filterwarnings("ignore")

import torch

from data.neo4j_loader import Neo4jLoader
from retrieval.query_to_subgraph import extract_subgraph
from utils.graph_utils import build_graph, graph_to_string
from models.rgcn import RGCN

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _infer_query_intent(query):
    q = query.lower()
    if "list" in q or "what are" in q or "which are" in q:
        return "list"
    if "compare" in q or "difference" in q:
        return "compare"
    if "who" in q or "where" in q or "when" in q:
        return "fact"
    return "default"


QUERY_KNOBS = {
    "list": {
        "limit_per_term": 2500,
        "max_edges": 12000,
        "top_k": 180,
        "hops": 0,
        "max_per_node": 4,
        "max_total": 80,
    },
    "fact": {
        "limit_per_term": 1200,
        "max_edges": 5000,
        "top_k": 20,
        "hops": 2,
        "max_per_node": 8,
        "max_total": 180,
    },
    "compare": {
        "limit_per_term": 1500,
        "max_edges": 6500,
        "top_k": 30,
        "hops": 2,
        "max_per_node": 10,
        "max_total": 220,
    },
    "default": {
        "limit_per_term": 1200,
        "max_edges": 5000,
        "top_k": 20,
        "hops": 2,
        "max_per_node": 8,
        "max_total": 180,
    },
}


def main():
    query = "List all dog breeds"
    intent = _infer_query_intent(query)
    knobs = QUERY_KNOBS[intent]
    print(f"Intent: {intent} | Knobs: {knobs}")

    # 1. Load graph from Neo4j
    loader = Neo4jLoader(
        uri="neo4j://127.0.0.1:7687",
        user="neo4j",
        password="Capstone_Data"
    )

    # Targeted search: fetch edges related to query entities (up to 10,000)
    edges = loader.search_edges_by_query(
        query,
        limit_per_term=knobs["limit_per_term"],
        max_edges=knobs["max_edges"],
    )
    print(f"Targeted search found {len(edges)} edges")

    # Only add general edges if targeted search found very few results
    if len(edges) < 500:
        print("Few targeted results — adding general context edges...")
        general_edges = loader.fetch_edges(limit=5000)
        seen = set()
        for e in edges:
            seen.add((e[0], e[1], e[2]))
        for e in general_edges:
            key = (e[0], e[1], e[2])
            if key not in seen:
                seen.add(key)
                edges.append(e)

    loader.close()

    print(f"Total: {len(edges)} unique edges")

    # 2. Query → relevant subgraph
    subgraph = extract_subgraph(
        query,
        edges,
        top_k=knobs["top_k"],
        hops=knobs["hops"],
        max_per_node=knobs["max_per_node"],
        max_total=knobs["max_total"],
    )

    # 3. Build tensors (with reverse edges + self-loops)
    nodes, relations, edge_index, edge_type = build_graph(subgraph, device)

    # +1 relation for the self-loop relation added by build_graph
    num_relations = len(relations) + 1

    # 4. GNN traversal
    model = RGCN(
        num_nodes=len(nodes),
        num_relations=num_relations
    ).to(device)

    with torch.no_grad():
        _ = model(edge_index, edge_type)

    # 5. Output STRING
    output = graph_to_string(subgraph)

    print(f"\nFINAL OUTPUT ({len(subgraph)} edges):\n")
    print(output)

if __name__ == "__main__":
    main()
