# utils/graph_utils.py

import torch

def build_graph(edges, device, add_reverse=True, add_self_loops=True):
    # Collect nodes and relations
    nodes = sorted({s for s,_,_,_ in edges} | {t for _,_,t,_ in edges})
    relations = sorted({r for _,r,_,_ in edges})

    node2id = {n: i for i, n in enumerate(nodes)}
    rel2id = {r: i for i, r in enumerate(relations)}

    edge_index = []
    edge_type = []

    # --- forward + reverse edges ---
    for s, r, t, _ in edges:
        src = node2id[s]
        dst = node2id[t]
        rel = rel2id[r]

        # forward edge
        edge_index.append([src, dst])
        edge_type.append(rel)

        # reverse edge
        if add_reverse:
            edge_index.append([dst, src])
            edge_type.append(rel)

    # --- self-loops ---
    if add_self_loops:
        self_loop_rel = len(relations)  # new relation id
        for node in node2id.values():
            edge_index.append([node, node])
            edge_type.append(self_loop_rel)

    edge_index = torch.tensor(edge_index).t().to(device)
    edge_type = torch.tensor(edge_type).to(device)

    return nodes, relations, edge_index, edge_type


def graph_to_string(edges):
    lines = []
    for s, r, t, props in edges:
        if props:
            prop_str = ", ".join(f"{k}={v}" for k, v in props.items())
            lines.append(f"{s} -[{r} {{{prop_str}}}]-> {t}")
        else:
            lines.append(f"{s} -[{r}]-> {t}")
    return "\n".join(lines)
