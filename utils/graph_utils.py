import torch

def build_graph(edges, device):
    nodes = sorted({s for s,_,_,_ in edges} | {t for _,_,t,_ in edges})
    relations = sorted({r for _,r,_,_ in edges})

    node2id = {n: i for i, n in enumerate(nodes)}
    rel2id = {r: i for i, r in enumerate(relations)}

    edge_index = []
    edge_type = []

    for s, r, t, _ in edges:
        edge_index.append([node2id[s], node2id[t]])
        edge_type.append(rel2id[r])

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
