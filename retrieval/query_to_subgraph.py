import torch
from sentence_transformers import SentenceTransformer

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

embedder = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device=device
)

def edge_to_text(edge):
    src, rel, dst, props = edge
    prop_text = " ".join(f"{k} {v}" for k, v in props.items())
    return f"{src} {rel} {dst} {prop_text}"

def extract_subgraph(query, edges, top_k=3):
    query_emb = embedder.encode(query, convert_to_tensor=True)

    scored = []
    for edge in edges:
        text = edge_to_text(edge)
        emb = embedder.encode(text, convert_to_tensor=True)
        score = torch.cosine_similarity(query_emb, emb, dim=0)
        scored.append((score.item(), edge))

    scored.sort(reverse=True)
    return [edge for _, edge in scored[:top_k]]
