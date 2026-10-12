# ============================================================
# step1_embeddings.py
# Generates entity embeddings using SentenceTransformer.
# Embedding text = Entity_Name + Entity_Aliases + Entity_Description snippet.
# Input:  entities.csv
# Output: entity_embeddings.npy, entity_id_to_row.pkl
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ENTITIES_PATH = "./entities.csv"
EMBEDDINGS_OUT_PATH = "./entity_embeddings.npy"
ID_MAP_OUT_PATH = "./entity_id_to_row.pkl"

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 512
DESC_SNIPPET_LEN = 300
MAX_ALIASES = 8
CHECKPOINT_DIR = "./emb_ckpt"


def log(msg):
    print(f"[step1] {msg}", flush=True)


def format_aliases(raw):
    if not raw:
        return ""
    picked = [a.strip() for a in raw.split("|") if a.strip()][:MAX_ALIASES]
    if not picked:
        return ""
    return " Also known as: " + ", ".join(picked)


def main():
    log("Loading entities.csv...")
    entities = pd.read_csv(ENTITIES_PATH)
    log(f"Entities: {len(entities):,}")

    # Build embedding text: "Name Also known as: a, b, c. Description snippet"
    names = entities["Entity_Name"].fillna("").astype(str)
    aliases_raw = entities["Entity_Aliases"].fillna("").astype(str)
    descs = entities["Entity_Description"].fillna("").astype(str).str.slice(0, DESC_SNIPPET_LEN)

    log("Formatting aliases...")
    alias_col = aliases_raw.map(format_aliases)

    log("Building embedding texts...")
    embed_text = [
        n + a + (". " + d if d else "")
        for n, a, d in zip(names, alias_col, descs)
    ]
    log(f"Embedding text sample: {embed_text[0][:160]}...")

    # Check for existing checkpoints
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    existing_batches = set()
    for f in os.listdir(CHECKPOINT_DIR):
        if f.startswith("emb_batch_") and f.endswith(".npy"):
            existing_batches.add(int(f.replace("emb_batch_", "").replace(".npy", "")))
    log(f"Found {len(existing_batches)} existing checkpoint batches")

    # Load model
    log(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cuda")

    # Process in batches with checkpointing
    all_embeddings = []
    total_batches = (len(embed_text) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in tqdm(range(total_batches), desc="Encoding"):
        if batch_idx in existing_batches:
            batch_emb = np.load(f"{CHECKPOINT_DIR}/emb_batch_{batch_idx:05d}.npy")
            all_embeddings.append(batch_emb)
            continue

        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(embed_text))
        batch_texts = embed_text[start:end]

        batch_emb = model.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=BATCH_SIZE,
        )

        np.save(f"{CHECKPOINT_DIR}/emb_batch_{batch_idx:05d}.npy", batch_emb)
        all_embeddings.append(batch_emb)

    # Concatenate all embeddings
    log("Concatenating embeddings...")
    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    log(f"Embeddings shape: {embeddings.shape}")

    # Build entity ID -> row mapping
    entity_ids = entities["Entity_ID"].tolist()
    entity_id_to_row = {eid: i for i, eid in enumerate(entity_ids)}

    # Save
    np.save(EMBEDDINGS_OUT_PATH, embeddings)
    with open(ID_MAP_OUT_PATH, "wb") as f:
        pickle.dump(entity_id_to_row, f)

    log(f"Saved embeddings to {EMBEDDINGS_OUT_PATH}")
    log(f"Saved ID map to {ID_MAP_OUT_PATH}")
    log("DONE.")


if __name__ == "__main__":
    main()
