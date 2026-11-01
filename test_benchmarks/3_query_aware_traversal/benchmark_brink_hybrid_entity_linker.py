#!/usr/bin/env python3
"""Evaluate the project's hybrid lexical+dense entity linker on BRINK validation.

BRINK QIDs are replaced with canonical labels only to construct readable query
strings. The linker receives only text. Gold topic IDs are used only for scoring.
The actual linker implementation is imported from partition/subg/entity_linker.py.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PARTITION_ROOT = ROOT.parent / "partition"
LINKER_DIR = PARTITION_ROOT / "subg"
sys.path.insert(0, str(LINKER_DIR))
sys.path.insert(0, str(ROOT))
import entity_linker as linker_module
from entity_linker import EntityLinker
from metrics.score_metrics import paired_bootstrap, summarize

def parse_list(s):
    try:
        x = ast.literal_eval(s or "[]")
        return list(x) if isinstance(x, (list, tuple, set)) else [x]
    except (ValueError, SyntaxError):
        return [s] if s else []

def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return {str(r["id"]): r for r in csv.DictReader(f, delimiter="\t")}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def get_candidates(linker, query, dense_top_k=50):
    qvec = linker.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    lex, qlen = linker._lexical_candidates(query)
    d_idx, d_val = linker._dense_candidates(qvec, dense_top_k)
    return lex, d_idx, d_val, qlen

def rank_candidates(linker, lex, d_idx, d_val, w_lex):
    if len(d_val) > 1:
        dmin, drange = float(d_val.min()), float(d_val.max()) - float(d_val.min()) + 1e-9
    else:
        dmin, drange = 0.0, 1.0
    max_lex = max(lex.values()) if lex else 0
    scores = {int(r): (1 - w_lex) * ((float(v) - dmin) / drange) for r, v in zip(d_idx, d_val)}
    for row, n_words in lex.items():
        scores[int(row)] = scores.get(int(row), 0.0) + w_lex * (n_words / max(max_lex, 1))
    ranked_rows = sorted(scores, key=lambda r: (-scores[r], r))
    return [(str(linker.row2id(r)), float(scores[r])) for r in ranked_rows]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", required=True)
    p.add_argument("--hard-answers", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--lexical-weights", default="0.5,0.6,0.7,0.8,0.9")
    p.add_argument("--dense-top-k", type=int, default=50)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    a = p.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Full dense linking over 4.8M entities is not run on CPU; use the offline lexical benchmark instead.")
    comp, hard = read_tsv(a.questions), read_tsv(a.hard_answers)
    ids = sorted(set(comp) & set(hard), key=int)
    if a.limit:
        ids = ids[:a.limit]
    t_load = time.perf_counter()
    linker = EntityLinker(device=a.device, use_gpu_dense=(a.device == "cuda"))
    startup_s = time.perf_counter() - t_load
    prepared = []
    for qid in ids:
        q = str(comp[qid].get("question") or "")
        topics = [str(x) for x in parse_list(hard[qid].get("q_entity", ""))]
        if len(topics) != 1 or topics[0] not in linker.id2name:
            continue
        eid = topics[0]
        name = str(linker.id2name.get(eid) or "").strip()
        if not name or eid not in q:
            continue
        string_query = q.replace(eid, name)
        prepared.append((qid, string_query, eid, name))
    if not prepared:
        raise SystemExit("No rows with a mapped topic label.")

    weights = [float(x) for x in a.lexical_weights.split(",")]
    rows, elapsed_by_weight = [], {w: [] for w in weights}
    for i, (qid, query, gold, name) in enumerate(prepared, 1):
        t0 = time.perf_counter()
        lex, d_idx, d_val, qlen = get_candidates(linker, query, a.dense_top_k)
        per_weight = {}
        for w in weights:
            ranked = rank_candidates(linker, lex, d_idx, d_val, w)
            per_weight[f"{w:.2f}"] = ranked
        elapsed = (time.perf_counter() - t0) * 1000
        for w in weights:
            elapsed_by_weight[w].append(elapsed)
        rows.append({"question_id": qid, "question": query, "gold_topic_id": gold,
                     "gold_topic_name": name, "candidate_rankings": per_weight,
                     "candidate_count_by_weight": {w: len(ranked) for w, ranked in per_weight.items()}})
        if i % 100 == 0 or i == len(prepared):
            print(f"Linked {i}/{len(prepared)} questions", flush=True)

    results, hit_vectors = {}, {}
    for w in weights:
        key = f"{w:.2f}"
        hits, rr = {}, []
        for k in (1, 3, 5, 10):
            vec = []
            for row in rows:
                ids_ranked = [x[0] for x in row["candidate_rankings"][key]]
                vec.append(float(row["gold_topic_id"] in ids_ranked[:k]))
            hits[f"hit@{k}"] = float(np.mean(vec))
        for row in rows:
            ids_ranked = [x[0] for x in row["candidate_rankings"][key]]
            rank = next((i for i, eid in enumerate(ids_ranked, 1) if eid == row["gold_topic_id"]), None)
            rr.append(1 / rank if rank else 0.0)
        results[key] = {**hits, "mrr": float(np.mean(rr)), "latency_ms": summarize(elapsed_by_weight[w])}
        hit_vectors[key] = [float(row["gold_topic_id"] in [x[0] for x in row["candidate_rankings"][key][:5]]) for row in rows]
    selected = max(weights, key=lambda w: results[f"{w:.2f}"]["hit@5"])
    default = "0.70" if "0.70" in results else f"{weights[0]:.2f}"
    paired = paired_bootstrap(hit_vectors[default], hit_vectors[f"{selected:.2f}"]) if f"{selected:.2f}" != default else None
    report = {"benchmark": "BRINK-Wikidata5m hybrid entity linking", "split": a.split,
              "protocol": "BRINK q_entity used only to substitute canonical Entity_Name into question text; linker receives QID-free string; gold topic ID used only for scoring.",
              "implementation": "partition/subg/entity_linker.py; lexical exact name/alias span candidates + dense top-k search over entity embeddings; no graph partition boost, no answer information",
              "n_questions": len(rows), "topic_label_coverage": len(rows) / len(ids),
              "all_queries_without_qids": all(not __import__("re").search(r"Q\d+", r["question"]) for r in rows),
              "device": a.device, "startup_seconds": startup_s, "dense_top_k": a.dense_top_k,
              "default_lexical_weight": float(default), "selected_lexical_weight_on_validation": selected if a.split == "val" else None,
              "metrics_by_lexical_weight": results,
              "paired_h5_default_vs_selected": paired,
              "provenance": {"python": platform.python_version(), "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                             "questions_sha256": sha256(a.questions), "hard_answers_sha256": sha256(a.hard_answers),
                             "entity_table_sha256": sha256(linker_module.ENTITIES_CSV), "entity_embeddings_sha256": sha256(linker_module.EMB_NPY),
                             "linker_index_sha256": sha256(linker_module.INDEX_CACHE)}}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with out.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out} and {out.with_suffix('.jsonl')}")
    linker.release_gpu()

if __name__ == "__main__":
    main()
