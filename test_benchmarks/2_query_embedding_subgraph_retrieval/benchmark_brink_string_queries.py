#!/usr/bin/env python3
"""BRINK validation routing with natural-language topic labels in place of QIDs.

This is a label-resolved query benchmark: BRINK's q_entity metadata selects the
topic ID and the local entity table supplies its canonical name. Gold answers
are used only for scoring. It does not measure an automatic entity linker.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, json, platform, re, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from metrics.score_metrics import paired_bootstrap, summarize
from src.query_subgraph.subgraph_retriever import SubgraphRetriever, tokenize

QID_RE = re.compile(r"Q\d+")

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
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def minmax(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-10)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", required=True, help="BRINK complete-answer TSV")
    p.add_argument("--hard-answers", required=True, help="BRINK hard-answer TSV")
    p.add_argument("--entity-csv", required=True, help="Entity_ID,Entity_Name table")
    p.add_argument("--index-dir", default=str(ROOT / "data/subg"))
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--weights", default="0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1", help="Dense weight; BM25 weight is 1-w")
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)
    p.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name or local path")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    comp, hard = read_tsv(a.questions), read_tsv(a.hard_answers)
    ids = sorted(set(comp) & set(hard), key=lambda x: int(x))
    if a.limit:
        ids = ids[:a.limit]
    topic_ids = {str(q) for i in ids for q in parse_list(hard[i].get("q_entity", ""))}
    names = {}
    with open(a.entity_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            eid = row.get("Entity_ID", "")
            if eid in topic_ids:
                names[eid] = str(row.get("Entity_Name") or "").strip()
                if len(names) == len(topic_ids):
                    break
    retr = SubgraphRetriever(subg_dir=a.index_dir, model_name=a.model, device=a.device)
    retr.load()
    pids = [int(x) for x in retr.partition_ids]
    pid_index = {pid: i for i, pid in enumerate(pids)}
    entity_to_pid = {str(e): int(pid) for pid, es in retr.entity_map.items() for e in es}
    weights = [float(w) for w in a.weights.split(",")]
    prepared = []
    for qid in ids:
        raw = str(comp[qid]["question"])
        topics = [str(x) for x in parse_list(hard[qid].get("q_entity", ""))]
        # Reject malformed rows; never infer topic IDs from answer labels.
        if len(topics) != 1 or not names.get(topics[0]):
            continue
        topic = topics[0]
        if topic not in raw:
            continue
        string_query = raw.replace(topic, names[topic])
        if QID_RE.search(string_query):
            continue
        answers = [str(x) for x in parse_list(comp[qid].get("answer", ""))]
        hard_answers = [str(x) for x in parse_list(hard[qid].get("answer", ""))]
        gold = sorted({entity_to_pid[x] for x in answers if x in entity_to_pid})
        hard_pid = entity_to_pid.get(hard_answers[0]) if hard_answers else None
        if not gold:
            continue

        prepared.append((qid, raw, string_query, topic, names[topic], gold, hard_pid))

    # Batch query encoding is important for a fair, practical CPU evaluation.
    raw_vecs = retr.model.encode([r[1] for r in prepared], normalize_embeddings=True, batch_size=128, show_progress_bar=True)
    string_vecs = retr.model.encode([r[2] for r in prepared], normalize_embeddings=True, batch_size=128, show_progress_bar=True)
    per_q, timings = [], []
    for j, (qid, raw, string_query, topic, topic_name, gold, hard_pid) in enumerate(prepared):
        t0 = time.perf_counter()
        # Same question encoder and original centroid/BM25 index for both forms.
        d_string = (retr.embeddings @ string_vecs[j]).astype(np.float32)
        b_string = np.asarray(retr.bm25.get_scores(tokenize(string_query)), dtype=np.float32)
        d_raw = (retr.embeddings @ raw_vecs[j]).astype(np.float32)
        b_raw = np.asarray(retr.bm25.get_scores(tokenize(raw)), dtype=np.float32)
        ranking = {}
        for mode, dense, bm25 in (("raw", d_raw, b_raw), ("string", d_string, b_string)):
            scores = {"dense": dense, "bm25": bm25}
            dn, bn = minmax(dense), minmax(bm25)
            for w in weights:
                scores[f"hybrid_{w:.2f}"] = w * dn + (1 - w) * bn
            dr = np.argsort(-dense, kind="stable")
            br = np.argsort(-bm25, kind="stable")
            rd, rb = np.empty(len(pids), np.int32), np.empty(len(pids), np.int32)
            rd[dr] = np.arange(1, len(pids) + 1)
            rb[br] = np.arange(1, len(pids) + 1)
            scores["rrf"] = 1 / (60 + rd) + 1 / (60 + rb)
            ranking[mode] = {m: [pids[int(x)] for x in np.argsort(-s, kind="stable")[:100]] for m, s in scores.items()}
        timings.append((time.perf_counter() - t0) * 1000)
        per_q.append({"question_id": qid, "raw_question": raw, "string_question": string_query,
                      "topic_id": topic, "topic_name": topic_name, "gold_partition_ids": gold,
                      "hard_partition_id": hard_pid, "rankings": ranking})
        if (j + 1) % 250 == 0 or j + 1 == len(prepared):
            print(f"Scored {j + 1}/{len(prepared)} question pairs", flush=True)

    if not per_q:
        raise SystemExit("No evaluable rows after topic-label mapping.")
    methods = list(per_q[0]["rankings"]["string"])
    summary = {}
    for mode in ("raw", "string"):
        summary[mode] = {}
        for method in methods:
            any5, hard5, reciprocal = [], [], []
            for row in per_q:
                rank = row["rankings"][mode][method]
                gold = set(row["gold_partition_ids"])
                any5.append(float(bool(gold.intersection(rank[:5]))))
                hard5.append(float(row["hard_partition_id"] is not None and row["hard_partition_id"] in rank[:5]))
                rr = next((1 / i for i, pid in enumerate(rank[:100], 1) if pid in gold), 0.0)
                reciprocal.append(rr)
            summary[mode][method] = {"any_hits@5": float(np.mean(any5)), "hard_hits@5": float(np.mean(hard5)),
                                     "mrr@100": float(np.mean(reciprocal)), "n": len(any5)}

    chosen = max((m for m in methods if m.startswith("hybrid_")), key=lambda m: summary["string"][m]["any_hits@5"])
    deltas = {}
    for method in ("dense", "bm25", "rrf", "hybrid_0.50"):
        if method in methods:
            x = [float(bool(set(r["rankings"]["raw"][method][:5]) & set(r["gold_partition_ids"]))) for r in per_q]
            y = [float(bool(set(r["rankings"]["string"][method][:5]) & set(r["gold_partition_ids"]))) for r in per_q]
            deltas[method] = {"raw_to_string_h5_delta": float(np.mean(np.array(y) - np.array(x))),
                              "paired_question_bootstrap_95ci": paired_bootstrap(x, y)}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"benchmark": "BRINK-Wikidata5m", "split": a.split,
              "protocol": "Natural string query: replace BRINK q_entity QID with canonical Entity_Name; no answer IDs/names used in query. Label-resolved topic mapping, not automatic entity linking.",
              "query_example": {"raw": per_q[0]["raw_question"], "string": per_q[0]["string_question"]},
              "n_questions": len(per_q), "topic_name_coverage": len(names) / max(1, len(topic_ids)),
              "all_queries_without_qids": all(not QID_RE.search(r["string_question"]) for r in per_q),
              "index_dir": str(Path(a.index_dir).resolve()), "partition_count": len(pids),
              "encoder": a.model, "device": retr.device,
              "latency_ms_query_pair": summarize(timings), "selected_hybrid_on_string_queries": chosen,
              "metrics": summary, "paired_raw_vs_string": deltas,
              "provenance": {"python": platform.python_version(), "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                             "input_sha256": {"questions": sha256(a.questions), "hard_answers": sha256(a.hard_answers), "entity_csv": sha256(a.entity_csv)},
                             "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None}}
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    raw_out = out.with_suffix(".jsonl")
    with raw_out.open("w", encoding="utf-8") as f:
        for row in per_q:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out} and {raw_out}")

if __name__ == "__main__":
    main()
