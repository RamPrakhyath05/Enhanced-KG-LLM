"""Dependency-light, reusable metrics for Enhanced-KG-LLM experiments."""
from __future__ import annotations
import json, math, re, statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

_TOKEN = re.compile(r"\w+", re.UNICODE)


def normalize_answer(text: str) -> str:
    return " ".join(t for t in _TOKEN.findall(str(text).lower()) if t not in {"a", "an", "the"})


def answer_em_f1(prediction: str, gold_answers: Sequence[str]) -> tuple[float, float]:
    """Return best normalized exact match and token F1 over alternative gold strings."""
    pred = normalize_answer(prediction)
    p = pred.split()
    best_em = best_f1 = 0.0
    for gold in gold_answers:
        g = normalize_answer(gold)
        best_em = max(best_em, float(pred == g))
        gt = g.split()
        common = sum((__import__('collections').Counter(p) & __import__('collections').Counter(gt)).values())
        if p and gt:
            precision, recall = common / len(p), common / len(gt)
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        else:
            f1 = float(not p and not gt)
        best_f1 = max(best_f1, f1)
    return best_em, best_f1



def prediction_set_metrics(predictions: Iterable[str], gold_answers: Iterable[str], hard_answer: str | None = None) -> dict[str, float]:
    """BRINK-style answer-set metrics after answer normalization."""
    pred = {normalize_answer(x) for x in predictions if normalize_answer(x)}
    gold = {normalize_answer(x) for x in gold_answers if normalize_answer(x)}
    overlap = pred & gold
    p = len(overlap) / len(pred) if pred else 0.0
    r = len(overlap) / len(gold) if gold else 0.0
    f1 = 2 * len(overlap) / (len(pred) + len(gold)) if pred or gold else 0.0
    hit_any = float(bool(overlap))
    hit_hard = float(normalize_answer(hard_answer or "") in pred and bool(hard_answer))
    return {"hits_any": hit_any, "precision": p, "recall": r, "f1": f1, "hits_hard": hit_hard}


def ranking_metrics(ranked_ids: Sequence[Any], gold_ids: Iterable[Any], ks=(1, 3, 5, 10)) -> dict[str, float]:
    gold = set(gold_ids)
    ranks = [i + 1 for i, value in enumerate(ranked_ids) if value in gold]
    first = min(ranks, default=0)
    out = {f"hits@{k}": float(first > 0 and first <= k) for k in ks}
    out["mrr"] = 1.0 / first if first else 0.0
    out["gold_count"] = float(len(gold))
    return out


def entity_link_metrics(ranked_entities: Sequence[Any], gold_entities: Iterable[Any], k=3) -> dict[str, float]:
    gold = set(gold_entities); pred = list(ranked_entities[:k]); predset = set(pred)
    ranks = [i + 1 for i, item in enumerate(ranked_entities) if item in gold]
    first = min(ranks, default=0)
    return {f"link_recall_at_{k}": len(predset & gold) / len(gold) if gold else 0.0,
            f"link_precision_at_{k}": len(predset & gold) / len(predset) if predset else 0.0,
            "link_hit_at_1": float(bool(ranked_entities and ranked_entities[0] in gold)),
            "link_mrr": 1.0 / first if first else 0.0, "link_gold_count": float(len(gold))}

def set_recall(retrieved: Iterable[Any], gold: Iterable[Any]) -> float | None:
    g = set(gold)
    return len(g & set(retrieved)) / len(g) if g else None


def path_coverage(retrieved: Iterable[Any], gold_paths: Sequence[Sequence[Any]]) -> float | None:
    paths = [set(p) for p in gold_paths if p]
    if not paths:
        return None
    got = set(retrieved)
    return float(any(path <= got for path in paths))


def citation_scores(citations: Iterable[Any], supported: Iterable[Any]) -> dict[str, float | None]:
    c, s = set(citations), set(supported)
    tp = len(c & s)
    return {"citation_precision": tp / len(c) if c else None,
            "citation_recall": tp / len(s) if s else None,
            "citation_f1": 2 * tp / (len(c) + len(s)) if c or s else None}


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(float(x) for x in values)
    pos = (len(v) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def summarize(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None, "p99": None}
    return {"n": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
            "p95": percentile(values, .95), "p99": percentile(values, .99)}


def paired_bootstrap(a: Sequence[float], b: Sequence[float], *, samples=10000, seed=42) -> dict[str, float]:
    """Percentile CI for paired mean difference (b-a), resampling examples."""
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be non-empty and equal length")
    import random
    rng = random.Random(seed)
    diffs = [float(y) - float(x) for x, y in zip(a, b)]
    means = []
    for _ in range(samples):
        means.append(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs))
    return {"mean_difference_b_minus_a": statistics.fmean(diffs),
            "ci95_low": percentile(means, .025), "ci95_high": percentile(means, .975),
            "n": len(diffs), "bootstrap_samples": samples, "seed": seed}


def paired_cluster_bootstrap(a: Sequence[float], b: Sequence[float], clusters: Sequence[Any], *, samples=10000, seed=42) -> dict[str, float]:
    """Paired percentile CI resampling clusters while retaining all rows per draw."""
    if len(a) != len(b) or len(a) != len(clusters) or not a:
        raise ValueError("paired values and cluster IDs must be non-empty and equal length")
    import random
    grouped: dict[Any, list[float]] = {}
    for x, y, cluster in zip(a, b, clusters):
        grouped.setdefault(cluster, []).append(float(y) - float(x))
    keys = list(grouped); rng = random.Random(seed); estimates = []
    for _ in range(samples):
        sampled = [keys[rng.randrange(len(keys))] for _ in keys]
        diffs = [d for key in sampled for d in grouped[key]]
        estimates.append(sum(diffs) / len(diffs))
    diffs = [float(y) - float(x) for x, y in zip(a, b)]
    return {"mean_difference_b_minus_a": statistics.fmean(diffs),
            "ci95_low": percentile(estimates, .025), "ci95_high": percentile(estimates, .975),
            "n_questions": len(diffs), "n_clusters": len(keys), "bootstrap_samples": samples, "seed": seed}

def score_record(row: dict[str, Any]) -> dict[str, Any]:
    """Score one generic record; absent metric inputs are omitted."""
    out: dict[str, Any] = {"question_id": row.get("question_id", row.get("id"))}
    if "resolved_seed_entity_ids" in row:
        out["entity_linking"] = entity_link_metrics(row["resolved_seed_entity_ids"], row.get("gold_seed_entity_ids", []))
    if "ranked_ids" in row:
        out["ranking"] = ranking_metrics(row["ranked_ids"], row.get("gold_ranked_ids", []))
    if "retrieved_evidence" in row:
        recall = set_recall(row["retrieved_evidence"], row.get("gold_evidence", []))
        coverage = path_coverage(row["retrieved_evidence"], row.get("gold_evidence_paths", []))
        if recall is not None: out["evidence_recall"] = recall
        if coverage is not None: out["complete_path_coverage"] = coverage
    if "prediction" in row and "gold_answers" in row:
        out["answer_em"], out["answer_f1"] = answer_em_f1(row["prediction"], row["gold_answers"])
    if "citations" in row:
        out["citations"] = citation_scores(row["citations"], row.get("supported_citations", []))
    return out


def aggregate_scored(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Macro-average metric fields, report counts and latency/resource summaries."""
    keys = ("answer_em", "answer_f1", "evidence_recall", "complete_path_coverage")
    result: dict[str, Any] = {"n_records": len(records)}
    for key in keys:
        vals = [r[key] for r in records if r.get(key) is not None]
        if vals: result[key] = {"mean": statistics.fmean(vals), "n": len(vals)}
    for name in ("link_recall_at_3", "link_precision_at_3", "link_hit_at_1", "link_mrr"):
        vals = [r["entity_linking"][name] for r in records if name in r.get("entity_linking", {})]
        if vals: result[name] = {"mean": statistics.fmean(vals), "n": len(vals)}
    for group, names in (("ranking", ("hits@1", "hits@3", "hits@5", "hits@10", "mrr")),
                         ("citations", ("citation_precision", "citation_recall", "citation_f1"))):
        for name in names:
            vals = [r[group][name] for r in records if r.get(group, {}).get(name) is not None]
            if vals: result[name] = {"mean": statistics.fmean(vals), "n": len(vals)}
    for key in ("latency_ms", "edges_examined", "adjacency_entries_fetched", "candidate_edges_scored", "scoring_operations", "context_tokens", "peak_memory_mb"):
        vals = [float(r[key]) for r in records if isinstance(r.get(key), (int, float))]
        if vals: result[key] = summarize(vals)
    result["failure_rate"] = sum(bool(r.get("error")) for r in records) / len(records) if records else None
    result["skipped_rate"] = sum(bool(r.get("skipped")) for r in records) / len(records) if records else None
    return result


def score_jsonl(input_path: str, output_path: str) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(input_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    scored = [score_record(r) for r in rows]
    report = aggregate_scored([{**row, **score} for row, score in zip(rows, scored)])
    report["input"] = str(Path(input_path).resolve())
    Path(output_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
