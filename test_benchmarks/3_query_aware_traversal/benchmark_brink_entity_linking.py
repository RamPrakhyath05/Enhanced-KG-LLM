#!/usr/bin/env python3
"""Offline lexical entity-linking benchmark for BRINK validation questions.

BRINK question strings contain QIDs, so the benchmark uses q_entity metadata
only to construct a readable input by substituting the canonical entity label.
The linker itself sees only that natural-language string and a local
name/alias index. Gold IDs are used solely to score ranked candidates.

This does not call the production Neo4j-backed EntitySeedResolver. Its mention
extraction and stop-word policy are reused, with exact canonical-name/alias
lookup as a reproducible offline candidate generator.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, json, platform, re, sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from metrics.score_metrics import entity_link_metrics, paired_bootstrap, summarize
from src.traversal.seed_resolver import EntitySeedResolver, MENTION_STOP_WORDS

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
QID_RE = re.compile(r"Q\d+")

def parse_list(s):
    try:
        v = ast.literal_eval(s or "[]")
        return list(v) if isinstance(v, (list, tuple, set)) else [v]
    except (ValueError, SyntaxError):
        return [s] if s else []

def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return {str(r["id"]): r for r in csv.DictReader(f, delimiter="\t")}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def normalize_mention(text):
    """Normalize names like the production resolver's mention extraction."""
    words = [w.casefold() for w in TOKEN_RE.findall(str(text)) if w.casefold() not in MENTION_STOP_WORDS and len(w) >= 2]
    if not words or len(words) > 4:
        return ""
    return " ".join(words)

def build_lexicon(path):
    """Map normalized labels/aliases to entity IDs; discard >4-token phrases."""
    index = defaultdict(set)
    n_entities = n_aliases = 0
    entity_names = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            eid = (row.get("Entity_ID") or "").strip()
            if not eid:
                continue
            n_entities += 1
            name = row.get("Entity_Name") or ""
            entity_names[eid] = name
            key = normalize_mention(name)
            if key:
                index[key].add(eid)
            for alias in (row.get("Entity_Aliases") or "").split("|"):
                key = normalize_mention(alias)
                if key:
                    index[key].add(eid)
                    n_aliases += 1
            if i % 500000 == 0:
                print(f"Indexed {i:,} entity rows; {len(index):,} normalized names/aliases", flush=True)
    return index, n_entities, n_aliases, entity_names

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", required=True)
    p.add_argument("--hard-answers", required=True)
    p.add_argument("--entity-csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    comp, hard = read_tsv(a.questions), read_tsv(a.hard_answers)
    ids = sorted(set(comp) & set(hard), key=lambda x: int(x))
    if a.limit:
        ids = ids[:a.limit]
    print(f"Building local name/alias lexicon for {len(ids):,} questions", flush=True)
    t0 = time.perf_counter()
    lexicon, n_entities, n_aliases, entity_names = build_lexicon(a.entity_csv)
    index_seconds = time.perf_counter() - t0
    # Reuse the same mention extraction rules as EntitySeedResolver, without DB calls.
    mentioner = EntitySeedResolver(db_client=None)
    rows, latencies, extraction_hits, candidate_hits = [], [], [], []
    for qid in ids:
        raw = str(comp[qid].get("question") or "")
        topics = [str(x) for x in parse_list(hard[qid].get("q_entity", ""))]
        if len(topics) != 1:
            continue
        gold_id = topics[0]
        # Lookup is only to create a no-QID benchmark input; it is not exposed to linker.
        topic_name = entity_names.get(gold_id, "")
        if not topic_name or gold_id not in raw:
            continue
        query = raw.replace(gold_id, topic_name)
        if QID_RE.search(query):
            continue
        start = time.perf_counter()
        mentions = mentioner.extract_mentions(query)
        # First occurrence of a mention retains its position/order from the production extractor.
        candidates = {}
        for mention_rank, mention in enumerate(mentions):
            key = normalize_mention(mention)
            if not key:
                continue
            for eid in lexicon.get(key, ()):
                # Favor specific phrases; then production extraction order; stable ID tie-break.
                score = min(0.3, 0.1 * len(key.split()))
                old = candidates.get(eid)
                if old is None:
                    candidates[eid] = (score, mention_rank, mention_rank)
                else:
                    best_score = max(score, old[0])
                    earliest = mention_rank if score > old[0] or (score == old[0] and mention_rank < old[1]) else old[1]
                    latest = mention_rank if score > old[0] or (score == old[0] and mention_rank > old[2]) else old[2]
                    candidates[eid] = (best_score, earliest, latest)
        ranked_earliest = sorted(candidates, key=lambda eid: (-candidates[eid][0], candidates[eid][1], eid))
        # Validation ablation: keep phrase specificity equal, but prefer later query mentions.
        ranked_latest = sorted(candidates, key=lambda eid: (-candidates[eid][0], -candidates[eid][2], eid))
        latencies.append((time.perf_counter() - start) * 1000)
        gold_in_mentions = any(normalize_mention(m) == normalize_mention(topic_name) for m in mentions)
        extraction_hits.append(float(gold_in_mentions))
        candidate_hits.append(float(gold_id in candidates))
        rows.append({"question_id": qid, "question": query, "gold_topic_id": gold_id,
                     "gold_topic_name": topic_name, "extracted_mentions": mentions,
                     "ranked_entity_ids_earliest": ranked_earliest[:20],
                     "ranked_entity_names_earliest": [entity_names.get(e, "") for e in ranked_earliest[:20]],
                     "ranked_entity_ids_latest": ranked_latest[:20],
                     "ranked_entity_names_latest": [entity_names.get(e, "") for e in ranked_latest[:20]],
                     "gold_rank_earliest": ranked_earliest.index(gold_id) + 1 if gold_id in candidates else None,
                     "gold_rank_latest": ranked_latest.index(gold_id) + 1 if gold_id in candidates else None,
                     "candidate_count": len(candidates)})
        if len(rows) % 500 == 0 or len(rows) == len(ids):
            print(f"Linked {len(rows):,}/{len(ids):,} questions", flush=True)

    if not rows:
        raise SystemExit("No questions were linkable after label construction.")
    variants = {}
    for mode in ("earliest", "latest"):
        metrics = {}
        for k in (1, 3, 5, 10):
            vals = [entity_link_metrics(r[f"ranked_entity_ids_{mode}"], [r["gold_topic_id"]], k=k) for r in rows]
            metrics[f"@{k}"] = {
                "recall": sum(v[f"link_recall_at_{k}"] for v in vals) / len(vals),
                "precision": sum(v[f"link_precision_at_{k}"] for v in vals) / len(vals),
                "hit_rate": sum(float(r["gold_topic_id"] in r[f"ranked_entity_ids_{mode}"][:k]) for r in rows) / len(rows),
            }
        ranks = [r[f"gold_rank_{mode}"] or 0 for r in rows]
        variants[mode] = {"metrics_at_k": metrics,
                          "hit_at_1": sum(float(x == 1) for x in ranks) / len(ranks),
                          "mrr": sum(1 / x if x else 0 for x in ranks) / len(ranks)}
    pair_a = [float(bool(r["gold_rank_earliest"] and r["gold_rank_earliest"] <= 1)) for r in rows]
    pair_b = [float(bool(r["gold_rank_latest"] and r["gold_rank_latest"] <= 1)) for r in rows]
    report = {
        "benchmark": "BRINK-Wikidata5m entity linking", "split": a.split,
        "protocol": "QID-free query constructed by replacing BRINK q_entity with its canonical Entity_Name; linker receives only query text and local canonical-name/alias lexicon. Gold topic ID used only for scoring.",
        "linker": "offline lexical candidate linker using EntitySeedResolver.extract_mentions + exact normalized name/alias lookup; no Neo4j, no partition boost, no answer information",
        "n_questions": len(rows), "topic_label_coverage": len(rows) / max(1, len(ids)),
        "mention_extraction_recall": sum(extraction_hits) / len(extraction_hits),
        "candidate_generation_recall": sum(candidate_hits) / len(candidate_hits),
        "ranking_variants": variants,
        "latest_vs_earliest_hit_at_1_paired_bootstrap": paired_bootstrap(pair_a, pair_b),
        "latency_ms_linker_only": summarize(latencies),
        "lexicon": {"entity_rows": n_entities, "alias_rows_indexed": n_aliases, "unique_normalized_keys": len(lexicon), "build_seconds": index_seconds},
        "provenance": {"python": platform.python_version(),
                       "input_sha256": {"questions": sha256(a.questions), "hard_answers": sha256(a.hard_answers), "entity_csv": sha256(a.entity_csv)}},
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with out.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out} and {out.with_suffix('.jsonl')}")

if __name__ == "__main__":
    main()
