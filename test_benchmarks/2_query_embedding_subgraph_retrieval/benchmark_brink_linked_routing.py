#!/usr/bin/env python3
"""Measure whether real linker expansions improve BRINK partition routing.

Joins the saved validation outputs of benchmark_brink_string_queries.py and
benchmark_brink_hybrid_entity_linker.py, appends the predicted entity name and
aliases to each string query, and re-runs the same partition router/index.
"""
from __future__ import annotations
import argparse, hashlib, json, pickle, platform, subprocess, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from metrics.score_metrics import paired_bootstrap, summarize
from src.query_subgraph.subgraph_retriever import SubgraphRetriever, tokenize

def read_jsonl(path):
    return {str(r["question_id"]): r for r in (json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip())}

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

def minmax(x):
    return (x-x.min(axis=1,keepdims=True))/(x.max(axis=1,keepdims=True)-x.min(axis=1,keepdims=True)+1e-10)

def summarize_rankings(records, mode, method_names):
    out={}
    for method in method_names:
        h={}; hard={}; rr={}
        for k in (1,3,5,10,50):
            any_v=[]; hard_v=[]
            for r in records:
                rank=r["rankings"][mode][method]
                any_v.append(float(bool(set(rank[:k])&set(r["gold_partition_ids"]))))
                hp=r["hard_partition_id"]
                hard_v.append(float(hp is not None and hp in rank[:k]))
            h[f"any_hits@{k}"]=float(np.mean(any_v)); hard[f"hard_hits@{k}"]=float(np.mean(hard_v))
        for r in records:
            rank=r["rankings"][mode][method]
            pos=next((i for i,pid in enumerate(rank[:100],1) if pid in set(r["gold_partition_ids"])),None)
            rr.setdefault("mrr@100_values",[]).append(1/pos if pos else 0.0)
        out[method]={**h,**hard,"mrr@100":float(np.mean(rr["mrr@100_values"]))}
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--string-run",required=True,help="brink_val_string_queries.jsonl")
    p.add_argument("--linker-run",required=True,help="brink_val_hybrid_entity_linker.jsonl")
    p.add_argument("--linker-index",default="/home/pesu-rf/Documents/Capstone_Team165/partition/subg/entity_linker_index.pkl")
    p.add_argument("--linker-weight",default="0.70",help="selected lexical weight from linker validation sweep")
    p.add_argument("--index-dir",default=str(ROOT/"data/subg"))
    p.add_argument("--out",required=True)
    p.add_argument("--model",default="all-MiniLM-L6-v2",help="SentenceTransformer checkpoint; must produce vectors matching the frozen centroid space")
    p.add_argument("--dense-weights",default="0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1")
    p.add_argument("--device",choices=["cpu","cuda"],default=None)
    a=p.parse_args()
    string_rows=read_jsonl(a.string_run); link_rows=read_jsonl(a.linker_run)
    ids=sorted(set(string_rows)&set(link_rows),key=int)
    if not ids: raise SystemExit("No shared questions between routing and linker outputs")
    with open(a.linker_index,"rb") as f: link_index=pickle.load(f)
    id2expand=link_index["id2expand"]
    queries={"string":[],"linked":[]}; records=[]
    for qid in ids:
        s=string_rows[qid]; l=link_rows[qid]
        linklist=l["candidate_rankings"].get(f"{float(a.linker_weight):.2f}",[])
        pred_id=linklist[0][0] if linklist else None
        expansion=str(id2expand.get(pred_id,"")) if pred_id else ""
        base=str(s["string_question"])
        linked=(base+" "+expansion).strip() if expansion else base
        queries["string"].append(base); queries["linked"].append(linked)
        records.append({"question_id":qid,"raw_question":s["raw_question"],"string_question":base,
                        "linked_query":linked,"predicted_topic_id":pred_id,
                        "predicted_topic_name":str(link_index["id2name"].get(pred_id,"")) if pred_id else None,
                        "gold_partition_ids":s["gold_partition_ids"],"hard_partition_id":s["hard_partition_id"],
                        "link_hit_at_1":bool(pred_id and pred_id==l["gold_topic_id"])})
    retr=SubgraphRetriever(subg_dir=a.index_dir,model_name=a.model,device=a.device); retr.load()
    # Batch each query condition for repeatable inference timing; score with same centroids/BM25 docs.
    timings=[]; all_scores={}
    for mode in ("string","linked"):
        t0=time.perf_counter()
        qvecs=retr.model.encode(queries[mode],normalize_embeddings=True,batch_size=128,show_progress_bar=True)
        dense=(qvecs@retr.embeddings.T).astype(np.float32)
        bm25=np.asarray([retr.bm25.get_scores(tokenize(q)) for q in queries[mode]],dtype=np.float32)
        timings.append((time.perf_counter()-t0)*1000/len(ids))
        all_scores[mode]=(dense,bm25)
    weights=[float(x) for x in a.dense_weights.split(",")]
    pids=[int(x) for x in retr.partition_ids]
    method_names=["dense","bm25","rrf"]+[f"hybrid_{w:.2f}" for w in weights]
    for mode in ("string","linked"):
        dense,bm25=all_scores[mode]
        scores={"dense":dense,"bm25":bm25}
        dn,bn=minmax(dense),minmax(bm25)
        for w in weights: scores[f"hybrid_{w:.2f}"]=(w*dn+(1-w)*bn).astype(np.float32)
        rrf=[]
        for i in range(len(ids)):
            dr=np.argsort(-dense[i],kind="stable"); br=np.argsort(-bm25[i],kind="stable")
            rd=np.empty(len(pids),np.int32); rb=np.empty(len(pids),np.int32)
            rd[dr]=np.arange(1,len(pids)+1); rb[br]=np.arange(1,len(pids)+1)
            rrf.append(1/(60+rd)+1/(60+rb))
        scores["rrf"]=np.asarray(rrf,dtype=np.float32)
        for i,r in enumerate(records):
            r.setdefault("rankings",{})[mode]={m:[pids[int(j)] for j in np.argsort(-v[i],kind="stable")[:100]] for m,v in scores.items()}
    metrics={mode:summarize_rankings(records,mode,method_names) for mode in ("string","linked")}
    chosen=max((f"hybrid_{w:.2f}" for w in weights),key=lambda m:metrics["linked"][m]["any_hits@5"])
    paired={}
    for method in ("dense","bm25","rrf","hybrid_0.50",chosen):
        if method not in method_names: continue
        x=[float(bool(set(r["rankings"]["string"][method][:5])&set(r["gold_partition_ids"]))) for r in records]
        y=[float(bool(set(r["rankings"]["linked"][method][:5])&set(r["gold_partition_ids"]))) for r in records]
        paired[method]={"string_to_linked_delta_h5":float(np.mean(np.array(y)-np.array(x))),"paired_question_bootstrap_95ci":paired_bootstrap(x,y)}
    report={"benchmark":"BRINK-Wikidata5m linked partition routing","split":"val",
            "protocol":"QID-free canonical-name query augmented with the top-1 expansion from the saved hybrid entity-linker run; query includes the predicted entity name and aliases. Same 3,625-partition index; query encoder is specified by --model (base or frozen fine-tuned checkpoint).",
            "linker_weight":float(a.linker_weight),"linker_hit_at_1_on_joined_questions":float(np.mean([r["link_hit_at_1"] for r in records])),
            "n_questions":len(records),"partition_count":len(pids),"selected_router_dense_weight_on_linked_validation":chosen,
            "query_encoder":a.model,
            "router_metrics":metrics,"paired_string_vs_linked_h5":paired,
            "mean_router_latency_ms_per_query_excludes_loading":dict(zip(("string","linked"),timings)),
            "device":retr.device,"gpu":torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "provenance":{"string_run_sha256":sha256(a.string_run),"linker_run_sha256":sha256(a.linker_run),"linker_index_sha256":sha256(a.linker_index)}}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    with out.with_suffix(".jsonl").open("w",encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(json.dumps(report,indent=2)); print(f"Wrote {out} and {out.with_suffix('.jsonl')}")

if __name__=="__main__": main()
