#!/usr/bin/env python3
"""Evaluate this repository's current partition router on BRINK QID TSVs.

This benchmark assesses answer-containing partition retrieval, not evidence recovery
or answer generation. Test evaluation is guarded because the project's BRINK test
has already been used during model iteration.
"""
from __future__ import annotations
import argparse, ast, csv, json, os, sys, time, hashlib, platform, subprocess, torch
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from metrics.score_metrics import ranking_metrics, paired_bootstrap, paired_cluster_bootstrap, summarize
from src.query_subgraph.subgraph_retriever import SubgraphRetriever


def parse_list(value):
    if not value: return []
    try:
        x=ast.literal_eval(value)
        return x if isinstance(x,(list,tuple,set)) else [x]
    except (ValueError,SyntaxError): return [value]

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()
def read_tsv(path):
    with open(path,encoding='utf-8') as f: return {str(r['id']):r for r in csv.DictReader(f,delimiter='\t')}

def norm(x):
    x=np.asarray(x,dtype=np.float32)
    return (x-x.min(axis=1,keepdims=True))/(x.max(axis=1,keepdims=True)-x.min(axis=1,keepdims=True)+1e-10)

def metric_for(rows, method, k):
    hits=[]; hard=[]; rr=[]
    for r in rows:
        ranks=r['rankings'][method]
        gold=set(r['gold_partition_ids']); h=r['hard_partition_id']
        x=ranking_metrics(ranks,gold,ks=(k,)); hits.append(x[f'hits@{k}']); rr.append(x['mrr'])
        hard.append(float(h is not None and h in ranks[:k]))
    return {'any_hits':float(np.mean(hits)),'hard_hits':float(np.mean(hard)),
            'mrr@100':float(np.mean(rr)),'n':len(rows)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--questions',required=True,help='BRINK complete-answer TSV')
    p.add_argument('--hard-answers',required=True,help='matching BRINK hard-answer TSV')
    p.add_argument('--split',choices=['train','val','test'],default='val')
    p.add_argument('--index-dir',default=str(ROOT/'data/subg'))
    p.add_argument('--out-dir',default=str(ROOT/'metrics/results/raw'))
    p.add_argument('--top-k',type=int,default=100)
    p.add_argument('--weights',default='0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1',help='dense weights for hybrid; BM25 weight is 1-w')
    p.add_argument('--allow-exploratory-test',action='store_true')
    a=p.parse_args()
    if a.split=='test' and not a.allow_exploratory_test:
        p.error('BRINK test has prior experimental reuse; pass --allow-exploratory-test to label this run exploratory.')
    comp=read_tsv(a.questions); hard=read_tsv(a.hard_answers)
    ids=sorted(set(comp)&set(hard),key=lambda x:int(x))
    if not ids: raise SystemExit('No matching question IDs in complete and hard answer files.')
    retr=SubgraphRetriever(subg_dir=a.index_dir); retr.load()
    pids=[int(x) for x in retr.partition_ids]
    entity_to_partition={str(e):int(pid) for pid,entities in retr.entity_map.items() for e in entities}
    weights=[float(x) for x in a.weights.split(',')]
    rows=[]; latencies=[]
    for qid in ids:
        question=comp[qid]['question']
        t0=time.perf_counter()
        qv=retr.model.encode([question],normalize_embeddings=True)[0]
        dense=(retr.embeddings @ qv).reshape(1,-1)
        toks=retr.bm25_docs
        from src.query_subgraph.subgraph_retriever import tokenize
        bm25=np.asarray(retr.bm25.get_scores(tokenize(question)),dtype=np.float32).reshape(1,-1)
        dn,bn=norm(dense),norm(bm25)
        scored={'dense':dense[0], 'bm25':bm25[0]}
        for w in weights: scored[f'hybrid_dense_{w:.2f}']=(w*dn+(1-w)*bn)[0]
        # Rank fusion baseline (RRF, k=60); dense/BM25 score ties are stable by row index.
        dr=np.argsort(-dense[0],kind='stable'); br=np.argsort(-bm25[0],kind='stable')
        rankd=np.empty(len(pids),dtype=np.int32); rankb=np.empty(len(pids),dtype=np.int32)
        rankd[dr]=np.arange(1,len(pids)+1); rankb[br]=np.arange(1,len(pids)+1)
        scored['rrf']=(1/(60+rankd)+1/(60+rankb))
        rankings={name:[pids[int(i)] for i in np.argsort(-scores,kind='stable')[:a.top_k]] for name,scores in scored.items()}
        sizes_by_pid={pid:len(retr.entity_map.get(pid,[])) for pid in pids}
        top5_entities={name:sum(sizes_by_pid[pid] for pid in rank[:5]) for name,rank in rankings.items()}
        answers=parse_list(comp[qid].get('answer',''))
        hard_answer=parse_list(hard[qid].get('answer',''))
        goldp=sorted({entity_to_partition[str(x)] for x in answers if str(x) in entity_to_partition})
        hp=entity_to_partition.get(str(hard_answer[0])) if hard_answer else None
        rows.append({'question_id':qid,'split':a.split,'question':question,'rule_index':comp[qid].get('rule_index'), 'gold_answer_count':len(answers),
          'gold_partition_ids':goldp,'hard_partition_id':hp,'rankings':rankings,'top5_entities_returned':top5_entities,
          'mapped_any_answer':bool(goldp),'mapped_hard_answer':hp is not None})
        latencies.append((time.perf_counter()-t0)*1000)
    valid=[r for r in rows if r['mapped_any_answer']]
    summaries={}
    names=list(rows[0]['rankings']) if rows else []
    for method in names:
        summaries[method]={f'@{k}':metric_for(valid,method,k) for k in (1,3,5,10,50)}
        summaries[method]['hard_answer_denominator']=sum(r['mapped_hard_answer'] for r in rows)
        sizes=[r['top5_entities_returned'][method] for r in valid]
        summaries[method]['top5_entities_returned']={'mean':float(np.mean(sizes)),'median':float(np.median(sizes)),'p95':float(np.quantile(sizes,.95))}
    # Validation-only model selection by any-answer Hits@5.
    hybrids=[n for n in names if n.startswith('hybrid_dense_')]
    selected=max(hybrids,key=lambda n:summaries[n]['@5']['any_hits']) if hybrids else None
    pairs={}
    if selected:
      for baseline in ('dense','bm25','rrf'):
        if baseline in names:
          x=[float(bool(set(r['rankings'][baseline][:5])&set(r['gold_partition_ids']))) for r in valid]
          y=[float(bool(set(r['rankings'][selected][:5])&set(r['gold_partition_ids']))) for r in valid]
          pairs[f'{selected}_vs_{baseline}']={'question_bootstrap':paired_bootstrap(x,y)}
          rules=[r.get('rule_index') or r['question_id'] for r in valid]
          if len(set(rules))>1: pairs[f'{selected}_vs_{baseline}']['rule_cluster_bootstrap']=paired_cluster_bootstrap(x,y,rules)
    outdir=Path(a.out_dir); outdir.mkdir(parents=True,exist_ok=True)
    stem=f'brink_{a.split}_routing_{time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())}'
    raw=outdir/f'{stem}.jsonl'; agg=outdir/f'{stem}.json'
    with raw.open('w',encoding='utf-8') as f:
      for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    try: commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: commit=None
    index_files=['centroids.npy','partition_ids.pkl','entity_map.pkl','bm25_docs.pkl','meta.json']
    source_files=[ROOT/'metrics/scripts/benchmark_brink_routing.py',ROOT/'metrics/score_metrics.py',ROOT/'src/query_subgraph/subgraph_retriever.py']
    report={'run_id':stem,'benchmark':'BRINK-Wikidata5m','split':a.split,'interpretation':'answer-containing partition retrieval; not evidence retrieval or answer accuracy',
      'n_questions':len(rows),'n_any_answer_mapped':len(valid),'any_answer_mapping_coverage':len(valid)/len(rows),
      'hard_answer_mapping_coverage':sum(r['mapped_hard_answer'] for r in rows)/len(rows),
      'index_dir':str(Path(a.index_dir).resolve()),'partition_count':len(pids),'query_protocol':'raw BRINK question text, including QIDs; no entity-linker/name substitution',
      'fusion_protocol':'per-query min-max normalization; dense_weight* dense + (1-dense_weight)*BM25; RRF k=60',
      'router_latency_ms':summarize(latencies),'candidate_scoring_ops_per_query_approx':2*len(pids),
      'provenance':{'code_commit':commit,'python':platform.python_version(),'platform':platform.platform(),
        'device':retr.device,'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        'model':retr.model_name,'input_sha256':{'complete_answers_tsv':sha256(a.questions),'hard_answers_tsv':sha256(a.hard_answers)},
        'index_sha256':{name:sha256(Path(a.index_dir)/name) for name in index_files},
        'code_sha256':{str(f.relative_to(ROOT)):sha256(f) for f in source_files}},
      'selected_hybrid_dev_only':selected if a.split=='val' else None,'metrics_by_method':summaries,'paired_bootstrap_h5':pairs,
      'artifacts':{'per_question':str(raw.resolve())}}
    agg.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
