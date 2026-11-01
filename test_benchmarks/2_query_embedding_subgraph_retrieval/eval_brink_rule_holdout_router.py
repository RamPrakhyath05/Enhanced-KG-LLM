#!/usr/bin/env python3
"""Evaluate a frozen BRINK rule-group holdout for partition routing.

Dev phase sweeps fusion weights and writes a config. Test phase accepts that
config only and evaluates its selected weights plus fixed dense/BM25/RRF/0.5
baselines once. All results are answer-partition retrieval, not QA accuracy.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, json, pickle, re, time
from pathlib import Path
import numpy as np
import torch
from rank_bm25 import BM25Okapi

ROOT=Path(__file__).resolve().parents[2]; TEAM=ROOT.parent; PART=TEAM/'partition'; V2=PART/'v2'
import sys; sys.path.insert(0,str(V2))
from encoder import HFEncoder
QID_RE=re.compile(r'Q\d+')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def tokenize(s): return re.findall(r'\w+',str(s).lower())
def parse_list(s):
 try:
  x=ast.literal_eval(s); return list(x) if isinstance(x,(list,tuple,set)) else [x]
 except Exception: return []
def read_tsv(p):
 with open(p,encoding='utf-8') as f: return list(csv.DictReader(f,delimiter='\t'))
def minmax(a): return (a-a.min(1,keepdims=True))/(a.max(1,keepdims=True)-a.min(1,keepdims=True)+1e-10)
def percentile_ci(x,seed,nboot=10000):
 x=np.asarray(x,dtype=float); rng=np.random.default_rng(seed); n=len(x); out=[]
 for start in range(0,nboot,250):
  ix=rng.integers(0,n,(min(250,nboot-start),n)); out.append(x[ix].mean(1))
 z=np.concatenate(out); return [float(v) for v in np.quantile(z,[.025,.975])]
def cluster_ci(x,rules,seed,nboot=10000):
 x=np.asarray(x,dtype=float); groups=sorted(set(rules)); by={g:np.flatnonzero(np.asarray(rules)==g) for g in groups}; rng=np.random.default_rng(seed); vals=[]
 for _ in range(nboot):
  sample=rng.choice(groups,size=len(groups),replace=True)
  ix=np.concatenate([by[g] for g in sample]); vals.append(float(x[ix].mean()))
 return [float(v) for v in np.quantile(vals,[.025,.975])]
def get_rows(qid_dir,manifest,split,entities_path,index_dir):
 groups=set(manifest['splits'][split]['rules'])
 comp=read_tsv(Path(qid_dir)/'train_cleaned_final_complete.tsv'); hard={r['id']:r for r in read_tsv(Path(qid_dir)/'train_cleaned_final.tsv')}
 id2name={}
 with open(entities_path,encoding='utf-8') as f:
  for r in csv.DictReader(f):
   if r.get('Entity_ID') and r.get('Entity_Name'): id2name[r['Entity_ID']]=r['Entity_Name']
 with open(Path(index_dir)/'entity_map.pkl','rb') as f: emap=pickle.load(f)
 e2p={str(e):int(pid) for pid,es in emap.items() for e in es}
 output=[]; missing_topic=missing_hard=missing_gold=0
 for r in comp:
  if r['rule_index'] not in groups: continue
  q=r['question']; topic_names={x:id2name.get(str(x)) for x in parse_list(r.get('q_entity',''))}
  if not topic_names or any(not n for n in topic_names.values()): missing_topic+=1; continue
  query=QID_RE.sub(lambda m: topic_names.get(m.group(0),m.group(0)),q)
  if QID_RE.search(query): missing_topic+=1; continue
  gold=sorted({e2p[str(e)] for e in parse_list(r.get('answer','')) if str(e) in e2p})
  if not gold: missing_gold+=1; continue
  h=hard.get(r['id']); ha=parse_list(h.get('answer','')) if h else []
  hp=e2p.get(str(ha[0])) if ha else None
  if hp is None: missing_hard+=1
  output.append({'question_id':r['id'],'rule_index':r['rule_index'],'query':query,'gold_partition_ids':gold,'hard_partition_id':hp})
 return output,{'questions':len(output),'missing_topic':missing_topic,'missing_gold_partition':missing_gold,'missing_hard_partition':missing_hard}
def metric(rows,rankings,k):
 anyhit=[float(bool(set(r['gold_partition_ids'])&set(rankings[i][:k]))) for i,r in enumerate(rows)]
 valid_h=[(i,r) for i,r in enumerate(rows) if r['hard_partition_id'] is not None]
 hard=[float(r['hard_partition_id'] in rankings[i][:k]) for i,r in valid_h]
 rec=[]; prec=[]; rr=[]
 for i,r in enumerate(rows):
  rel=set(r['gold_partition_ids']); rank=rankings[i][:k]
  rec.append(len(rel&set(rank))/len(rel)); prec.append(len(rel&set(rank))/k)
  top=rankings[i][:100]; pos=next((j for j,p in enumerate(top,1) if p in rel),None); rr.append(1/pos if pos else 0.)
 return {'any_hits':float(np.mean(anyhit)),'hard_hits':float(np.mean(hard)) if hard else None,
         'mean_partition_recall':float(np.mean(rec)),'mean_partition_precision':float(np.mean(prec)),
         'mrr@100':float(np.mean(rr)),'n_any':len(rows),'n_hard':len(valid_h)}
def rank_all(dense,bm25,pids,weights):
 dn,bn=minmax(dense),minmax(bm25); result={}
 result['dense']=[np.asarray(pids)[np.argsort(-x,kind='stable')[:100]].tolist() for x in dense]
 result['bm25']=[np.asarray(pids)[np.argsort(-x,kind='stable')[:100]].tolist() for x in bm25]
 rrf=[]
 for i in range(len(dense)):
  dr=np.argsort(-dense[i],kind='stable'); br=np.argsort(-bm25[i],kind='stable'); rd=np.empty(len(pids),np.int32); rb=np.empty(len(pids),np.int32)
  rd[dr]=np.arange(1,len(pids)+1); rb[br]=np.arange(1,len(pids)+1)
  rrf.append(np.asarray(pids)[np.argsort(-(1/(60+rd)+1/(60+rb)),kind='stable')[:100]].tolist())
 result['rrf']=rrf
 for w in weights:
  scores=w*dn+(1-w)*bn
  result[f'hybrid_{w:.2f}']=[np.asarray(pids)[np.argsort(-x,kind='stable')[:100]].tolist() for x in scores]
 return result

def load_assets(index_dir):
 index=Path(index_dir)
 centroids=np.load(index/'centroids.npy').astype(np.float32)
 with open(index/'partition_ids.pkl','rb') as f: pids=list(pickle.load(f))
 with open(index/'entity_map.pkl','rb') as f: emap=pickle.load(f)
 with open(index/'bm25_docs.pkl','rb') as f: docs=pickle.load(f)
 token_docs=docs if docs and isinstance(docs[0],list) else [tokenize(x) for x in docs]
 bm25=BM25Okapi(token_docs)
 return centroids,pids,emap,bm25

def run_model(rows,checkpoint,centroids,bm25,pids,device,weights):
 enc=HFEncoder(checkpoint,device=device,max_length=128)
 start=time.perf_counter(); qs=[r['query'] for r in rows]
 qv=enc.encode(qs,batch_size=128,normalize=True,show_progress=False); dense=qv@centroids.T
 dense_seconds=time.perf_counter()-start
 start=time.perf_counter(); bs=np.asarray([bm25.get_scores(tokenize(q)) for q in qs],dtype=np.float32); bm25_seconds=time.perf_counter()-start
 return rank_all(dense,bs,pids,weights),{'mean_query_encode_ms':dense_seconds*1000/len(rows),'mean_bm25_ms':bm25_seconds*1000/len(rows)}
def summarize_sweep(rows,rankings):
 out={}
 for name,ranks in rankings.items():
  out[name]={f'@{k}':metric(rows,ranks,k) for k in [1,3,5,10,50]}
 return out

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--phase',required=True,choices=['dev','test'])
 ap.add_argument('--manifest',default=str(ROOT/'metrics/configs/brink_rule_holdout_20261028.json'))
 ap.add_argument('--qid-dir',default=str(TEAM/'new/benchmark_data/brink_wikidata5m/wikidata5m/qid'))
 ap.add_argument('--entities',default=str(PART/'entities.csv')); ap.add_argument('--index-dir',default=str(ROOT/'data/subg'))
 ap.add_argument('--checkpoint',default=str(ROOT/'metrics/results/models/brink_rule_holdout_minilm_20261028'))
 ap.add_argument('--config',default=str(ROOT/'metrics/configs/brink_rule_holdout_frozen.json'))
 ap.add_argument('--out-dir',default=str(ROOT/'metrics/results/raw/brink_rule_holdout_20261028'))
 ap.add_argument('--device',default='cuda')
 a=ap.parse_args()
 if a.device=='cuda' and not torch.cuda.is_available(): raise SystemExit('CUDA requested but unavailable')
 manifest=json.loads(Path(a.manifest).read_text()); rows,coverage=get_rows(a.qid_dir,manifest,'dev' if a.phase=='dev' else 'test',a.entities,a.index_dir)
 print('Loaded questions',coverage,flush=True)
 centroids,pids,emap,bm25=load_assets(a.index_dir)
 weights=[round(x,2) for x in np.arange(0,1.0001,.05)]
 models={'base':'sentence-transformers/all-MiniLM-L6-v2','finetuned':a.checkpoint}
 if a.phase=='test':
  config=json.loads(Path(a.config).read_text())
  if config.get('status')!='frozen_for_one_time_test': raise SystemExit('Test config is not frozen')
  if config['split_manifest_sha256']!=sha(a.manifest): raise SystemExit('Split manifest changed after dev selection')
  if config['index_sha256']!={n:sha(Path(a.index_dir)/n) for n in ['centroids.npy','partition_ids.pkl','entity_map.pkl','bm25_docs.pkl']}: raise SystemExit('Index files changed after dev selection')
  weights_by_model={k:float(v) for k,v in config['selected_fusion_weight_by_model'].items()}
 else: weights_by_model={}
 all_reports={}; all_perq={}; wall={}
 for model_name,checkpoint in models.items():
  use_weights=weights if a.phase=='dev' else [weights_by_model[model_name],.50]
  ranks,timing=run_model(rows,checkpoint,centroids,bm25,pids,a.device,use_weights)
  wall[model_name]=timing
  if a.phase=='dev':
   sums=summarize_sweep(rows,ranks); all_reports[model_name]={'checkpoint':checkpoint,'metrics_by_weight':sums,'latency':timing}
   best=max(weights,key=lambda w:(sums[f'hybrid_{w:.2f}']['@5']['any_hits'],-w))
   weights_by_model[model_name]=best
   selected=f'hybrid_{best:.2f}'
   # Retain compact per-question outcomes for all dev sweep points to make the selected weight auditable.
   all_perq[model_name]=[{'question_id':r['question_id'],'rule_index':r['rule_index'],'gold_partition_ids':r['gold_partition_ids'],'hard_partition_id':r['hard_partition_id'],
                         'h5_by_weight':{f'{w:.2f}':float(bool(set(r['gold_partition_ids'])&set(ranks[f"hybrid_{w:.2f}"][i][:5]))) for w in weights}} for i,r in enumerate(rows)]
  else:
   w=weights_by_model[model_name]; selected=f'hybrid_{w:.2f}'; names=['dense','bm25','rrf','hybrid_0.50',selected]
   chosen={n:ranks[n] for n in dict.fromkeys(names)}
   summaries=summarize_sweep(rows,chosen)
   all_reports[model_name]={'checkpoint':checkpoint,'selected_weight':w,'metrics_by_method':summaries,'latency':timing}
   for name in chosen:
    for i,r in enumerate(rows): all_perq.setdefault(r['question_id'],{'question_id':r['question_id'],'rule_index':r['rule_index'],'query':r['query'],
      'gold_partition_ids':r['gold_partition_ids'],'hard_partition_id':r['hard_partition_id'],'rankings':{}})['rankings'][f'{model_name}:{name}']=chosen[name][i]
 if a.phase=='dev':
  cfg={'status':'frozen_for_one_time_test','created':'2026-10-28','split_manifest_sha256':sha(a.manifest),
       'split_policy':'rule_index-disjoint; train/dev/test group IDs frozen before model fitting',
       'selection_metric':'answer-containing partition Hits@5 on dev; ties choose smaller dense weight',
       'selected_fusion_weight_by_model':weights_by_model,
       'base_checkpoint':models['base'],'finetuned_checkpoint':models['finetuned'],
       'finetuned_training_manifest_sha256':sha(Path(a.checkpoint)/'training_manifest.json'),
       'index_sha256':{n:sha(Path(a.index_dir)/n) for n in ['centroids.npy','partition_ids.pkl','entity_map.pkl','bm25_docs.pkl']},
       'question_files_sha256':{n:sha(Path(a.qid_dir)/n) for n in ['train_cleaned_final_complete.tsv','train_cleaned_final.tsv']},
       'query_protocol':'replace topic QIDs using BRINK q_entity canonical names; no automatic linker; answer annotations used only for labels',
       'test_policy':'no test scores or rankings before this config is frozen'}
  Path(a.config).write_text(json.dumps(cfg,indent=2)+'\n')
 else:
  # Paired test uncertainty: fine-tuned vs base at their dev-selected weights.
  base_w=weights_by_model['base']; ft_w=weights_by_model['finetuned']
  base_key=f'base:hybrid_{base_w:.2f}'; ft_key=f'finetuned:hybrid_{ft_w:.2f}'
  qids=[r['question_id'] for r in rows]; rules=[r['rule_index'] for r in rows]
  base_hits=[]; ft_hits=[]
  for r in rows:
   d=all_perq[r['question_id']]['rankings']; gold=set(r['gold_partition_ids'])
   base_hits.append(float(bool(gold&set(d[base_key][:5])))); ft_hits.append(float(bool(gold&set(d[ft_key][:5]))))
  delta=np.asarray(ft_hits)-np.asarray(base_hits)
  pair={'comparison':'fine-tuned minus base at per-model dev-selected fusion weights','metric':'answer-containing partition Hits@5',
        'base_h5':float(np.mean(base_hits)),'finetuned_h5':float(np.mean(ft_hits)),'delta':float(np.mean(delta)),
        'question_paired_bootstrap_95ci':percentile_ci(delta,20261028),
        'rule_cluster_paired_bootstrap_95ci':cluster_ci(delta,rules,20261029),
        'bootstrap_replicates':10000,'n_questions':len(rows),'n_rule_clusters':len(set(rules))}
  all_reports['paired_primary_comparison']=pair
 report={'benchmark':'BRINK-Wikidata5M internal rule-heldout partition routing','phase':a.phase,
  'n_questions':len(rows),'coverage':coverage,'split_manifest_sha256':sha(a.manifest),
  'index_dir':str(Path(a.index_dir).resolve()),'partition_count':len(pids),'query_encoder':models,
  'interpretation':'answer-containing partition ranking; not answer correctness or evidence/path retrieval',
  'primary_metric':'any-answer Hits@5; additionally hard-answer Hits@5, partition recall/precision@K, MRR@100, and candidate entity volume',
  'models':all_reports,'device':a.device,'gpu':torch.cuda.get_device_name() if a.device=='cuda' else None,
  'provenance':{'script_sha256':sha(__file__),'manifest_sha256':sha(a.manifest),'centroids_sha256':sha(Path(a.index_dir)/'centroids.npy')}}
 out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); stem=f'brink_{a.phase}_rule_holdout'
 (out/f'{stem}.json').write_text(json.dumps(report,indent=2)+'\n')
 with (out/f'{stem}.jsonl').open('w') as f:
  for x in all_perq.values() if a.phase=='test' else [row for model in all_perq.values() for row in model]: f.write(json.dumps(x,ensure_ascii=False)+'\n')
 print(json.dumps(report,indent=2)); print('Wrote',out/f'{stem}.json',out/f'{stem}.jsonl')
if __name__=='__main__': main()
