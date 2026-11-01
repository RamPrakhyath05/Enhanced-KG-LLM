#!/usr/bin/env python3
"""Paper analysis from the frozen, rule-heldout BRINK test predictions."""
from __future__ import annotations
import json,pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'metrics/results/raw/brink_rule_holdout_20261028/brink_test_rule_holdout.jsonl'
AGG=ROOT/'metrics/results/raw/brink_rule_holdout_20261028/brink_test_rule_holdout.json'
OUT=ROOT/'metrics/results/reports/brink_rule_holdout_paper_analysis.json'

def boot_ci(x,seed,n=10000):
 x=np.asarray(x,dtype=float); rng=np.random.default_rng(seed); vals=[]
 for st in range(0,n,250):
  ix=rng.integers(0,len(x),(min(250,n-st),len(x))); vals.extend(x[ix].mean(1))
 return [float(v) for v in np.quantile(vals,[.025,.975])]
def cluster_ci(x,rules,seed,n=10000):
 x=np.asarray(x,dtype=float); groups=sorted(set(rules)); ix={g:np.flatnonzero(np.asarray(rules)==g) for g in groups}; rng=np.random.default_rng(seed); vals=[]
 for _ in range(n):
  samp=rng.choice(groups,len(groups),replace=True); ids=np.concatenate([ix[g] for g in samp]); vals.append(float(x[ids].mean()))
 return [float(v) for v in np.quantile(vals,[.025,.975])]
def main():
 rows=[json.loads(s) for s in RAW.open() if s.strip()]; agg=json.loads(AGG.read_text());
 with open(ROOT/'data/subg/entity_map.pkl','rb') as f: emap=pickle.load(f)
 sizes={int(pid):len(entities) for pid,entities in emap.items()}
 test_model=agg['models']; base_w=test_model['base']['selected_weight']; ft_w=test_model['finetuned']['selected_weight']
 keys={'base':f'base:hybrid_{base_w:.2f}','finetuned':f'finetuned:hybrid_{ft_w:.2f}','bm25':'finetuned:bm25','rrf_base':'base:rrf','rrf_finetuned':'finetuned:rrf'}
 out={'status':'post-hoc analysis of frozen BRINK internal rule-heldout test; no retrieval rerun/tuning',
      'n_questions':len(rows),'n_rule_clusters':len(set(r['rule_index'] for r in rows)),
      'selected_weights':{'base':base_w,'finetuned':ft_w},'candidate_entity_volume_top5':{},
      'paired_h5_comparisons':{},'h_at_k':{},'metrics_at_5':{},
      'statistical_note':'Question and rule-cluster paired bootstrap intervals are unadjusted. Rule-cluster intervals are primary for this grouped split.'}
 rules=[r['rule_index'] for r in rows]
 hit_arrays={}
 for label,key in keys.items():
  any_hit=[]; hard_hit=[]; rec=[]; prec=[]; rr=[]; volume=[]
  for r in rows:
   rank=r['rankings'][key]; gold=set(r['gold_partition_ids']); top=rank[:5]
   any_hit.append(float(bool(gold&set(top)))); hard_hit.append(float(r['hard_partition_id'] in top))
   rec.append(len(gold&set(top))/len(gold)); prec.append(len(gold&set(top))/5)
   pos=next((i for i,p in enumerate(rank[:100],1) if p in gold),None); rr.append(1/pos if pos else 0.)
   volume.append(sum(sizes[p] for p in top))
  hit_arrays[label]=np.asarray(any_hit)
  out['candidate_entity_volume_top5'][label]={'mean':float(np.mean(volume)),'median':float(np.median(volume)),'p95':float(np.quantile(volume,.95))}
  out['h_at_k'][label]={}
  for k in [1,3,5,10,50]:
   h=[float(bool(set(r['gold_partition_ids'])&set(r['rankings'][key][:k]))) for r in rows]
   out['h_at_k'][label][str(k)]={'hits':float(np.mean(h)),'question_bootstrap_ci95':boot_ci(h,20261030+k)}
  out['metrics_at_5'][label]={'hard_hits@5':float(np.mean(hard_hit)),'partition_recall@5':float(np.mean(rec)),
      'partition_precision@5':float(np.mean(prec)),'mrr@100':float(np.mean(rr))}
 for baseline in ['base','bm25','rrf_base','rrf_finetuned']:
  delta=hit_arrays['finetuned']-hit_arrays[baseline]
  out['paired_h5_comparisons'][f'finetuned_minus_{baseline}']={'delta':float(delta.mean()),
      'question_bootstrap_ci95':boot_ci(delta,20261101),'rule_cluster_bootstrap_ci95':cluster_ci(delta,rules,20261102),
      'n_questions':len(rows),'n_rule_clusters':len(set(rules))}
 OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(out,indent=2)+'\n')
 fig,ax=plt.subplots(figsize=(6.3,4.3),constrained_layout=True)
 colors={'base':'#5b6573','finetuned':'#16836c','bm25':'#df8730','rrf_base':'#2780b9','rrf_finetuned':'#8b6bb3'}
 names={'base':'Base MiniLM + hybrid','finetuned':'Fine-tuned MiniLM + hybrid','bm25':'BM25','rrf_base':'RRF (base dense)','rrf_finetuned':'RRF (fine-tuned dense)'}
 for label in keys:
  ks=[1,3,5,10,50]; vals=[out['h_at_k'][label][str(k)]['hits'] for k in ks]
  ax.plot(ks,vals,marker='o',linewidth=2,label=names[label],color=colors[label])
 ax.set_xscale('log'); ax.set_xticks([1,3,5,10,50],labels=['1','3','5','10','50']); ax.set_ylim(0,1)
 ax.set_xlabel('Retrieved partitions, k'); ax.set_ylabel('Answer-containing partition Hits@k')
 ax.set_title('BRINK rule-heldout test (n=4,157; 27 unseen rules)')
 ax.grid(True,axis='y',alpha=.25); ax.legend(frameon=False,fontsize=8,loc='lower right')
 fig.savefig(OUT.parent/'brink_rule_holdout_h_at_k.png',dpi=300,bbox_inches='tight')
 fig.savefig(OUT.parent/'brink_rule_holdout_h_at_k.pdf',bbox_inches='tight')
 print(json.dumps(out,indent=2)); print('Saved',OUT)
if __name__=='__main__': main()
