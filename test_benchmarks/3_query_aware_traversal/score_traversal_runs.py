#!/usr/bin/env python3
"""Compare traversal outputs at identical edge-inspection budgets.

JSONL schema per question: {id, gold_evidence_paths:[[triple_id,...],...],
methods:{name:{budgets:{"25":{evidence:[...],latency_ms:..,edges_examined:..,
scoring_operations:..}, ...}}}}.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from metrics.score_metrics import set_recall,path_coverage,paired_bootstrap,paired_cluster_bootstrap,summarize

def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--out',required=True); a=p.parse_args()
 rows=[json.loads(x) for x in open(a.input,encoding='utf8') if x.strip()]
 budgets=sorted({int(b) for r in rows for m in r.get('methods',{}).values() for b in m.get('budgets',{})})
 methods=sorted({m for r in rows for m in r.get('methods',{})}); report={'n_questions':len(rows),'budgets':{}}
 per={}
 for b in budgets:
  report['budgets'][str(b)]={}; per[str(b)]={}
  for method in methods:
   recall=[]; complete=[]; latency=[]; entries=[]; ops=[]
   for r in rows:
    x=r.get('methods',{}).get(method,{}).get('budgets',{}).get(str(b))
    if x is None: continue
    rr=set_recall(x.get('evidence',[]),[e for path in r.get('gold_evidence_paths',[]) for e in path])
    pc=path_coverage(x.get('evidence',[]),r.get('gold_evidence_paths',[]))
    if rr is not None: recall.append(rr)
    if pc is not None: complete.append(pc)
    for key,arr in [('latency_ms',latency),('scoring_operations',ops)]:
     if isinstance(x.get(key),(int,float)): arr.append(float(x[key]))
    inspected=x.get('adjacency_entries_fetched',x.get('edges_examined'))
    if isinstance(inspected,(int,float)): entries.append(float(inspected))
   report['budgets'][str(b)][method]={'evidence_recall_macro':sum(recall)/len(recall) if recall else None,
     'complete_path_coverage':sum(complete)/len(complete) if complete else None,'n_recall':len(recall),'n_paths':len(complete),
     'latency_ms':summarize(latency),'adjacency_entries':summarize(entries),'scoring_operations':summarize(ops)}
   per[str(b)][method]={'recall':recall,'complete':complete}
  if methods:
   baseline=methods[0]
   for method in methods[1:]:
    common=[]
    for r in rows:
     x=r.get('methods',{}).get(baseline,{}).get('budgets',{}).get(str(b)); y=r.get('methods',{}).get(method,{}).get('budgets',{}).get(str(b))
     if x is not None and y is not None and r.get('gold_evidence_paths'):
      common.append((r,path_coverage(x.get('evidence',[]),r['gold_evidence_paths']),path_coverage(y.get('evidence',[]),r['gold_evidence_paths'])))
    if common:
     aa=[x for _,x,_ in common]; bb=[y for _,_,y in common]; cluster=[r.get('rule_index') or r.get('id') for r,_,_ in common]
     item={'question_bootstrap':paired_bootstrap(aa,bb)}
     if len(set(cluster))>1: item['rule_cluster_bootstrap']=paired_cluster_bootstrap(aa,bb,cluster)
     report['budgets'][str(b)].setdefault('paired_complete_path_ci',{})[f'{method}_minus_{baseline}']=item
 report['notes']='Recall is over the union of annotated evidence triples; path coverage is whether at least one listed gold path is complete. State the traversal adjacency-budget accounting rule alongside this report.'
 Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(report,indent=2)+'\n',encoding='utf8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
