#!/usr/bin/env python3
"""Paired same-question comparison of two raw JSONL runs after metric scoring."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from metrics.score_metrics import score_record,paired_bootstrap,paired_cluster_bootstrap

def get_path(x,path):
 for part in path.split('.'):
  if isinstance(x,dict): x=x.get(part)
  else: return None
 return x

def load(path):
 out={}
 for line in open(path,encoding='utf8'):
  if not line.strip(): continue
  row=json.loads(line); key=str(row.get('question_id',row.get('id')))
  out[key]=(row,score_record(row))
 return out

def main():
 p=argparse.ArgumentParser(); p.add_argument('--run-a',required=True); p.add_argument('--run-b',required=True)
 p.add_argument('--metric',required=True,help='Scored field, e.g. evidence_recall or entity_linking.link_recall_at_3')
 p.add_argument('--out',required=True); p.add_argument('--bootstrap-samples',type=int,default=10000); a=p.parse_args()
 aa,bb=load(a.run_a),load(a.run_b); common=sorted(set(aa)&set(bb)); x=[]; y=[]; clusters=[]
 for qid in common:
  va=get_path(aa[qid][1],a.metric); vb=get_path(bb[qid][1],a.metric)
  if isinstance(va,(int,float)) and isinstance(vb,(int,float)):
   x.append(float(va)); y.append(float(vb)); clusters.append(aa[qid][0].get('rule_index') or qid)
 if not x: raise SystemExit('No paired examples have this metric in both runs.')
 report={'run_a':str(Path(a.run_a).resolve()),'run_b':str(Path(a.run_b).resolve()),'metric':a.metric,
  'paired_questions':len(x),'mean_a':sum(x)/len(x),'mean_b':sum(y)/len(y),
  'question_bootstrap':paired_bootstrap(x,y,samples=a.bootstrap_samples)}
 if len(set(clusters))>1: report['rule_cluster_bootstrap']=paired_cluster_bootstrap(x,y,clusters,samples=a.bootstrap_samples)
 Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(report,indent=2)+'\n',encoding='utf8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
