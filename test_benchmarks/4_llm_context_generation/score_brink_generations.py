#!/usr/bin/env python3
"""Score free-form BRINK outputs with official-style normalized answer-set metrics."""
import argparse, json, re, string
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from metrics.score_metrics import prediction_set_metrics, paired_bootstrap, paired_cluster_bootstrap
ARTICLES={'a','an','the'}
def normalize(s):
 s=str(s).lower().replace('<pad>',' '); s=''.join(c for c in s if c not in string.punctuation)
 return ' '.join(t for t in s.split() if t not in ARTICLES)
def candidates(s): return list(dict.fromkeys(normalize(x) for x in re.split(r'[,\n\r]+',str(s or '')) if normalize(x)))
def main():
 p=argparse.ArgumentParser(); p.add_argument('--gold',required=True,help='JSON produced by prepare_brink_gold.py'); p.add_argument('--predictions',required=True,help='JSONL with id and prediction'); p.add_argument('--compare',help='Optional second run JSONL for paired comparison'); p.add_argument('--out',required=True); a=p.parse_args()
 gold=json.load(open(a.gold,encoding='utf8'))['gold']; pred={}
 for line in open(a.predictions,encoding='utf8'):
  if line.strip():
   x=json.loads(line); pred[str(x['id'])]=x.get('prediction','')
 per=[]; keys=('hits_any','precision','recall','f1','hits_hard')
 for g in gold:
  ps=candidates(pred.get(str(g['id']),'')); m=prediction_set_metrics(ps,g['answers'],g.get('hard_answer'))
  per.append({'id':str(g['id']),**m,'prediction_present':str(g['id']) in pred})
 n=len(per); agg={k:sum(x[k] for x in per)/n for k in keys} if n else {}
 if a.compare:
  other={}
  for line in open(a.compare,encoding='utf8'):
   if line.strip():
    z=json.loads(line); other[str(z['id'])]=z.get('prediction','')
  gold_by_id={str(g['id']):g for g in gold}; common=[str(g['id']) for g in gold if str(g['id']) in pred and str(g['id']) in other]
  paired={}
  for key in keys:
   left=[]; right=[]; groups=[]
   for qid in common:
    g=gold_by_id[qid]; left.append(prediction_set_metrics(candidates(pred[qid]),g['answers'],g.get('hard_answer'))[key]); right.append(prediction_set_metrics(candidates(other[qid]),g['answers'],g.get('hard_answer'))[key]); groups.append(g.get('rule_index') or qid)
   if left:
    paired[key]={'question_bootstrap':paired_bootstrap(left,right)}
    if len(set(groups))>1: paired[key]['rule_cluster_bootstrap']=paired_cluster_bootstrap(left,right,groups)
  agg['paired_comparison']={'run_a':a.predictions,'run_b':a.compare,'n_shared_questions':len(common),'difference_is_b_minus_a':paired}
 agg['HHR']=agg['hits_hard']/agg['hits_any'] if agg.get('hits_any',0)>0 else 0.0
 agg.update({'n_questions':n,'missing_predictions':sum(not x['prediction_present'] for x in per),
             'per_question':per,'benchmark':'BRINK-Wikidata5m; official-style normalized answer-set evaluation'})
 Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(agg,indent=2)+'\n',encoding='utf8'); print(json.dumps({k:v for k,v in agg.items() if k!='per_question'},indent=2))
if __name__=='__main__': main()
