#!/usr/bin/env python3
"""Map BRINK answer QIDs to labels for free-form answer scoring."""
import argparse, ast, csv, json
from pathlib import Path

def parse(x):
 try:
  v=ast.literal_eval(x); return [str(y) for y in v] if isinstance(v,(list,tuple,set)) else [str(v)]
 except Exception: return [x] if x else []

def main():
 p=argparse.ArgumentParser(); p.add_argument('--complete',required=True); p.add_argument('--hard',required=True); p.add_argument('--entities-csv',required=True); p.add_argument('--out',required=True); a=p.parse_args()
 with open(a.complete,encoding='utf8') as f: comp={r['id']:r for r in csv.DictReader(f,delimiter='\t')}
 with open(a.hard,encoding='utf8') as f: hard={r['id']:r for r in csv.DictReader(f,delimiter='\t')}
 wanted={x for r in comp.values() for x in parse(r.get('answer',''))}
 wanted|={x for r in hard.values() for x in parse(r.get('answer',''))}
 labels={}
 with open(a.entities_csv,encoding='utf8',newline='') as f:
  for r in csv.DictReader(f):
   if r.get('Entity_ID') in wanted: labels[r['Entity_ID']]=r.get('Entity_Name','')
 rows=[]; missing=set()
 for qid,row in comp.items():
  ans=[]
  for x in parse(row.get('answer','')):
   if x in labels: ans.append(labels[x])
   else: missing.add(x)
  ha=parse(hard[qid].get('answer',''))
  hard_label=labels.get(ha[0]) if ha else None
  if ha and hard_label is None: missing.add(ha[0])
  rows.append({'id':qid,'question':row.get('question',''),'rule_index':row.get('rule_index'),'answers':ans,'hard_answer':hard_label})
 out={'gold':rows,'mapping_coverage':1-len(missing)/len(wanted) if wanted else 1.0,'unmapped_qids':sorted(missing)}
 Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
 print(f"questions={len(rows)} unique_answer_ids={len(wanted)} mapped={len(wanted)-len(missing)} coverage={out['mapping_coverage']:.6f} output={a.out}")
if __name__=='__main__': main()
