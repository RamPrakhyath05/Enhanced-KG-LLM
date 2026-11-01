#!/usr/bin/env python3
"""Train a fresh BRINK query encoder on the train groups in a frozen rule split.

Uses multi-positive partition cross-entropy so alternative gold answer
partitions are not mislabeled as negatives. Dev/test groups are never read.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, json, random, re, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
TEAM=ROOT.parent
PART=TEAM/'partition'
V2=PART/'v2'
sys.path.insert(0,str(V2))
from encoder import HFEncoder

QID_RE=re.compile(r'Q\d+')
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def parse_list(s):
 try:
  x=ast.literal_eval(s); return list(x) if isinstance(x,(list,tuple,set)) else [x]
 except Exception: return []
def read_tsv(p):
 with open(p,encoding='utf-8') as f: return list(csv.DictReader(f,delimiter='\t'))
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--manifest',default=str(ROOT/'metrics/configs/brink_rule_holdout_20261028.json'))
 ap.add_argument('--qid-dir',default=str(TEAM/'new/benchmark_data/brink_wikidata5m/wikidata5m/qid'))
 ap.add_argument('--entities',default=str(PART/'entities.csv'))
 ap.add_argument('--index-dir',default=str(ROOT/'data/subg'))
 ap.add_argument('--out',required=True)
 ap.add_argument('--epochs',type=int,default=5); ap.add_argument('--batch-size',type=int,default=64)
 ap.add_argument('--lr',type=float,default=2e-5); ap.add_argument('--tau',type=float,default=.05)
 ap.add_argument('--seed',type=int,default=20261028); ap.add_argument('--device',default='cuda')
 a=ap.parse_args();
 if a.device=='cuda' and not torch.cuda.is_available(): raise SystemExit('CUDA requested but unavailable')
 random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
 if a.device=='cuda': torch.cuda.manual_seed_all(a.seed)
 manifest=json.loads(Path(a.manifest).read_text())
 if manifest.get('status','').startswith('fresh rule-grouped') is False: raise SystemExit('Unexpected split manifest')
 train_rules=set(manifest['splits']['train']['rules']); dev_rules=set(manifest['splits']['dev']['rules']); test_rules=set(manifest['splits']['test']['rules'])
 if train_rules&dev_rules or train_rules&test_rules or dev_rules&test_rules: raise SystemExit('Rule split overlap')
 # Load the frozen w_cosine centroids and answer entity -> partition map.
 centroids=np.load(Path(a.index_dir)/'centroids.npy').astype(np.float32)
 import pickle
 with open(Path(a.index_dir)/'partition_ids.pkl','rb') as f: pids=list(pickle.load(f))
 with open(Path(a.index_dir)/'entity_map.pkl','rb') as f: emap=pickle.load(f)
 pidrow={int(pid):i for i,pid in enumerate(pids)}; e2p={str(e):int(pid) for pid,es in emap.items() for e in es}
 # BRINK topic-name mapping; no answer entities used to build queries.
 id2name={}
 with open(a.entities,encoding='utf-8') as f:
  for row in csv.DictReader(f):
   eid=row.get('Entity_ID',''); name=row.get('Entity_Name','')
   if eid and name: id2name[eid]=name
 rows=read_tsv(Path(a.qid_dir)/'train_cleaned_final_complete.tsv')
 queries=[]; positives=[]; used=[]; no_topic=0; no_gold=0
 for r in rows:
  if r['rule_index'] not in train_rules: continue
  q=str(r['question'])
  topics=parse_list(r.get('q_entity',''))
  topic_names={str(x):id2name.get(str(x)) for x in topics}
  if any(not v for v in topic_names.values()): no_topic+=1; continue
  q=QID_RE.sub(lambda m: topic_names.get(m.group(0),m.group(0)),q)
  if QID_RE.search(q): no_topic+=1; continue
  gold=set()
  for answer in parse_list(r.get('answer','')):
   pid=e2p.get(str(answer))
   if pid in pidrow: gold.add(pidrow[pid])
  if not gold: no_gold+=1; continue
  queries.append(q); positives.append(sorted(gold)); used.append(r['id'])
 if not queries: raise SystemExit('No training pairs')
 print(json.dumps({'train_pairs':len(queries),'no_topic_label':no_topic,'no_mapped_gold':no_gold,'centroids':centroids.shape,'device':a.device},indent=2),flush=True)
 C=torch.from_numpy(centroids).to(a.device)
 enc=HFEncoder('sentence-transformers/all-MiniLM-L6-v2',device=a.device,max_length=128)
 opt=torch.optim.AdamW(enc.parameters(),lr=a.lr)
 n=len(queries); batch=a.batch_size; steps=a.epochs*((n+batch-1)//batch)
 sched=torch.optim.lr_scheduler.LinearLR(opt,start_factor=1.0,end_factor=.1,total_iters=max(steps,1))
 started=time.time()
 for ep in range(a.epochs):
  enc.train(); order=np.random.permutation(n); losses=[]
  for off in range(0,n,batch):
   ix=order[off:off+batch]; qs=[queries[int(i)] for i in ix]
   emb=enc.encode_torch(qs,normalize=True); logits=emb@C.T/a.tau
   row_losses=[]
   for j,i in enumerate(ix):
    pos=torch.as_tensor(positives[int(i)],device=a.device,dtype=torch.long)
    row_losses.append(torch.logsumexp(logits[j],dim=0)-torch.logsumexp(logits[j].index_select(0,pos),dim=0))
   loss=torch.stack(row_losses).mean()
   opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); losses.append(float(loss.detach()))
   if (off//batch)%100==0: print(f'epoch={ep+1} batch={off//batch}/{(n+batch-1)//batch} loss={np.mean(losses):.4f}',flush=True)
  print(f'epoch={ep+1} mean_loss={np.mean(losses):.5f}',flush=True)
  enc.eval()
 out=Path(a.out)
 if out.exists() and any(out.iterdir()): raise SystemExit(f'Output already exists; refusing overwrite: {out}')
 out.mkdir(parents=True,exist_ok=True); enc.save(str(out))
 metadata={'objective':'multi-positive softmax cross-entropy over frozen w_cosine centroids; all answer partitions treated positive',
  'base_model':'sentence-transformers/all-MiniLM-L6-v2','train_split':'fresh rule-grouped train portion only','n_training_pairs':len(queries),
  'groups':len(train_rules),'epochs':a.epochs,'batch_size':batch,'learning_rate':a.lr,'temperature':a.tau,'seed':a.seed,
  'elapsed_seconds':time.time()-started,'device':a.device,'gpu':torch.cuda.get_device_name() if a.device=='cuda' else None,
  'split_manifest_sha256':sha(a.manifest),'index_centroids_sha256':sha(Path(a.index_dir)/'centroids.npy'),
  'train_questions_sha256':sha(Path(a.qid_dir)/'train_cleaned_final_complete.tsv'),'entity_table_sha256':sha(a.entities),
  'query_protocol':'BRINK q_entity used only to substitute canonical entity names for QIDs; no answer data in the query text',
  'train_question_ids_sha256':hashlib.sha256(('\n'.join(sorted(used))).encode()).hexdigest(),
  'excluded_due_missing_topic':no_topic,'excluded_due_unmapped_answer':no_gold}
 (out/'training_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
 print('SAVED',out,flush=True)
 print(json.dumps(metadata,indent=2),flush=True)
if __name__=='__main__': main()
