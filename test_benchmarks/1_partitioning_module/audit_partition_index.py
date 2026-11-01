#!/usr/bin/env python3
"""Integrity and footprint audit for a built partition retrieval index."""
import argparse,hashlib,json,pickle,statistics
from pathlib import Path
import numpy as np

def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--index-dir',required=True); p.add_argument('--out'); p.add_argument('--hash',action='store_true'); a=p.parse_args()
 d=Path(a.index_dir); names=['centroids.npy','partition_ids.pkl','entity_map.pkl','bm25_docs.pkl','meta.json']
 missing=[n for n in names if not (d/n).is_file()]
 if missing: raise SystemExit(f'missing index files: {missing}')
 cent=np.load(d/'centroids.npy',mmap_mode='r'); pids=pickle.load(open(d/'partition_ids.pkl','rb'))
 emap=pickle.load(open(d/'entity_map.pkl','rb')); docs=pickle.load(open(d/'bm25_docs.pkl','rb')); meta=json.load(open(d/'meta.json'))
 sizes=[len(emap.get(pid,[])) for pid in pids]
 ids=[int(x) for x in pids]; unique=len(set(ids))
 report={'index_dir':str(d.resolve()),'metadata':meta,'checks':{'centroid_rows_match_partition_ids':len(cent)==len(pids),
  'bm25_docs_match_partition_ids':len(docs)==len(pids),'partition_ids_unique':unique==len(pids),
  'entity_map_has_every_partition':all(pid in emap for pid in pids),'centroids_all_finite':bool(np.isfinite(cent).all()),
  'centroid_dim':int(cent.shape[1]),'partition_count':len(pids),'unique_partition_ids':unique},
  'partition_size':{'entities_total':sum(sizes),'min':min(sizes) if sizes else 0,
  'median':statistics.median(sizes) if sizes else 0,'p95':float(np.quantile(sizes,.95)) if sizes else 0,
  'max':max(sizes) if sizes else 0,'empty':sum(x==0 for x in sizes)},
  'artifacts':{n:{'bytes':(d/n).stat().st_size} for n in names}}
 if a.hash:
  for n in names: report['artifacts'][n]['sha256']=digest(d/n)
 report['status']='PASS' if all(v for k,v in report['checks'].items() if isinstance(v,bool)) else 'FAIL'
 out=Path(a.out) if a.out else None
 if out: out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
 if report['status']!='PASS': raise SystemExit(2)
if __name__=='__main__': main()
