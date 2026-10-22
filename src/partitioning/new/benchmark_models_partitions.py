#!/usr/bin/env python3
"""Benchmark all grouped-split encoders against rebuilt common-source indexes.

Only one temporary full BM25 index is materialized at a time to fit available
disk. Memberships, compact centroids, per-question top-100 predictions and
aggregate metrics are retained under paper_eval/controlled.
"""
import ast,csv,json,os,pickle,subprocess,sys,time,shutil
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

HERE=Path(__file__).resolve().parent
PART=Path('/home/pesu-rf/Documents/Capstone_Team165/partition')
DATA=Path('/home/pesu-rf/Documents/Capstone_Team165/new/benchmark_data/brink_wikidata5m/wikidata5m/qid')
OUT=HERE/'controlled'; INDEX_BUILDER=PART/'subg/build_partition_index.py'
SPLIT=json.load(open(HERE/'grouped_internal_split.json'))
sys.path.insert(0,str(HERE))
from run_grouped_encoder_eval import QueryModel,load_model,tokenize

MODELS={'minilm_zero_shot':('sentence-transformers/all-MiniLM-L6-v2',None),
        'minilm_finetuned':('sentence-transformers/all-MiniLM-L6-v2',str(HERE/'models/minilm_finetuned')),
        'mpnet_adapter':('sentence-transformers/all-mpnet-base-v2',str(HERE/'models/mpnet_adapter'))}
WEIGHTS=[round(x,2) for x in np.arange(0,1.001,.05)]
KS=[1,3,5,10,20,50,100]

def query_data(part='dev'):
    d=json.load(open(HERE/f'results/minilm_zero_shot/{part}_per_question.json'))
    d={int(x['id']):x for x in d}
    with open(DATA/'train_cleaned_final_complete.tsv',encoding='utf8') as f: comp={int(r['id']):r for r in csv.DictReader(f,delimiter='\t')}
    with open(DATA/'train_cleaned_final.tsv',encoding='utf8') as f: hard={int(r['id']):r for r in csv.DictReader(f,delimiter='\t')}
    ids=sorted(SPLIT['splits'][part]['question_ids'])
    texts=[d[i]['question'] for i in ids]
    gold=[set(ast.literal_eval(comp[i]['answer'])) for i in ids]
    hardans=[ast.literal_eval(hard[i]['answer'])[0] for i in ids]
    rules=[hard[i]['rule_index'] for i in ids]
    return {'ids':ids,'text':texts,'gold':gold,'hard':hardans,'rules':rules}

def encode_queries():
    splits={s:query_data(s) for s in ('dev','test')}
    qemb={}
    for name,(model_id,checkpoint) in MODELS.items():
        path=HERE/'controlled'/f'queries_{name}.npz'
        if path.exists():
            z=np.load(path); qemb[name]={s:z[s] for s in splits}
        else:
            m=load_model(checkpoint,model_id)
            qemb[name]={s:m.encode(d['text']) for s,d in splits.items()}
            np.savez_compressed(path,**qemb[name]); del m
    return splits,qemb

def norm(a):
    return (a-a.min(1,keepdims=True))/(a.max(1,keepdims=True)-a.min(1,keepdims=True)+1e-10)

def topk(scores,k=100):
    ix=np.argpartition(-scores,k-1,axis=1)[:,:k]
    vals=np.take_along_axis(scores,ix,axis=1)
    srt=np.argsort(-vals,axis=1)
    return np.take_along_axis(ix,srt,axis=1),np.take_along_axis(vals,srt,axis=1)

def labels(data,e2row,pids):
    prow={p:i for i,p in enumerate(pids)}
    gr=[]; hr=[]
    for g,h in zip(data['gold'],data['hard']):
        gr.append({prow[e2row[a]] for a in g if a in e2row and e2row[a] in prow})
        hr.append(prow.get(e2row.get(h),-1))
    return gr,hr

def calculate(rank,gr,hr,sizes):
    n=len(rank); ar=np.zeros(n,dtype=np.int32); hard=np.zeros(n,dtype=np.int32)
    for i,row in enumerate(rank):
        pos={int(c):j+1 for j,c in enumerate(row)}
        ar[i]=min((pos[x] for x in gr[i] if x in pos),default=0)
        hard[i]=pos.get(hr[i],0) if hr[i]>=0 else 0
    m={}
    for k in KS:
        m[f'any_H{k}']=float(np.mean((ar>0)&(ar<=k)))
        m[f'hard_H{k}']=float(np.mean((hard>0)&(hard<=k)))
    m['MRR_any@10']=float(np.mean(np.where((ar>0)&(ar<=10),1/np.maximum(ar,1),0)))
    m['MRR_hard@10']=float(np.mean(np.where((hard>0)&(hard<=10),1/np.maximum(hard,1),0)))
    es=sizes[rank[:,:5]].sum(1)
    m['top5_entities_mean']=float(es.mean()); m['top5_entities_median']=float(np.median(es)); m['top5_entities_p95']=float(np.quantile(es,.95))
    return m,ar,hard

def main():
    OUT.mkdir(exist_ok=True)
    splits,qemb=encode_queries()
    summary={'embedding_file':str(PART/'entity_embeddings.npy'),'grouping':'held-out rule_index internal split',
      'partition_build':'one shared MiniLM entity embedding matrix; RBConfiguration; resolution=3; seed=42; 2 iterations; same four-phase cleanup',
      'retrieval':'dense cosine to rebuilt centroids + BM25 over community names/aliases; per-query min-max; model×partition BM25 weight chosen by dev any-answer H@5',
      'weight_grid':WEIGHTS,'models':list(MODELS),'partitions':{}}
    for label in ('w','w2','w3','unweighted'):
        membership=OUT/f'clean_{label}_membership.pkl'
        idxdir=OUT/f'index_tmp_{label}'
        logpath=OUT/f'index_build_{label}.log'
        if idxdir.exists():
            shutil.rmtree(idxdir)
        log(f'building temporary index {label}')
        t=time.time()
        cmd=[sys.executable,str(INDEX_BUILDER),'--membership',str(membership),'--node-ids',str(PART/'graph_node_ids_w2.pkl'),
             '--embeddings',str(PART/'entity_embeddings.npy'),'--id-map',str(PART/'entity_id_to_row.pkl'),
             '--entities',str(PART/'entities.csv'),'--out-dir',str(idxdir),'--label',label,'--size-cap','8000']
        with open(logpath,'w') as lf: subprocess.run(cmd,check=True,stdout=lf,stderr=subprocess.STDOUT)
        cent=np.load(idxdir/'centroids.npy').astype(np.float32)
        pids=pickle.load(open(idxdir/'partition_ids.pkl','rb'))
        entity_map=pickle.load(open(idxdir/'entity_map.pkl','rb'))
        docs=pickle.load(open(idxdir/'bm25_docs.pkl','rb'))
        e2p={e:p for p,es in entity_map.items() for e in es}
        sizes=np.asarray([len(entity_map[p]) for p in pids],dtype=np.int32)
        bm=BM25Okapi(docs)
        bm_scores={s:np.asarray([bm.get_scores(tokenize(q)) for q in d['text']],dtype=np.float32) for s,d in splits.items()}
        del bm,docs,entity_map
        result={'n_partitions':len(pids),'index_build_seconds':round(time.time()-t,2),'models':{}}
        for model in MODELS:
            dense={s:qemb[model][s]@cent.T for s in ('dev','test')}
            dn={s:norm(dense[s]) for s in dense}; bn={s:norm(bm_scores[s]) for s in bm_scores}
            gr={s:None for s in splits}; hr={s:None for s in splits}
            for s in splits: gr[s],hr[s]=labels(splits[s],e2p,pids)
            best=(-1,None,None)
            for w in WEIGHTS:
                sc=(1-w)*dn['dev']+w*bn['dev']
                ix,_=topk(sc,5)
                mm,_,_=calculate(ix,gr['dev'],hr['dev'],sizes)
                if mm['any_H5']>best[0]: best=(mm['any_H5'],w,mm)
            w=best[1]; model_out={'selected_bm25_weight':w,'dev':best[2]}
            for s in ('dev','test'):
                sc=(1-w)*dn[s]+w*bn[s]
                ix,sv=topk(sc,100)
                mm,ar,hard=calculate(ix,gr[s],hr[s],sizes); model_out[s]=mm
                rows=[]
                for i in range(len(ix)):
                    rows.append({'id':int(splits[s]['ids'][i]),'rule_index':str(splits[s]['rules'][i]),
                      'answer_rank_top100':int(ar[i]),'hard_rank_top100':int(hard[i]),
                      'gold_candidate_coverage':bool(ar[i]),'hard_mapped':bool(hr[s][i]>=0),
                      'top100_partition_ids':[int(pids[j]) for j in ix[i]],
                      'top100_scores':[float(v) for v in sv[i]]})
                dest=OUT/'predictions'/label/model
                dest.mkdir(parents=True,exist_ok=True)
                json.dump(rows,open(dest/f'{s}.json','w'))
            result['models'][model]=model_out
            log(f'{label}/{model}: w_bm25={w:.2f} dev H5={best[0]:.4f} test H5={model_out["test"]["any_H5"]:.4f} hard={model_out["test"]["hard_H5"]:.4f}')
        summary['partitions'][label]=result
        json.dump(summary,open(OUT/'model_partition_results.json','w'),indent=2)
        # Retain compact centroids and metadata for reproducibility; discard
        # 300+ MB token/entity maps and BM25 pickle before the next variant.
        keep=OUT/f'index_{label}'; keep.mkdir(exist_ok=True)
        shutil.copy2(idxdir/'centroids.npy',keep/'centroids.npy')
        shutil.copy2(idxdir/'partition_ids.pkl',keep/'partition_ids.pkl')
        shutil.copy2(idxdir/'meta.json',keep/'meta.json')
        shutil.rmtree(idxdir)
        del cent,pids,e2p,sizes,bm_scores
    summary['warnings']=['Internal held-out split from BRINK train; rule groups are disjoint.','One training seed.','Community counts differ modestly after identical cleanup; report sizes and avoid interpreting weighting as fully granularity-controlled.','H@K is answer-partition retrieval, not evidence-path or generated-answer quality.']
    json.dump(summary,open(OUT/'model_partition_results.json','w'),indent=2)
    log('all model × partition runs complete')

def log(s): print('[model×partition]',s,flush=True)
if __name__=='__main__': main()
