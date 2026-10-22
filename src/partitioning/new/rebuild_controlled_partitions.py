#!/usr/bin/env python3
"""Rebuild cleaned graph-weight variants from one shared entity embedding set.

Uses the existing root MiniLM embeddings and structural graph topology. The
embedding file is memory-mapped and original project artifacts are read-only.
All new memberships and manifests are written below paper_eval/controlled.
"""
import json, os, pickle, sys, time
from pathlib import Path

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd

PART=Path('/home/pesu-rf/Documents/Capstone_Team165/partition')
OUT=Path(__file__).resolve().parent/'controlled'
OUT.mkdir(parents=True,exist_ok=True)
EMB=PART/'entity_embeddings.npy'
IDMAP=PART/'entity_id_to_row.pkl'
TRIPLES=PART/'structural_triples.csv'
NODES=PART/'graph_node_ids_w2.pkl'
TOPOLOGY=PART/'weighted_graph.pkl'
PWR={'w':1,'w2':2,'w3':3,'unweighted':0}

sys.path.insert(0,str(PART/'new_new'))
from step4_cleanup import split_oversized,merge_small,merge_isolated

def log(s): print(f'[controlled-partitions] {s}',flush=True)

def main():
    todo=sys.argv[1:] or list(PWR)
    unknown=set(todo)-set(PWR)
    if unknown: raise ValueError(f'unknown variants {unknown}')
    t0=time.time()
    log('loading shared embeddings by memory map and graph inputs')
    X=np.load(EMB,mmap_mode='r')
    with open(IDMAP,'rb') as f: id2row=pickle.load(f)
    with open(NODES,'rb') as f: nodes=pickle.load(f)
    g=ig.Graph.Read_Pickle(str(TOPOLOGY))
    if len(nodes)!=g.vcount(): raise RuntimeError(f'node/graph mismatch {len(nodes)} vs {g.vcount()}')
    node2idx={eid:i for i,eid in enumerate(nodes)}
    node_rows=np.asarray([id2row[eid] for eid in nodes],dtype=np.int64)
    log(f'embeddings={X.shape} graph={g.vcount():,} nodes/{g.ecount():,} unique edges')

    log('reading structural triples and mapping edge endpoints')
    triples=pd.read_csv(TRIPLES,usecols=['Entity_1_ID','Entity_2_ID'])
    src=triples['Entity_1_ID'].map(node2idx).to_numpy(dtype=np.int32)
    dst=triples['Entity_2_ID'].map(node2idx).to_numpy(dtype=np.int32)
    if (src<0).any() or (dst<0).any(): raise RuntimeError('structural endpoint missing from node list')
    del triples
    log(f'raw structural edges={len(src):,}')

    log('mapping raw edges onto the shared simplified graph topology')
    ep=np.asarray(g.get_edgelist(),dtype=np.int32)
    n=g.vcount()
    edge_codes=np.minimum(ep[:,0],ep[:,1]).astype(np.int64)*n+np.maximum(ep[:,0],ep[:,1]).astype(np.int64)
    edge_order=np.argsort(edge_codes)
    sorted_codes=edge_codes[edge_order]
    raw_codes=np.minimum(src,dst).astype(np.int64)*n+np.maximum(src,dst).astype(np.int64)
    pos=np.searchsorted(sorted_codes,raw_codes)
    valid=pos<len(sorted_codes)
    valid[valid] &= sorted_codes[pos[valid]]==raw_codes[valid]
    edge_idx=np.full(len(src),-1,dtype=np.int32)
    edge_idx[valid]=edge_order[pos[valid]].astype(np.int32)
    if valid.sum()!=len(src):
        # Self-loops are removed by simplify; report and skip those rows.
        log(f'warning: {len(src)-int(valid.sum()):,} raw rows have no simplified edge (usually self-loops)')
    if valid.mean()<.99: raise RuntimeError(f'only {valid.mean():.4%} raw rows mapped to graph edges')

    log('computing shared raw cosine weights (one pass over triples)')
    cos=np.empty(len(src),dtype=np.float32)
    chunk=250_000
    for start in range(0,len(src),chunk):
        end=min(start+chunk,len(src))
        a=np.asarray(X[node_rows[src[start:end]]],dtype=np.float32)
        b=np.asarray(X[node_rows[dst[start:end]]],dtype=np.float32)
        cos[start:end]=np.einsum('ij,ij->i',a,b,optimize=True)
        if start%(chunk*10)==0: log(f'  cosine {end:,}/{len(src):,}')
    shift=abs(float(cos[valid].min()))+1e-4
    log(f'cosine min={cos[valid].min():.6f} max={cos[valid].max():.6f} shift={shift:.6f}')
    del ep,edge_codes,edge_order,sorted_codes,raw_codes,pos,valid,src,dst,node2idx,node_rows

    manifest={'embedding_file':str(EMB),'embedding_shape':list(X.shape),'edge_source':str(TRIPLES),
              'common_cosine_shift':shift,'leiden_objective':'RBConfigurationVertexPartition',
              'resolution':3.0,'iterations':2,'seed':42,
              'cleanup':'same A/B/C/D implementation and parameters from partition/new_new/step4_cleanup.py',
              'variants':{}}
    for label,power in PWR.items():
        if label not in todo: continue
        start=time.time(); log(f'building {label} (power={power})')
        if power==0:
            agg=np.bincount(edge_idx[edge_idx>=0],minlength=g.ecount()).astype(np.float32)
            # Unweighted means one unit per unique undirected edge.
            agg.fill(1.0)
        else:
            raw=np.power(cos+shift,power,dtype=np.float32)
            good=edge_idx>=0
            agg=np.bincount(edge_idx[good],weights=raw[good],minlength=g.ecount()).astype(np.float32)
            del raw,good
        g.es['weight']=agg.tolist()
        del agg
        log(f'Leiden RBConfiguration resolution=3, seed=42, n_iterations=2: {label}')
        partition=leidenalg.find_partition(g,leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=3.0,weights='weight',n_iterations=2,seed=42)
        membership=partition.membership
        raw_n=len(set(membership)); del partition
        raw_path=OUT/f'raw_{label}_membership.pkl'
        with open(raw_path,'wb') as f: pickle.dump(membership,f,protocol=pickle.HIGHEST_PROTOCOL)
        log(f'{label}: raw communities={raw_n:,}; cleanup A')
        membership=split_oversized(membership,nodes,X,id2row,1200)
        log(f'{label}: cleanup B')
        membership=merge_small(membership,g,nodes,X,id2row)
        log(f'{label}: cleanup C')
        membership=merge_isolated(membership,g,nodes,X,id2row)
        log(f'{label}: cleanup D')
        membership=split_oversized(membership,nodes,X,id2row,1500)
        sizes=np.bincount(np.asarray(membership,dtype=np.int64)); sizes=sizes[sizes>0]
        out=OUT/f'clean_{label}_membership.pkl'
        with open(out,'wb') as f: pickle.dump(membership,f,protocol=pickle.HIGHEST_PROTOCOL)
        manifest['variants'][label]={'power':power,'raw_communities':raw_n,'clean_communities':int(len(sizes)),
            'graph_node_community_size_min':int(sizes.min()),'median':float(np.median(sizes)),
            'max':int(sizes.max()),'clean_membership':str(out),'raw_membership':str(raw_path),
            'seconds':round(time.time()-start,1)}
        json.dump(manifest,open(OUT/'build_manifest.json','w'),indent=2)
        log(f'{label}: cleaned={len(sizes):,}, size min/median/max={sizes.min()}/{np.median(sizes):.0f}/{sizes.max()}, elapsed={time.time()-start:.1f}s')
        # Avoid retaining stale membership arrays across full-graph variants.
        del membership,sizes
    manifest['total_seconds']=round(time.time()-t0,1)
    json.dump(manifest,open(OUT/'build_manifest.json','w'),indent=2)
    log(f'completed variants={list(manifest["variants"])} total={manifest["total_seconds"]:.1f}s')

if __name__=='__main__': main()
