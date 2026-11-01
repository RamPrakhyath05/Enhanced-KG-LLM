#!/usr/bin/env python3
"""Capture real end-to-end pipeline outputs on BRINK; requires live Neo4j + Ollama.

No gold topic entities are given to the system. Gold answers/seeds are output only
for post-hoc scoring. Offline/mock mode is prohibited.
"""
from __future__ import annotations
import argparse, ast, csv, json, sys, time, platform, subprocess, resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.generation.pipeline import EnhancedKGRAGPipeline
from src.query_subgraph.subgraph_retriever import tokenize
from metrics.data_utils import parse_tsv_list, parse_grounding_paths

def parse_list(x): return parse_tsv_list(x)
def git_commit():
 try: return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
 except Exception: return None

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--questions',required=True); p.add_argument('--hard-answers',required=True)
 p.add_argument('--split',choices=['train','val','test'],default='val')
 p.add_argument('--subg-dir',default=str(ROOT/'data/subg')); p.add_argument('--out',required=True)
 p.add_argument('--model',default=None); p.add_argument('--top-k',type=int,default=5); p.add_argument('--limit',type=int)
 p.add_argument('--filter-mode',choices=['auto','partition_id','entity_id'],default='auto'); p.add_argument('--allow-exploratory-test',action='store_true')
 a=p.parse_args()
 if a.split=='test' and not a.allow_exploratory_test: p.error('BRINK test has prior experimental reuse; pass --allow-exploratory-test to label the run exploratory.')
 out_path=Path(a.out)
 if out_path.exists(): raise SystemExit(f'Refusing to overwrite existing output: {out_path}')
 with open(a.questions,encoding='utf8') as f: comp={r['id']:r for r in csv.DictReader(f,delimiter='\t')}
 with open(a.hard_answers,encoding='utf8') as f: hard={r['id']:r for r in csv.DictReader(f,delimiter='\t')}
 ids=sorted(set(comp)&set(hard),key=int)
 if a.limit: ids=ids[:a.limit]
 pipeline=EnhancedKGRAGPipeline(subg_dir=a.subg_dir,filter_mode=a.filter_mode,allow_offline_fallback=False)
 if a.model: pipeline.ollama.model_name=a.model
 # Establish a live graph connection before a run and explicitly measure cold setup.
 start=time.perf_counter(); pipeline.db_client.run_query('RETURN 1 AS ping')
 if not pipeline.ollama.is_available(): raise SystemExit('Ollama is unavailable; no mock fallback is allowed for this benchmark.')
 pipeline.retriever.load()
 if ids:
  warm_question=comp[ids[0]]['question']
  pipeline.retriever.model.encode([warm_question],normalize_embeddings=True)
  pipeline.retriever.bm25.get_scores(tokenize(warm_question))
 startup_ms=(time.perf_counter()-start)*1000
 out_path.parent.mkdir(parents=True,exist_ok=True); n_failed=0
 with out_path.open('w',encoding='utf8') as out:
  for n,qid in enumerate(ids,1):
   row=comp[qid]; start_q=time.perf_counter()
   try:
    result=pipeline.run(row['question'],top_k_partitions=a.top_k)
    answer=result.grounded_answer
    ranked_seeds=[s.entity_id for s in result.resolved_seeds]
    gold_seeds=[str(x) for x in parse_list(row.get('q_entity',''))]
    rec={'id':qid,'question':row['question'],'rule_index':row.get('rule_index'),'gold_answer_ids':[str(x) for x in parse_list(row.get('answer',''))],
      'hard_answer':(parse_list(hard[qid].get('answer','')) or [None])[0], 'gold_seed_entity_ids':gold_seeds,
      'resolved_seed_entity_ids':ranked_seeds,'seed_entity_ids_used':list(result.seed_entity_ids),
      'prediction':answer.answer,'predicted_candidate_ids':list(result.bundle.candidate_answer_ids),
      'selected_partition_ids':list(result.selected_partition_ids),
      'evidence_triples':[c['edge_id'] for c in result.bundle.citations],
      'retrieved_evidence':[c['edge_id'] for c in result.bundle.citations],
      'gold_evidence_paths':parse_grounding_paths(row.get('grounding','')),
      'gold_evidence':sorted({edge for path in parse_grounding_paths(row.get('grounding','')) for edge in path}),
      'citations':list(answer.citations_used),
      'supported_citations':[c['label'] for c in result.bundle.citations],
      'stage_timings_ms':result.timings_ms,'latency_ms':(time.perf_counter()-start_q)*1000,
      'edges_examined':result.bundle.traversal_metadata.get('edges_examined'),
      'adjacency_entries_fetched':result.bundle.traversal_metadata.get('adjacency_entries_fetched'),
      'candidate_edges_scored':result.bundle.traversal_metadata.get('candidate_edges_scored'),
      'scoring_operations':result.bundle.traversal_metadata.get('scoring_operations'),
      'context_tokens':len(result.bundle.context.split()),'llm_model':answer.model_name,
      'llm_is_mock':answer.is_mock,'llm_was_skipped':answer.was_skipped,
      'error':None}
   except Exception as exc:
    n_failed+=1
    rec={'id':qid,'question':row['question'],'rule_index':row.get('rule_index'),'gold_answer_ids':[str(x) for x in parse_list(row.get('answer',''))],
      'gold_seed_entity_ids':[str(x) for x in parse_list(row.get('q_entity',''))],
      'hard_answer':(parse_list(hard[qid].get('answer','')) or [None])[0],
      'error':f'{type(exc).__name__}: {exc}','latency_ms':(time.perf_counter()-start_q)*1000}
   out.write(json.dumps(rec,ensure_ascii=False)+'\n'); out.flush()
   if n%25==0: print(f'completed {n}/{len(ids)}',flush=True)
 manifest={'benchmark':'BRINK-Wikidata5m','split':a.split,'n_questions':len(ids),'n_failed':n_failed,
   'query_seed_policy':'automatic; gold topic IDs used only for post-hoc linking metrics',
   'index_dir':str(Path(a.subg_dir).resolve()),'ollama_model':pipeline.ollama.model_name,
   'startup_and_warmup_ms':startup_ms,'cold_start_latency_separately_recorded':False,
   'git_commit':git_commit(),'python':platform.python_version(),'platform':platform.platform(),
   'offline_fallback':False,'process_peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
   'per_question_output':str(out_path.resolve())}
 out_path.with_suffix(out_path.suffix+'.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
 print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
