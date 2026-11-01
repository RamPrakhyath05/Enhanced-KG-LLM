#!/usr/bin/env python3
"""Aggregate generic per-question benchmark JSONL with score_metrics.py."""
import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from metrics.score_metrics import score_jsonl
p=argparse.ArgumentParser()
p.add_argument('--input',required=True,help='JSONL; see metrics/protocols/benchmark_protocol.md')
p.add_argument('--output',required=True)
a=p.parse_args()
print(score_jsonl(a.input,a.output))
