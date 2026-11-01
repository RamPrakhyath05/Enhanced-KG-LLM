"""Parsers for the pinned local BRINK TSV annotation format."""
from __future__ import annotations
import ast

def parse_tsv_list(value):
    if not value: return []
    try:
        parsed=ast.literal_eval(value)
        return list(parsed) if isinstance(parsed,(list,tuple,set)) else [parsed]
    except (ValueError,SyntaxError):
        return [value]

def parse_grounding_paths(value):
    """Convert BRINK grounding strings to paths of canonical h|r|t edge IDs.

    BRINK annotations use premise triples separated by `|`, followed optionally
    by `=>` and the held-out conclusion. Only the listed premise triples form
    the retrieved-evidence target; the conclusion is deliberately excluded.
    """
    paths=[]
    for annotation in parse_tsv_list(value):
        premises=str(annotation).split('=>',1)[0]
        edges=[]
        for item in premises.split('|'):
            parts=item.strip().split()
            if len(parts)>=3:
                edges.append('|'.join(parts[:3]))
        if edges: paths.append(edges)
    return paths
