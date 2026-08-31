"""
audit_relations.py - Audits all 825 relations in Wikidata5M.
Calculates triple usage frequencies, identifies high-degree hub relations
(P31, P17, P27, etc.), checks for unused relations, and detects self-loops.
"""

import csv
import sys
from collections import Counter

csv.field_size_limit(sys.maxsize)

RELATIONS_FILE = "dataset/relations.csv"
TRIPLES_FILE = "dataset/triples.csv"

def audit_relations(rel_file=RELATIONS_FILE, triples_file=TRIPLES_FILE):
    print("=" * 60)
    print("RELATION USAGE & HUB FREQUENCY AUDIT")
    print("=" * 60)
    
    relation_names = {}
    relation_counts = Counter()
    self_loops = 0
    total_triples = 0
    
    try:
        with open(rel_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                relation_names[row["Relation_ID"]] = row.get("Relation_Name", "")

        print(f"Total Unique Relations Loaded: {len(relation_names):,}")

        with open(triples_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_triples += 1
                rel = row.get("Relation_ID")
                head = row.get("Head_ID")
                tail = row.get("Tail_ID")
                
                relation_counts[rel] += 1
                if head == tail:
                    self_loops += 1

        print(f"Total Triples Scanned:        {total_triples:,}")
        print(f"Relations Used in Triples:    {len(relation_counts):,}")
        print(f"Unused Relations:             {len(relation_names) - len(relation_counts):,}")
        print(f"Self-loops Detected:          {self_loops:,}")
        
        print("\n--- TOP 10 MOST FREQUENT HUB RELATIONS ---")
        for rel_id, count in relation_counts.most_common(10):
            name = relation_names.get(rel_id, "Unknown")
            print(f"  {rel_id:6s} | {name:25s} | {count:10,} triples ({count/total_triples*100:.2f}%)")

    except FileNotFoundError as e:
        print(f"Dataset file not found: {e}. (Check dataset path)")

if __name__ == "__main__":
    audit_relations()
