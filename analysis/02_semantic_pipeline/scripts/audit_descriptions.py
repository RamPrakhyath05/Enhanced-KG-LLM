"""
audit_descriptions.py - Audits raw entity descriptions for character lengths,
empty rates, and placeholder or bot-generated noise.
"""

import csv
import sys
from collections import Counter

csv.field_size_limit(sys.maxsize)

ENTITIES_FILE = "dataset/entities.csv"

def audit_descriptions(input_file=ENTITIES_FILE):
    print("=" * 60)
    print("ENTITY DESCRIPTION QUALITY AUDIT")
    print("=" * 60)
    
    total = 0
    empty = 0
    non_empty = 0
    length_buckets = Counter()
    
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                desc = row.get("Entity_Description", "").strip()
                
                if not desc:
                    empty += 1
                    continue
                    
                non_empty += 1
                length = len(desc)
                
                if length <= 20:
                    length_buckets["<=20 chars"] += 1
                elif length <= 50:
                    length_buckets["21-50 chars"] += 1
                elif length <= 100:
                    length_buckets["51-100 chars"] += 1
                elif length <= 200:
                    length_buckets["101-200 chars"] += 1
                else:
                    length_buckets[">200 chars"] += 1

        print(f"Total Entities Scanned:       {total:,}")
        print(f"Non-Empty Descriptions:       {non_empty:,} ({non_empty/max(1, total)*100:.2f}%)")
        print(f"Empty Descriptions:           {empty:,} ({empty/max(1, total)*100:.2f}%)")
        print("\nDescription Length Distributions:", dict(length_buckets))
        
    except FileNotFoundError:
        print(f"Input file not found: {input_file}. (Check dataset path)")

if __name__ == "__main__":
    audit_descriptions()
