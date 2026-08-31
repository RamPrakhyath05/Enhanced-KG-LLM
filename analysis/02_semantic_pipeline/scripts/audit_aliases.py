"""
audit_aliases.py - Audits raw entity alias compositions and distributions.
Produces summary statistics on alias frequencies, duplicate aliases,
canonical-name repetitions, and outlier suspicious Wikipedia strings.
"""

import csv
import sys
from collections import Counter

csv.field_size_limit(sys.maxsize)

ENTITIES_FILE = "dataset/entities.csv"

def audit_aliases(input_file=ENTITIES_FILE):
    print("=" * 60)
    print("ALIAS COMPOSITION AUDIT")
    print("=" * 60)
    
    total_entities = 0
    total_aliases = 0
    duplicate_alias_entities = 0
    name_as_alias_entities = 0
    alias_counts = Counter()
    outliers = []

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_entities += 1
                name = row.get("Entity_Name", "").strip().lower()
                raw_aliases = row.get("Entity_Aliases", "")
                
                if not raw_aliases:
                    alias_counts[0] += 1
                    continue
                    
                aliases = [a.strip() for a in raw_aliases.split("|") if a.strip()]
                count = len(aliases)
                
                if count == 1:
                    alias_counts[1] += 1
                elif 2 <= count <= 5:
                    alias_counts["2-5"] += 1
                elif 6 <= count <= 10:
                    alias_counts["6-10"] += 1
                elif 11 <= count <= 25:
                    alias_counts["11-25"] += 1
                elif 26 <= count <= 50:
                    alias_counts["26-50"] += 1
                else:
                    alias_counts[">50"] += 1
                    outliers.append((count, row.get("Entity_ID"), row.get("Entity_Name")))

                total_aliases += count
                
                # Check for duplicates within entity
                if len(aliases) != len(set(a.lower() for a in aliases)):
                    duplicate_alias_entities += 1
                
                # Check for canonical name repetition
                if any(a.lower() == name for a in aliases):
                    name_as_alias_entities += 1
                    
        print(f"Total Entities Scanned:       {total_entities:,}")
        print(f"Total Raw Aliases:            {total_aliases:,}")
        print(f"Average Aliases / Entity:     {total_aliases / max(1, total_entities):.2f}")
        print(f"Entities with Duplicates:     {duplicate_alias_entities:,}")
        print(f"Entities with Name as Alias:  {name_as_alias_entities:,}")
        print("\nAlias Count Buckets:", dict(alias_counts))
        print(f"\nFound {len(outliers)} entities with > 50 aliases (outliers).")

    except FileNotFoundError:
        print(f"Input file not found: {input_file}. (Check dataset path)")

if __name__ == "__main__":
    audit_aliases()
