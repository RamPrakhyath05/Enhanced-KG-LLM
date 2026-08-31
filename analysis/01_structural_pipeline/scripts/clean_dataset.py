import csv
import os
import sys
import argparse

csv.field_size_limit(sys.maxsize)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dir",
    default="dataset",
    help="Directory containing entities.csv, relations.csv, and triples.csv"
)
args = parser.parse_args()

DATASET_DIR = args.data_dir

ENTITIES_FILE = "dataset/entities.csv"
RELATIONS_FILE = "dataset/relations.csv"
TRIPLES_FILE = "dataset/triples.csv"

OUTPUT_DIR = "dataset/cleaned"

CLEAN_ENTITIES_FILE = os.path.join(OUTPUT_DIR, "entities.csv")
CLEAN_RELATIONS_FILE = os.path.join(OUTPUT_DIR, "relations.csv")
CLEAN_TRIPLES_FILE = os.path.join(OUTPUT_DIR, "triples.csv")
REPORT_FILE = os.path.join(OUTPUT_DIR, "cleaning_report.txt")

def load_ids(filename, column):
    ids = set()

    with open(filename, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            ids.add(row[column])

    return ids


def copy_file(source, destination):
    with open(source, "r", encoding="utf-8") as src, \
         open(destination, "w", encoding="utf-8") as dst:

        for line in src:
            dst.write(line)


def main():
    print("=" * 60)
    print("WIKIDATA5M DATASET CLEANING")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # Load valid entity and relation IDs
    # ---------------------------------------------------------
    print("\nLoading entity IDs...")
    entity_ids = load_ids(ENTITIES_FILE, "Entity_ID")
    print(f"Entities:  {len(entity_ids):,}")

    print("Loading relation IDs...")
    relation_ids = load_ids(RELATIONS_FILE, "Relation_ID")
    print(f"Relations: {len(relation_ids):,}")

    # ---------------------------------------------------------
    # Copy entity and relation files unchanged
    # ---------------------------------------------------------
    print("\nCopying entities.csv...")
    copy_file(ENTITIES_FILE, CLEAN_ENTITIES_FILE)

    print("Copying relations.csv...")
    copy_file(RELATIONS_FILE, CLEAN_RELATIONS_FILE)

    # ---------------------------------------------------------
    # Clean triples
    # ---------------------------------------------------------
    print("\nCleaning triples...")

    total = 0
    kept = 0

    missing_source = 0
    missing_target = 0
    missing_relation = 0

    missing_source_and_target = 0

    self_loops = 0

    with open(TRIPLES_FILE, "r", encoding="utf-8", newline="") as src, \
         open(CLEAN_TRIPLES_FILE, "w", encoding="utf-8", newline="") as dst:

        reader = csv.DictReader(src)

        writer = csv.DictWriter(
            dst,
            fieldnames=[
                "Entity_1_ID",
                "Relation_ID",
                "Entity_2_ID",
            ],
        )

        writer.writeheader()

        for row in reader:
            total += 1

            source = row["Entity_1_ID"]
            relation = row["Relation_ID"]
            target = row["Entity_2_ID"]

            source_missing = source not in entity_ids
            target_missing = target not in entity_ids
            relation_missing = relation not in relation_ids

            if source_missing:
                missing_source += 1

            if target_missing:
                missing_target += 1

            if source_missing and target_missing:
                missing_source_and_target += 1

            if relation_missing:
                missing_relation += 1

            if source == target:
                self_loops += 1

            # Remove structurally invalid triples.
            if source_missing or target_missing or relation_missing:
                continue

            writer.writerow(row)
            kept += 1

    removed = total - kept

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------
    report = f"""WIKIDATA5M DATASET CLEANING REPORT
============================================================

INPUT
------------------------------------------------------------
Entities:        {len(entity_ids):,}
Relations:       {len(relation_ids):,}
Triples:         {total:,}

CLEANING RULES
------------------------------------------------------------
1. Remove triples with a missing source entity.
2. Remove triples with a missing target entity.
3. Remove triples with a missing relation.
4. Preserve self-loops.

MISSING ENTITY REFERENCES
------------------------------------------------------------
Missing source references:       {missing_source:,}
Missing target references:       {missing_target:,}
Both endpoints missing:          {missing_source_and_target:,}

MISSING RELATION REFERENCES
------------------------------------------------------------
Missing relation references:     {missing_relation:,}

SELF-LOOPS
------------------------------------------------------------
Self-loops preserved:            {self_loops:,}

RESULT
------------------------------------------------------------
Original triples:                {total:,}
Cleaned triples:                 {kept:,}
Removed triples:                 {removed:,}

Removal percentage:              {(removed / total * 100):.6f}%

OUTPUT
------------------------------------------------------------
{CLEAN_ENTITIES_FILE}
{CLEAN_RELATIONS_FILE}
{CLEAN_TRIPLES_FILE}
"""

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)

    print(f"\nOriginal triples:  {total:,}")
    print(f"Cleaned triples:   {kept:,}")
    print(f"Removed triples:   {removed:,}")

    print(f"\nMissing sources:   {missing_source:,}")
    print(f"Missing targets:   {missing_target:,}")
    print(f"Missing relations: {missing_relation:,}")
    print(f"Self-loops kept:   {self_loops:,}")

    print(f"\nOutput directory:")
    print(f"  {OUTPUT_DIR}/")

    print(f"\nReport:")
    print(f"  {REPORT_FILE}")


if __name__ == "__main__":
    main()
