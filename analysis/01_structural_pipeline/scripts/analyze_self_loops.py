import csv
import os
import sys
from collections import Counter

csv.field_size_limit(sys.maxsize)

DATASET_DIR = "dataset/cleaned"

TRIPLES_FILE = os.path.join(DATASET_DIR, "triples.csv")
RELATIONS_FILE = os.path.join(DATASET_DIR, "relations.csv")


def load_relations():
    relations = {}

    with open(RELATIONS_FILE, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            relations[row["Relation_ID"]] = row["Relation_Name"]

    return relations


def main():
    print("=" * 60)
    print("SELF-LOOP ANALYSIS")
    print("=" * 60)

    relations = load_relations()

    self_loops = []
    relation_counts = Counter()

    with open(TRIPLES_FILE, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            source = row["Entity_1_ID"]
            relation = row["Relation_ID"]
            target = row["Entity_2_ID"]

            if source == target:
                self_loops.append((source, relation, target))
                relation_counts[relation] += 1

    print(f"\nTotal self-loops: {len(self_loops):,}")

    print("\n" + "=" * 60)
    print("SELF-LOOPS BY RELATION")
    print("=" * 60)

    for relation, count in relation_counts.most_common():
        name = relations.get(relation, "[UNKNOWN]")
        print(f"{relation:<10} {count:>5}  {name}")

    print("\n" + "=" * 60)
    print("ALL SELF-LOOPS")
    print("=" * 60)

    for source, relation, target in self_loops:
        name = relations.get(relation, "[UNKNOWN]")
        print(f"{source} --{relation} ({name})--> {target}")


if __name__ == "__main__":
    main()
