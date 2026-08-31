import csv
import os
import sys

csv.field_size_limit(sys.maxsize)

DATASET_DIR = "dataset/cleaned"

ENTITIES_FILE = os.path.join(DATASET_DIR, "entities.csv")
RELATIONS_FILE = os.path.join(DATASET_DIR, "relations.csv")
TRIPLES_FILE = os.path.join(DATASET_DIR, "triples.csv")


def load_ids(filepath, column):
    ids = set()

    with open(filepath, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            value = row[column].strip()

            if value:
                ids.add(value)

    return ids


def main():
    print("=" * 60)
    print("FINAL STRUCTURAL CHECK")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load entities and relations
    # ---------------------------------------------------------

    print("\nLoading entity IDs...")
    entity_ids = load_ids(ENTITIES_FILE, "Entity_ID")

    print("Loading relation IDs...")
    relation_ids = load_ids(RELATIONS_FILE, "Relation_ID")

    # ---------------------------------------------------------
    # Scan triples
    # ---------------------------------------------------------

    print("\nScanning triples...")

    referenced_entities = set()
    used_relations = set()

    total_triples = 0
    duplicate_triples = 0
    missing_source = 0
    missing_target = 0
    missing_relation = 0
    self_loops = 0

    seen_triples = set()

    with open(TRIPLES_FILE, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            source = row["Entity_1_ID"].strip()
            relation = row["Relation_ID"].strip()
            target = row["Entity_2_ID"].strip()

            total_triples += 1

            triple = (source, relation, target)

            if triple in seen_triples:
                duplicate_triples += 1
            else:
                seen_triples.add(triple)

            referenced_entities.add(source)
            referenced_entities.add(target)
            used_relations.add(relation)

            if source not in entity_ids:
                missing_source += 1

            if target not in entity_ids:
                missing_target += 1

            if relation not in relation_ids:
                missing_relation += 1

            if source == target:
                self_loops += 1

    # ---------------------------------------------------------
    # Derived statistics
    # ---------------------------------------------------------

    isolated_entities = entity_ids - referenced_entities
    unused_relations = relation_ids - used_relations

    malformed_entity_ids = [
        entity_id
        for entity_id in entity_ids
        if not entity_id.startswith("Q")
    ]

    malformed_relation_ids = [
        relation_id
        for relation_id in relation_ids
        if not relation_id.startswith("P")
    ]

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print("\n=== DATASET COUNTS ===")
    print(f"Entities:              {len(entity_ids):,}")
    print(f"Relations:             {len(relation_ids):,}")
    print(f"Triples:               {total_triples:,}")

    print("\n=== GRAPH INTEGRITY ===")
    print(f"Referenced entities:   {len(referenced_entities):,}")
    print(f"Isolated entities:     {len(isolated_entities):,}")
    print(f"Used relations:        {len(used_relations):,}")
    print(f"Unused relations:      {len(unused_relations):,}")
    print(f"Duplicate triples:     {duplicate_triples:,}")
    print(f"Missing source:        {missing_source:,}")
    print(f"Missing target:        {missing_target:,}")
    print(f"Missing relation:      {missing_relation:,}")
    print(f"Self-loops:            {self_loops:,}")

    print("\n=== ID FORMAT ===")
    print(f"Malformed entity IDs:  {len(malformed_entity_ids):,}")
    print(f"Malformed relation IDs:{len(malformed_relation_ids):,}")

    # ---------------------------------------------------------
    # Verdict
    # ---------------------------------------------------------

    problems = []

    if duplicate_triples:
        problems.append("duplicate triples")

    if missing_source:
        problems.append("missing source entities")

    if missing_target:
        problems.append("missing target entities")

    if missing_relation:
        problems.append("missing relations")

    if malformed_entity_ids:
        problems.append("malformed entity IDs")

    if malformed_relation_ids:
        problems.append("malformed relation IDs")

    print("\n=== VERDICT ===")

    if problems:
        print("STRUCTURAL CHECK FAILED")

        for problem in problems:
            print(f"- {problem}")

    else:
        print("STRUCTURAL CLEANING CHECK PASSED")
        print("Ready for semantic cleaning.")


if __name__ == "__main__":
    main()
