import csv
from collections import Counter

DATASET_DIR = "dataset"

ENTITIES_FILE = f"{DATASET_DIR}/entities.csv"
RELATIONS_FILE = f"{DATASET_DIR}/relations.csv"
TRIPLES_FILE = f"{DATASET_DIR}/triples.csv"

# Python's default CSV field limit is too small for some Wikidata descriptions.
csv.field_size_limit(10**9)


def load_ids(filepath, column):
    ids = set()

    with open(filepath, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            ids.add(row[column])

    return ids


def main():
    print("=" * 60)
    print("INVALID TRIPLE ANALYSIS")
    print("=" * 60)

    print("\nLoading entity and relation IDs...")

    entity_ids = load_ids(ENTITIES_FILE, "Entity_ID")
    relation_ids = load_ids(RELATIONS_FILE, "Relation_ID")

    print(f"Entities loaded:  {len(entity_ids):,}")
    print(f"Relations loaded: {len(relation_ids):,}")

    missing_sources = Counter()
    missing_targets = Counter()
    missing_relations = Counter()
    self_loops = Counter()

    examples_missing_source = []
    examples_missing_target = []
    examples_missing_relation = []
    examples_self_loop = []

    print("\nScanning triples...")

    with open(TRIPLES_FILE, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            source = row["Entity_1_ID"]
            relation = row["Relation_ID"]
            target = row["Entity_2_ID"]

            # Missing source
            if source not in entity_ids:
                missing_sources[source] += 1

                if len(examples_missing_source) < 10:
                    examples_missing_source.append(
                        (source, relation, target)
                    )

            # Missing target
            if target not in entity_ids:
                missing_targets[target] += 1

                if len(examples_missing_target) < 10:
                    examples_missing_target.append(
                        (source, relation, target)
                    )

            # Missing relation
            if relation not in relation_ids:
                missing_relations[relation] += 1

                if len(examples_missing_relation) < 10:
                    examples_missing_relation.append(
                        (source, relation, target)
                    )

            # Self-loop
            if source == target:
                self_loops[(source, relation)] += 1

                if len(examples_self_loop) < 20:
                    examples_self_loop.append(
                        (source, relation, target)
                    )

    # ---------------------------------------------------------
    # Missing sources
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("MISSING SOURCE ENTITIES")
    print("=" * 60)

    print(f"Invalid triples:       {sum(missing_sources.values()):,}")
    print(f"Unique missing IDs:    {len(missing_sources):,}")

    if missing_sources:
        print("\nMost frequent missing source IDs:")

        for entity_id, count in missing_sources.most_common(10):
            print(f"  {entity_id}: {count:,} triples")

        print("\nExamples:")

        for source, relation, target in examples_missing_source:
            print(f"  {source} --{relation}--> {target}")

    # ---------------------------------------------------------
    # Missing targets
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("MISSING TARGET ENTITIES")
    print("=" * 60)

    print(f"Invalid triples:       {sum(missing_targets.values()):,}")
    print(f"Unique missing IDs:    {len(missing_targets):,}")

    if missing_targets:
        print("\nMost frequent missing target IDs:")

        for entity_id, count in missing_targets.most_common(10):
            print(f"  {entity_id}: {count:,} triples")

        print("\nExamples:")

        for source, relation, target in examples_missing_target:
            print(f"  {source} --{relation}--> {target}")

    # ---------------------------------------------------------
    # Missing relations
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("MISSING RELATIONS")
    print("=" * 60)

    print(f"Invalid triples:       {sum(missing_relations.values()):,}")
    print(f"Unique missing IDs:    {len(missing_relations):,}")

    if missing_relations:
        print("\nMissing relation IDs:")

        for relation_id, count in missing_relations.most_common():
            print(f"  {relation_id}: {count:,} triples")

        print("\nExamples:")

        for source, relation, target in examples_missing_relation:
            print(f"  {source} --{relation}--> {target}")

    # ---------------------------------------------------------
    # Self loops
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("SELF-LOOPS")
    print("=" * 60)

    print(f"Self-loop triples:     {sum(self_loops.values()):,}")
    print(f"Unique entity/relation pairs: {len(self_loops):,}")

    if self_loops:
        print("\nExamples:")

        for source, relation, target in examples_self_loop:
            print(f"  {source} --{relation}--> {target}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
