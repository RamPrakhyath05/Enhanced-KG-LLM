import csv
import os
import sys
from collections import Counter


csv.field_size_limit(sys.maxsize)

# ============================================================
# INPUT FILES
# ============================================================

ORIGINAL_ENTITIES = "dataset/entities.csv"
FINAL_ENTITIES = "dataset/final/entities.csv"

ORIGINAL_RELATIONS = "dataset/relations.csv"
FINAL_RELATIONS = "dataset/final/relations.csv"

ORIGINAL_TRIPLES = "dataset/triples.csv"
FINAL_TRIPLES = "dataset/final/triples.csv"

REPORT_FILE = "dataset/semantic/analysis/original_vs_final_report.txt"


# ============================================================
# HELPERS
# ============================================================

def normalize(value):
    if value is None:
        return ""
    return value.strip()


def pct(part, total):
    if total == 0:
        return 0.0
    return (part / total) * 100


def write_header(report, title):
    report.write("\n")
    report.write("=" * 70 + "\n")
    report.write(title + "\n")
    report.write("=" * 70 + "\n")


# ============================================================
# ENTITY COMPARISON
# ============================================================

def load_entities(path, label):
    entities = {}

    total = 0
    empty_names = 0
    empty_aliases = 0
    empty_descriptions = 0
    duplicate_ids = 0

    print(f"Loading {label} entities...")

    with open(
        path,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        required = {
            "Entity_ID",
            "Entity_Name",
            "Entity_Aliases",
            "Entity_Description",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"{path} is missing columns: {sorted(missing)}"
            )

        for row in reader:
            total += 1

            entity_id = normalize(row["Entity_ID"])

            if entity_id in entities:
                duplicate_ids += 1

            entities[entity_id] = {
                "name": normalize(row["Entity_Name"]),
                "aliases": normalize(row["Entity_Aliases"]),
                "description": normalize(
                    row["Entity_Description"]
                ),
            }

            if not entities[entity_id]["name"]:
                empty_names += 1

            if not entities[entity_id]["aliases"]:
                empty_aliases += 1

            if not entities[entity_id]["description"]:
                empty_descriptions += 1

            if total % 500_000 == 0:
                print(
                    f"  {label}: {total:,} entities"
                )

    print(
        f"  {label}: {total:,} entities loaded"
    )

    return {
        "data": entities,
        "total": total,
        "empty_names": empty_names,
        "empty_aliases": empty_aliases,
        "empty_descriptions": empty_descriptions,
        "duplicate_ids": duplicate_ids,
    }


def compare_entities(original, final, report):

    write_header(
        report,
        "ENTITY COMPARISON",
    )

    old = original["data"]
    new = final["data"]

    old_ids = set(old)
    new_ids = set(new)

    added_ids = new_ids - old_ids
    removed_ids = old_ids - new_ids
    common_ids = old_ids & new_ids

    names_changed = 0
    aliases_changed = 0
    descriptions_added = 0
    descriptions_removed = 0
    descriptions_changed = 0
    unchanged_descriptions = 0

    for entity_id in common_ids:

        old_row = old[entity_id]
        new_row = new[entity_id]

        if old_row["name"] != new_row["name"]:
            names_changed += 1

        if old_row["aliases"] != new_row["aliases"]:
            aliases_changed += 1

        old_desc = old_row["description"]
        new_desc = new_row["description"]

        if not old_desc and new_desc:
            descriptions_added += 1

        elif old_desc and not new_desc:
            descriptions_removed += 1

        elif old_desc and new_desc:
            if old_desc != new_desc:
                descriptions_changed += 1
            else:
                unchanged_descriptions += 1

    report.write(
        f"Original entity rows:              "
        f"{original['total']:,}\n"
    )

    report.write(
        f"Final entity rows:                 "
        f"{final['total']:,}\n"
    )

    report.write(
        f"Entity rows added:                 "
        f"{len(added_ids):,}\n"
    )

    report.write(
        f"Entity rows removed:               "
        f"{len(removed_ids):,}\n"
    )

    report.write(
        f"Common Entity_IDs:                 "
        f"{len(common_ids):,}\n"
    )

    report.write(
        f"Duplicate IDs - original:          "
        f"{original['duplicate_ids']:,}\n"
    )

    report.write(
        f"Duplicate IDs - final:             "
        f"{final['duplicate_ids']:,}\n"
    )

    report.write(
        f"Names changed:                     "
        f"{names_changed:,}\n"
    )

    report.write(
        f"Aliases changed:                   "
        f"{aliases_changed:,}\n"
    )

    report.write(
        f"Descriptions added:                "
        f"{descriptions_added:,}\n"
    )

    report.write(
        f"Descriptions removed:              "
        f"{descriptions_removed:,}\n"
    )

    report.write(
        f"Descriptions changed:              "
        f"{descriptions_changed:,}\n"
    )

    report.write(
        f"Descriptions unchanged:            "
        f"{unchanged_descriptions:,}\n"
    )

    report.write(
        f"Original empty descriptions:       "
        f"{original['empty_descriptions']:,}\n"
    )

    report.write(
        f"Final empty descriptions:          "
        f"{final['empty_descriptions']:,}\n"
    )

    report.write(
        f"Original non-empty descriptions:   "
        f"{original['total'] - original['empty_descriptions']:,}\n"
    )

    report.write(
        f"Final non-empty descriptions:      "
        f"{final['total'] - final['empty_descriptions']:,}\n"
    )

    report.write("\nDESCRIPTION COVERAGE\n")

    old_nonempty = (
        original["total"]
        - original["empty_descriptions"]
    )

    new_nonempty = (
        final["total"]
        - final["empty_descriptions"]
    )

    report.write(
        f"Original coverage:                 "
        f"{pct(old_nonempty, original['total']):.2f}%\n"
    )

    report.write(
        f"Final coverage:                    "
        f"{pct(new_nonempty, final['total']):.2f}%\n"
    )

    report.write(
        f"Coverage change:                   "
        f"{pct(new_nonempty, final['total']) - pct(old_nonempty, original['total']):+.2f} percentage points\n"
    )

    return {
        "added": len(added_ids),
        "removed": len(removed_ids),
        "common": len(common_ids),
        "names_changed": names_changed,
        "aliases_changed": aliases_changed,
        "descriptions_added": descriptions_added,
        "descriptions_removed": descriptions_removed,
        "descriptions_changed": descriptions_changed,
    }


# ============================================================
# RELATION COMPARISON
# ============================================================

def load_relations(path, label):

    relations = {}

    total = 0
    duplicate_ids = 0
    duplicate_names = 0

    names_seen = set()

    print(f"Loading {label} relations...")

    with open(
        path,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        required = {
            "Relation_ID",
            "Relation_Name",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"{path} is missing columns: {sorted(missing)}"
            )

        for row in reader:

            total += 1

            relation_id = normalize(
                row["Relation_ID"]
            )

            relation_name = normalize(
                row["Relation_Name"]
            )

            if relation_id in relations:
                duplicate_ids += 1

            if relation_name in names_seen:
                duplicate_names += 1

            names_seen.add(relation_name)

            relations[relation_id] = relation_name

    print(
        f"  {label}: {total:,} relations loaded"
    )

    return {
        "data": relations,
        "total": total,
        "duplicate_ids": duplicate_ids,
        "duplicate_names": duplicate_names,
    }


def compare_relations(original, final, report):

    write_header(
        report,
        "RELATION COMPARISON",
    )

    old = original["data"]
    new = final["data"]

    old_ids = set(old)
    new_ids = set(new)

    added_ids = new_ids - old_ids
    removed_ids = old_ids - new_ids
    common_ids = old_ids & new_ids

    names_changed = 0

    for relation_id in common_ids:
        if old[relation_id] != new[relation_id]:
            names_changed += 1

    report.write(
        f"Original relation rows:            "
        f"{original['total']:,}\n"
    )

    report.write(
        f"Final relation rows:               "
        f"{final['total']:,}\n"
    )

    report.write(
        f"Relation rows added:               "
        f"{len(added_ids):,}\n"
    )

    report.write(
        f"Relation rows removed:             "
        f"{len(removed_ids):,}\n"
    )

    report.write(
        f"Common Relation_IDs:               "
        f"{len(common_ids):,}\n"
    )

    report.write(
        f"Relation names changed:            "
        f"{names_changed:,}\n"
    )

    report.write(
        f"Duplicate IDs - original:          "
        f"{original['duplicate_ids']:,}\n"
    )

    report.write(
        f"Duplicate IDs - final:             "
        f"{final['duplicate_ids']:,}\n"
    )

    report.write(
        f"Duplicate names - original:        "
        f"{original['duplicate_names']:,}\n"
    )

    report.write(
        f"Duplicate names - final:           "
        f"{final['duplicate_names']:,}\n"
    )

    return {
        "added": len(added_ids),
        "removed": len(removed_ids),
        "common": len(common_ids),
        "names_changed": names_changed,
    }


# ============================================================
# TRIPLE COMPARISON
# ============================================================

def load_triples(path, label):

    triples = set()

    total = 0
    malformed = 0
    self_loops = 0
    duplicate_triples = 0

    print(f"Loading {label} triples...")

    with open(
        path,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        required = {
            "Entity_1_ID",
            "Relation_ID",
            "Entity_2_ID",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"{path} is missing columns: {sorted(missing)}"
            )

        for row in reader:

            total += 1

            a = normalize(row["Entity_1_ID"])
            r = normalize(row["Relation_ID"])
            b = normalize(row["Entity_2_ID"])

            if not a or not r or not b:
                malformed += 1
                continue

            triple = (a, r, b)

            if triple in triples:
                duplicate_triples += 1

            triples.add(triple)

            if a == b:
                self_loops += 1

            if total % 1_000_000 == 0:
                print(
                    f"  {label}: {total:,} triples"
                )

    print(
        f"  {label}: {total:,} triples loaded"
    )

    return {
        "data": triples,
        "total": total,
        "unique": len(triples),
        "malformed": malformed,
        "self_loops": self_loops,
        "duplicate_triples": duplicate_triples,
    }


def compare_triples(original, final, report):

    write_header(
        report,
        "TRIPLE COMPARISON",
    )

    old = original["data"]
    new = final["data"]

    added = new - old
    removed = old - new
    common = old & new

    report.write(
        f"Original triple rows:               "
        f"{original['total']:,}\n"
    )

    report.write(
        f"Final triple rows:                  "
        f"{final['total']:,}\n"
    )

    report.write(
        f"Original unique triples:            "
        f"{original['unique']:,}\n"
    )

    report.write(
        f"Final unique triples:               "
        f"{final['unique']:,}\n"
    )

    report.write(
        f"Triples added:                      "
        f"{len(added):,}\n"
    )

    report.write(
        f"Triples removed:                    "
        f"{len(removed):,}\n"
    )

    report.write(
        f"Triples preserved:                  "
        f"{len(common):,}\n"
    )

    report.write(
        f"Duplicate triples - original:       "
        f"{original['duplicate_triples']:,}\n"
    )

    report.write(
        f"Duplicate triples - final:          "
        f"{final['duplicate_triples']:,}\n"
    )

    report.write(
        f"Self-loops - original:              "
        f"{original['self_loops']:,}\n"
    )

    report.write(
        f"Self-loops - final:                 "
        f"{final['self_loops']:,}\n"
    )

    report.write(
        f"Malformed triples - original:       "
        f"{original['malformed']:,}\n"
    )

    report.write(
        f"Malformed triples - final:          "
        f"{final['malformed']:,}\n"
    )

    report.write("\nTRIPLE PRESERVATION\n")

    report.write(
        f"Original triples preserved:         "
        f"{pct(len(common), original['unique']):.4f}%\n"
    )

    report.write(
        f"Final triples originating from old: "
        f"{pct(len(common), final['unique']):.4f}%\n"
    )

    return {
        "added": len(added),
        "removed": len(removed),
        "common": len(common),
    }


# ============================================================
# CROSS-FILE CONSISTENCY
# ============================================================

def cross_file_check(
    entities,
    relations,
    triples,
    report,
):

    write_header(
        report,
        "CROSS-FILE CONSISTENCY",
    )

    entity_ids = set(entities["data"])
    relation_ids = set(relations["data"])
    triple_set = triples["data"]

    missing_entities = set()
    missing_relations = set()

    referenced_entities = set()
    referenced_relations = set()

    for a, r, b in triple_set:

        referenced_entities.add(a)
        referenced_entities.add(b)
        referenced_relations.add(r)

        if a not in entity_ids:
            missing_entities.add(a)

        if b not in entity_ids:
            missing_entities.add(b)

        if r not in relation_ids:
            missing_relations.add(r)

    isolated_entities = (
        entity_ids - referenced_entities
    )

    unused_relations = (
        relation_ids - referenced_relations
    )

    report.write(
        f"Entities referenced by triples:    "
        f"{len(referenced_entities):,}\n"
    )

    report.write(
        f"Isolated entities:                  "
        f"{len(isolated_entities):,}\n"
    )

    report.write(
        f"Relations used by triples:          "
        f"{len(referenced_relations):,}\n"
    )

    report.write(
        f"Unused relations:                   "
        f"{len(unused_relations):,}\n"
    )

    report.write(
        f"Missing entity references:          "
        f"{len(missing_entities):,}\n"
    )

    report.write(
        f"Missing relation references:        "
        f"{len(missing_relations):,}\n"
    )

    return {
        "missing_entities": len(missing_entities),
        "missing_relations": len(missing_relations),
        "isolated_entities": len(isolated_entities),
        "unused_relations": len(unused_relations),
    }


# ============================================================
# FINAL SUMMARY
# ============================================================

def write_summary(
    report,
    entity_comparison,
    relation_comparison,
    triple_comparison,
    cross_file,
):

    write_header(
        report,
        "FINAL ORIGINAL VS FINAL SUMMARY",
    )

    report.write(
        "\nENTITY CHANGES\n"
    )

    report.write(
        f"  Added:                         "
        f"{entity_comparison['added']:,}\n"
    )

    report.write(
        f"  Removed:                       "
        f"{entity_comparison['removed']:,}\n"
    )

    report.write(
        f"  Names changed:                 "
        f"{entity_comparison['names_changed']:,}\n"
    )

    report.write(
        f"  Aliases changed:               "
        f"{entity_comparison['aliases_changed']:,}\n"
    )

    report.write(
        f"  Descriptions added:            "
        f"{entity_comparison['descriptions_added']:,}\n"
    )

    report.write(
        f"  Descriptions removed:          "
        f"{entity_comparison['descriptions_removed']:,}\n"
    )

    report.write(
        f"  Descriptions changed:          "
        f"{entity_comparison['descriptions_changed']:,}\n"
    )

    report.write(
        "\nRELATION CHANGES\n"
    )

    report.write(
        f"  Added:                         "
        f"{relation_comparison['added']:,}\n"
    )

    report.write(
        f"  Removed:                       "
        f"{relation_comparison['removed']:,}\n"
    )

    report.write(
        f"  Names changed:                 "
        f"{relation_comparison['names_changed']:,}\n"
    )

    report.write(
        "\nTRIPLE CHANGES\n"
    )

    report.write(
        f"  Added:                         "
        f"{triple_comparison['added']:,}\n"
    )

    report.write(
        f"  Removed:                       "
        f"{triple_comparison['removed']:,}\n"
    )

    report.write(
        f"  Preserved:                     "
        f"{triple_comparison['common']:,}\n"
    )

    report.write(
        "\nCROSS-FILE INTEGRITY\n"
    )

    report.write(
        f"  Missing entity refs:           "
        f"{cross_file['missing_entities']:,}\n"
    )

    report.write(
        f"  Missing relation refs:         "
        f"{cross_file['missing_relations']:,}\n"
    )

    report.write(
        f"  Isolated entities:             "
        f"{cross_file['isolated_entities']:,}\n"
    )

    report.write(
        f"  Unused relations:              "
        f"{cross_file['unused_relations']:,}\n"
    )

    # --------------------------------------------------------
    # Overall status
    # --------------------------------------------------------

    failures = []

    if entity_comparison["added"]:
        failures.append(
            "Unexpected entity additions"
        )

    if entity_comparison["removed"]:
        failures.append(
            "Unexpected entity removals"
        )

    if relation_comparison["added"]:
        failures.append(
            "Unexpected relation additions"
        )

    if relation_comparison["removed"]:
        failures.append(
            "Unexpected relation removals"
        )

    if cross_file["missing_entities"]:
        failures.append(
            "Missing entity references"
        )

    if cross_file["missing_relations"]:
        failures.append(
            "Missing relation references"
        )

    report.write(
        "\nFINAL COMPARISON STATUS\n"
    )

    if failures:

        report.write("  REVIEW REQUIRED\n")

        for failure in failures:
            report.write(
                f"  - {failure}\n"
            )

    else:

        report.write(
            "  PASS\n"
        )

        report.write(
            "  No unexpected entity/relation additions or removals.\n"
        )

        report.write(
            "  All triple references remain valid.\n"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ORIGINAL VS FINAL DATASET COMPARISON")
    print("=" * 60)

    files = [
        ORIGINAL_ENTITIES,
        FINAL_ENTITIES,
        ORIGINAL_RELATIONS,
        FINAL_RELATIONS,
        ORIGINAL_TRIPLES,
        FINAL_TRIPLES,
    ]

    print("\nChecking files...")

    for path in files:

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        print(f"  OK: {path}")

    os.makedirs(
        os.path.dirname(REPORT_FILE),
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    original_entities = load_entities(
        ORIGINAL_ENTITIES,
        "Original",
    )

    final_entities = load_entities(
        FINAL_ENTITIES,
        "Final",
    )

    original_relations = load_relations(
        ORIGINAL_RELATIONS,
        "Original",
    )

    final_relations = load_relations(
        FINAL_RELATIONS,
        "Final",
    )

    original_triples = load_triples(
        ORIGINAL_TRIPLES,
        "Original",
    )

    final_triples = load_triples(
        FINAL_TRIPLES,
        "Final",
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as report:

        report.write(
            "ORIGINAL VS FINAL DATASET COMPARISON\n"
        )

        report.write(
            "=" * 70 + "\n"
        )

        report.write(
            "Original:\n"
        )

        report.write(
            f"  Entities:  {ORIGINAL_ENTITIES}\n"
        )

        report.write(
            f"  Relations: {ORIGINAL_RELATIONS}\n"
        )

        report.write(
            f"  Triples:   {ORIGINAL_TRIPLES}\n"
        )

        report.write(
            "\nFinal:\n"
        )

        report.write(
            f"  Entities:  {FINAL_ENTITIES}\n"
        )

        report.write(
            f"  Relations: {FINAL_RELATIONS}\n"
        )

        report.write(
            f"  Triples:   {FINAL_TRIPLES}\n"
        )

        entity_comparison = compare_entities(
            original_entities,
            final_entities,
            report,
        )

        relation_comparison = compare_relations(
            original_relations,
            final_relations,
            report,
        )

        triple_comparison = compare_triples(
            original_triples,
            final_triples,
            report,
        )

        cross_file = cross_file_check(
            final_entities,
            final_relations,
            final_triples,
            report,
        )

        write_summary(
            report,
            entity_comparison,
            relation_comparison,
            triple_comparison,
            cross_file,
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)

    print(
        f"\nEntities:"
        f"\n  Original: {original_entities['total']:,}"
        f"\n  Final:    {final_entities['total']:,}"
    )

    print(
        f"\nRelations:"
        f"\n  Original: {original_relations['total']:,}"
        f"\n  Final:    {final_relations['total']:,}"
    )

    print(
        f"\nTriples:"
        f"\n  Original: {original_triples['total']:,}"
        f"\n  Final:    {final_triples['total']:,}"
        f"\n  Removed:  {triple_comparison['removed']:,}"
    )

    print(
        f"\nDescriptions:"
        f"\n  Added:    "
        f"{entity_comparison['descriptions_added']:,}"
        f"\n  Removed:  "
        f"{entity_comparison['descriptions_removed']:,}"
        f"\n  Changed:  "
        f"{entity_comparison['descriptions_changed']:,}"
    )

    print(
        f"\nReport:"
        f"\n  {REPORT_FILE}"
    )


if __name__ == "__main__":
    main()
