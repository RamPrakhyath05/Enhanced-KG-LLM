import csv
import os
import shutil
import sys

csv.field_size_limit(sys.maxsize)

# ============================================================
# INPUTS
# ============================================================

SEMANTIC_ENTITIES = (
    "dataset/semantic/cleaned/entities_final_clean.csv"
)

RELATIONS = (
    "dataset/cleaned/relations.csv"
)

SEMANTIC_TRIPLES = (
    "dataset/semantic/cleaned/triples_final.csv"
)

# ============================================================
# OUTPUT
# ============================================================

FINAL_DIR = "dataset/final"

FINAL_ENTITIES = os.path.join(
    FINAL_DIR,
    "entities.csv",
)

FINAL_RELATIONS = os.path.join(
    FINAL_DIR,
    "relations.csv",
)

FINAL_TRIPLES = os.path.join(
    FINAL_DIR,
    "triples.csv",
)

REPORT_FILE = (
    "dataset/semantic/analysis/"
    "final_kg_build_report.txt"
)


# ============================================================
# Helpers
# ============================================================

def require_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required file does not exist: {path}"
        )


def ensure_directory():
    os.makedirs(FINAL_DIR, exist_ok=True)


def copy_file(source, destination):
    shutil.copy2(source, destination)


# ============================================================
# Entity validation
# ============================================================

def validate_entities(path):
    print("\nValidating entities...")

    total = 0
    ids = set()

    duplicate_ids = 0
    blank_ids = 0
    blank_names = 0
    descriptions = 0
    empty_descriptions = 0

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
                f"Entities missing columns: {sorted(missing)}"
            )

        for row in reader:
            total += 1

            entity_id = row["Entity_ID"].strip()
            name = row["Entity_Name"].strip()
            description = row[
                "Entity_Description"
            ].strip()

            if not entity_id:
                blank_ids += 1
            elif entity_id in ids:
                duplicate_ids += 1
            else:
                ids.add(entity_id)

            if not name:
                blank_names += 1

            if description:
                descriptions += 1
            else:
                empty_descriptions += 1

            if total % 250_000 == 0:
                print(
                    f"Entities checked: {total:,}",
                    flush=True,
                )

    print(
        f"Entities: {total:,} | "
        f"Descriptions: {descriptions:,} | "
        f"Empty: {empty_descriptions:,}"
    )

    return {
        "total": total,
        "ids": ids,
        "duplicate_ids": duplicate_ids,
        "blank_ids": blank_ids,
        "blank_names": blank_names,
        "descriptions": descriptions,
        "empty_descriptions": empty_descriptions,
    }


# ============================================================
# Relation validation
# ============================================================

def validate_relations(path):
    print("\nValidating relations...")

    total = 0
    ids = set()

    duplicate_ids = 0
    blank_ids = 0
    blank_names = 0

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
                f"Relations missing columns: {sorted(missing)}"
            )

        for row in reader:
            total += 1

            relation_id = row[
                "Relation_ID"
            ].strip()

            relation_name = row[
                "Relation_Name"
            ].strip()

            if not relation_id:
                blank_ids += 1
            elif relation_id in ids:
                duplicate_ids += 1
            else:
                ids.add(relation_id)

            if not relation_name:
                blank_names += 1

    print(
        f"Relations: {total:,}"
    )

    return {
        "total": total,
        "ids": ids,
        "duplicate_ids": duplicate_ids,
        "blank_ids": blank_ids,
        "blank_names": blank_names,
    }


# ============================================================
# Triple validation
# ============================================================

def validate_triples(
    path,
    entity_ids,
    relation_ids,
):
    print("\nValidating triples...")

    total = 0
    unique = set()

    duplicate = 0
    self_loops = 0
    malformed = 0
    missing_entities = 0
    missing_relations = 0

    used_entities = set()
    used_relations = set()

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
                f"Triples missing columns: {sorted(missing)}"
            )

        for row in reader:
            total += 1

            a = row["Entity_1_ID"].strip()
            r = row["Relation_ID"].strip()
            b = row["Entity_2_ID"].strip()

            if not a or not r or not b:
                malformed += 1
                continue

            triple = (a, r, b)

            if triple in unique:
                duplicate += 1
            else:
                unique.add(triple)

            if a == b:
                self_loops += 1

            if a not in entity_ids:
                missing_entities += 1

            if b not in entity_ids:
                missing_entities += 1

            if r not in relation_ids:
                missing_relations += 1

            used_entities.add(a)
            used_entities.add(b)
            used_relations.add(r)

            if total % 1_000_000 == 0:
                print(
                    f"Triples checked: {total:,}",
                    flush=True,
                )

    unused_entities = (
        len(entity_ids - used_entities)
    )

    unused_relations = (
        len(relation_ids - used_relations)
    )

    print(
        f"Triples: {total:,}"
    )

    return {
        "total": total,
        "unique": len(unique),
        "duplicate": duplicate,
        "self_loops": self_loops,
        "malformed": malformed,
        "missing_entities": missing_entities,
        "missing_relations": missing_relations,
        "used_entities": len(used_entities),
        "unused_entities": unused_entities,
        "used_relations": len(used_relations),
        "unused_relations": unused_relations,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("BUILD FINAL KNOWLEDGE GRAPH")
    print("=" * 60)

    print("\nChecking input files...")

    for path in [
        SEMANTIC_ENTITIES,
        RELATIONS,
        SEMANTIC_TRIPLES,
    ]:
        require_file(path)
        print(f"  OK: {path}")

    ensure_directory()

    # --------------------------------------------------------
    # Validate source files BEFORE copying
    # --------------------------------------------------------

    entities = validate_entities(
        SEMANTIC_ENTITIES
    )

    relations = validate_relations(
        RELATIONS
    )

    triples = validate_triples(
        SEMANTIC_TRIPLES,
        entities["ids"],
        relations["ids"],
    )

    # --------------------------------------------------------
    # Hard validation
    # --------------------------------------------------------

    failures = []

    if entities["duplicate_ids"]:
        failures.append(
            f"Duplicate entity IDs: "
            f"{entities['duplicate_ids']:,}"
        )

    if entities["blank_ids"]:
        failures.append(
            f"Blank entity IDs: "
            f"{entities['blank_ids']:,}"
        )

    if entities["blank_names"]:
        failures.append(
            f"Blank entity names: "
            f"{entities['blank_names']:,}"
        )

    if relations["duplicate_ids"]:
        failures.append(
            f"Duplicate relation IDs: "
            f"{relations['duplicate_ids']:,}"
        )

    if relations["blank_ids"]:
        failures.append(
            f"Blank relation IDs: "
            f"{relations['blank_ids']:,}"
        )

    if relations["blank_names"]:
        failures.append(
            f"Blank relation names: "
            f"{relations['blank_names']:,}"
        )

    if triples["duplicate"]:
        failures.append(
            f"Duplicate triples: "
            f"{triples['duplicate']:,}"
        )

    if triples["self_loops"]:
        failures.append(
            f"Self-loops: "
            f"{triples['self_loops']:,}"
        )

    if triples["malformed"]:
        failures.append(
            f"Malformed triples: "
            f"{triples['malformed']:,}"
        )

    if triples["missing_entities"]:
        failures.append(
            f"Missing entity references: "
            f"{triples['missing_entities']:,}"
        )

    if triples["missing_relations"]:
        failures.append(
            f"Missing relation references: "
            f"{triples['missing_relations']:,}"
        )

    # --------------------------------------------------------
    # DO NOT BUILD IF SOURCE VALIDATION FAILS
    # --------------------------------------------------------

    if failures:
        print("\n" + "=" * 60)
        print("FINAL KG BUILD ABORTED")
        print("=" * 60)

        for failure in failures:
            print(f"FAIL: {failure}")

        raise RuntimeError(
            "Source validation failed. "
            "No final dataset was created."
        )

    # --------------------------------------------------------
    # Copy validated files
    # --------------------------------------------------------

    print("\nBuilding final dataset...")

    copy_file(
        SEMANTIC_ENTITIES,
        FINAL_ENTITIES,
    )

    copy_file(
        RELATIONS,
        FINAL_RELATIONS,
    )

    copy_file(
        SEMANTIC_TRIPLES,
        FINAL_TRIPLES,
    )

    print(
        f"  Entities  -> {FINAL_ENTITIES}"
    )

    print(
        f"  Relations -> {FINAL_RELATIONS}"
    )

    print(
        f"  Triples   -> {FINAL_TRIPLES}"
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(REPORT_FILE),
        exist_ok=True,
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as report:

        report.write(
            "FINAL KNOWLEDGE GRAPH BUILD REPORT\n"
        )
        report.write(
            "=" * 70 + "\n\n"
        )

        report.write("SOURCE FILES\n")
        report.write(
            f"Entities:  {SEMANTIC_ENTITIES}\n"
        )
        report.write(
            f"Relations: {RELATIONS}\n"
        )
        report.write(
            f"Triples:   {SEMANTIC_TRIPLES}\n\n"
        )

        report.write("FINAL FILES\n")
        report.write(
            f"Entities:  {FINAL_ENTITIES}\n"
        )
        report.write(
            f"Relations: {FINAL_RELATIONS}\n"
        )
        report.write(
            f"Triples:   {FINAL_TRIPLES}\n\n"
        )

        report.write("COUNTS\n")
        report.write(
            f"Entities:                  "
            f"{entities['total']:,}\n"
        )
        report.write(
            f"Non-empty descriptions:    "
            f"{entities['descriptions']:,}\n"
        )
        report.write(
            f"Empty descriptions:       "
            f"{entities['empty_descriptions']:,}\n"
        )
        report.write(
            f"Relations:                 "
            f"{relations['total']:,}\n"
        )
        report.write(
            f"Triples:                   "
            f"{triples['total']:,}\n"
        )

        report.write("\nVALIDATION\n")
        report.write(
            f"Unique triples:            "
            f"{triples['unique']:,}\n"
        )
        report.write(
            f"Duplicate triples:        "
            f"{triples['duplicate']:,}\n"
        )
        report.write(
            f"Self-loops:               "
            f"{triples['self_loops']:,}\n"
        )
        report.write(
            f"Malformed triples:        "
            f"{triples['malformed']:,}\n"
        )
        report.write(
            f"Missing entity refs:      "
            f"{triples['missing_entities']:,}\n"
        )
        report.write(
            f"Missing relation refs:    "
            f"{triples['missing_relations']:,}\n"
        )

        report.write("\nGRAPH COVERAGE\n")
        report.write(
            f"Referenced entities:      "
            f"{triples['used_entities']:,}\n"
        )
        report.write(
            f"Unused entities:          "
            f"{triples['unused_entities']:,}\n"
        )
        report.write(
            f"Used relations:           "
            f"{triples['used_relations']:,}\n"
        )
        report.write(
            f"Unused relations:         "
            f"{triples['unused_relations']:,}\n"
        )

        report.write("\nFINAL STATUS\n")
        report.write(
            "PASS\n"
        )

    print("\n" + "=" * 60)
    print("FINAL KNOWLEDGE GRAPH BUILD COMPLETE")
    print("=" * 60)

    print("\nFinal dataset:")
    print(f"  {FINAL_DIR}/")

    print("\nReport:")
    print(f"  {REPORT_FILE}")

    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()
