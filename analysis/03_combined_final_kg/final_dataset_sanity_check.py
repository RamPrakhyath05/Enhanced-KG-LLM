import csv
import os
import re
import sys
from collections import Counter

csv.field_size_limit(sys.maxsize)

# ============================================================
# FINAL DATASET INPUTS
# ============================================================

FINAL_DIR = "dataset/final"

ENTITIES_FILE = os.path.join(
    FINAL_DIR,
    "entities.csv",
)

RELATIONS_FILE = os.path.join(
    FINAL_DIR,
    "relations.csv",
)

TRIPLES_FILE = os.path.join(
    FINAL_DIR,
    "triples.csv",
)

REPORT_FILE = (
    "dataset/semantic/analysis/"
    "final_dataset_sanity_check.txt"
)


# ============================================================
# Expected schemas
# ============================================================

ENTITY_COLUMNS = {
    "Entity_ID",
    "Entity_Name",
    "Entity_Aliases",
    "Entity_Description",
}

RELATION_COLUMNS = {
    "Relation_ID",
    "Relation_Name",
}

TRIPLE_COLUMNS = {
    "Entity_1_ID",
    "Relation_ID",
    "Entity_2_ID",
}


# ============================================================
# Helpers
# ============================================================

def write_section(report, title):
    report.write("\n")
    report.write("=" * 70 + "\n")
    report.write(title + "\n")
    report.write("=" * 70 + "\n")


def add_sample(samples, value, limit=20):
    if len(samples) < limit:
        samples.append(value)


def looks_like_wikidata_id(value):
    return bool(
        re.fullmatch(
            r"Q[1-9][0-9]*",
            value,
        )
    )


def looks_like_relation_id(value):
    return bool(
        re.fullmatch(
            r"P[1-9][0-9]*",
            value,
        )
    )


def normalize_text(value):
    return " ".join(
        value.split()
    ).strip()


# ============================================================
# ENTITY AUDIT
# ============================================================

def audit_entities(report):

    write_section(
        report,
        "ENTITY AUDIT",
    )

    total = 0

    ids = set()
    names = Counter()

    duplicate_ids = 0
    blank_ids = 0
    malformed_ids = 0

    blank_names = 0
    duplicate_name_rows = 0

    blank_aliases = 0
    malformed_alias_rows = 0

    empty_descriptions = 0
    whitespace_descriptions = 0

    navigation_descriptions = 0
    wiki_artifact_descriptions = 0
    repetitive_descriptions = 0

    sample_duplicate_ids = []
    sample_malformed_ids = []
    sample_blank_names = []
    sample_malformed_aliases = []
    sample_navigation = []
    sample_wiki_artifacts = []
    sample_repetitive = []

    # --------------------------------------------------------
    # First pass
    # --------------------------------------------------------

    with open(
        ENTITIES_FILE,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(
                "entities.csv has no header."
            )

        missing_columns = (
            ENTITY_COLUMNS
            - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "entities.csv missing columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:

            total += 1

            entity_id = (
                row["Entity_ID"] or ""
            ).strip()

            name = (
                row["Entity_Name"] or ""
            ).strip()

            aliases = (
                row["Entity_Aliases"] or ""
            ).strip()

            description = (
                row["Entity_Description"] or ""
            )

            # ------------------------------------------------
            # Entity ID
            # ------------------------------------------------

            if not entity_id:
                blank_ids += 1

                add_sample(
                    sample_malformed_ids,
                    f"Blank Entity_ID at row {total}",
                )

            else:

                if not looks_like_wikidata_id(
                    entity_id
                ):
                    malformed_ids += 1

                    add_sample(
                        sample_malformed_ids,
                        entity_id,
                    )

                if entity_id in ids:
                    duplicate_ids += 1

                    add_sample(
                        sample_duplicate_ids,
                        entity_id,
                    )

                else:
                    ids.add(entity_id)

            # ------------------------------------------------
            # Name
            # ------------------------------------------------

            if not name:
                blank_names += 1

                add_sample(
                    sample_blank_names,
                    f"Blank Entity_Name at row {total}",
                )

            else:
                names[name] += 1

            # ------------------------------------------------
            # Aliases
            # ------------------------------------------------

            if not aliases:
                blank_aliases += 1

            else:

                alias_values = aliases.split("|")

                cleaned_aliases = [
                    a.strip()
                    for a in alias_values
                ]

                if any(
                    not alias
                    for alias in cleaned_aliases
                ):
                    malformed_alias_rows += 1

                    add_sample(
                        sample_malformed_aliases,
                        (
                            f"{entity_id}: "
                            f"{aliases[:200]}"
                        ),
                    )

            # ------------------------------------------------
            # Description
            # ------------------------------------------------

            if not description.strip():
                empty_descriptions += 1

            else:

                if description != description.strip():
                    whitespace_descriptions += 1

                normalized = normalize_text(
                    description
                )

                lower = normalized.lower()

                # Navigation / list indicators
                navigation_patterns = [
                    "may refer to",
                    "can refer to",
                    "refers to",
                    "disambiguation",
                    "list of ",
                    "lists of ",
                    "index of ",
                    "outline of ",
                    "category:",
                    "template:",
                    "portal:",
                    "navigation",
                ]

                if any(
                    pattern in lower
                    for pattern in navigation_patterns
                ):
                    navigation_descriptions += 1

                    add_sample(
                        sample_navigation,
                        (
                            f"{entity_id}: "
                            f"{normalized[:300]}"
                        ),
                    )

                # Wiki artifacts
                wiki_patterns = [
                    "[[",
                    "]]",
                    "{{",
                    "}}",
                    "<ref",
                    "</ref>",
                    "&nbsp;",
                    "http://",
                    "https://",
                    "www.",
                ]

                if any(
                    pattern in lower
                    for pattern in wiki_patterns
                ):
                    wiki_artifact_descriptions += 1

                    add_sample(
                        sample_wiki_artifacts,
                        (
                            f"{entity_id}: "
                            f"{normalized[:300]}"
                        ),
                    )

                # Obvious repetitive text
                words = normalized.split()

                if len(words) >= 8:

                    unique_words = set(
                        word.lower()
                        for word in words
                    )

                    uniqueness_ratio = (
                        len(unique_words)
                        / len(words)
                    )

                    if uniqueness_ratio < 0.45:
                        repetitive_descriptions += 1

                        add_sample(
                            sample_repetitive,
                            (
                                f"{entity_id}: "
                                f"{normalized[:300]}"
                            ),
                        )

    # --------------------------------------------------------
    # Duplicate names
    # --------------------------------------------------------

    for name, count in names.items():

        if count > 1:
            duplicate_name_rows += count

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report.write(
        f"Entities processed:             {total:,}\n"
    )

    report.write(
        f"Unique Entity_IDs:              {len(ids):,}\n"
    )

    report.write(
        f"Duplicate Entity_IDs:           {duplicate_ids:,}\n"
    )

    report.write(
        f"Blank Entity_IDs:               {blank_ids:,}\n"
    )

    report.write(
        f"Malformed Entity_IDs:           {malformed_ids:,}\n"
    )

    report.write(
        f"Blank Entity_Names:              {blank_names:,}\n"
    )

    report.write(
        f"Duplicate-name rows:             "
        f"{duplicate_name_rows:,}\n"
    )

    report.write(
        f"Blank alias fields:              "
        f"{blank_aliases:,}\n"
    )

    report.write(
        f"Malformed alias rows:            "
        f"{malformed_alias_rows:,}\n"
    )

    report.write(
        f"Empty descriptions:              "
        f"{empty_descriptions:,}\n"
    )

    report.write(
        f"Descriptions with outer whitespace:"
        f" {whitespace_descriptions:,}\n"
    )

    report.write(
        f"Navigation/list descriptions:    "
        f"{navigation_descriptions:,}\n"
    )

    report.write(
        f"Possible wiki artifacts:         "
        f"{wiki_artifact_descriptions:,}\n"
    )

    report.write(
        f"Possible repetitive descriptions:"
        f" {repetitive_descriptions:,}\n"
    )

    def samples(title, values):
        report.write(f"\n{title}\n")

        if not values:
            report.write("  None detected.\n")
        else:
            for value in values:
                report.write(
                    f"  {value}\n"
                )

    samples(
        "Sample duplicate Entity_IDs",
        sample_duplicate_ids,
    )

    samples(
        "Sample malformed Entity_IDs",
        sample_malformed_ids,
    )

    samples(
        "Sample blank Entity_Names",
        sample_blank_names,
    )

    samples(
        "Sample malformed alias rows",
        sample_malformed_aliases,
    )

    samples(
        "Sample navigation/list descriptions",
        sample_navigation,
    )

    samples(
        "Sample wiki artifacts",
        sample_wiki_artifacts,
    )

    samples(
        "Sample repetitive descriptions",
        sample_repetitive,
    )

    return {
        "total": total,
        "ids": ids,
        "duplicate_ids": duplicate_ids,
        "blank_ids": blank_ids,
        "malformed_ids": malformed_ids,
        "blank_names": blank_names,
        "duplicate_name_rows": duplicate_name_rows,
        "blank_aliases": blank_aliases,
        "malformed_alias_rows": malformed_alias_rows,
        "empty_descriptions": empty_descriptions,
        "whitespace_descriptions": whitespace_descriptions,
        "navigation_descriptions": navigation_descriptions,
        "wiki_artifact_descriptions": wiki_artifact_descriptions,
        "repetitive_descriptions": repetitive_descriptions,
    }


# ============================================================
# RELATION AUDIT
# ============================================================

def audit_relations(report):

    write_section(
        report,
        "RELATION AUDIT",
    )

    total = 0

    ids = set()
    names = Counter()

    duplicate_ids = 0
    blank_ids = 0
    malformed_ids = 0

    blank_names = 0
    duplicate_name_rows = 0

    sample_duplicate_ids = []
    sample_malformed_ids = []
    sample_blank_names = []

    with open(
        RELATIONS_FILE,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(
                "relations.csv has no header."
            )

        missing_columns = (
            RELATION_COLUMNS
            - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "relations.csv missing columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:

            total += 1

            relation_id = (
                row["Relation_ID"] or ""
            ).strip()

            relation_name = (
                row["Relation_Name"] or ""
            ).strip()

            # ------------------------------------------------
            # ID
            # ------------------------------------------------

            if not relation_id:
                blank_ids += 1

            else:

                if not looks_like_relation_id(
                    relation_id
                ):
                    malformed_ids += 1

                    add_sample(
                        sample_malformed_ids,
                        relation_id,
                    )

                if relation_id in ids:
                    duplicate_ids += 1

                    add_sample(
                        sample_duplicate_ids,
                        relation_id,
                    )

                else:
                    ids.add(relation_id)

            # ------------------------------------------------
            # Name
            # ------------------------------------------------

            if not relation_name:
                blank_names += 1

                add_sample(
                    sample_blank_names,
                    f"Blank Relation_Name at row {total}",
                )

            else:
                names[relation_name] += 1

    for name, count in names.items():

        if count > 1:
            duplicate_name_rows += count

    report.write(
        f"Relations processed:            {total:,}\n"
    )

    report.write(
        f"Unique Relation_IDs:            {len(ids):,}\n"
    )

    report.write(
        f"Duplicate Relation_IDs:         {duplicate_ids:,}\n"
    )

    report.write(
        f"Blank Relation_IDs:             {blank_ids:,}\n"
    )

    report.write(
        f"Malformed Relation_IDs:         {malformed_ids:,}\n"
    )

    report.write(
        f"Duplicate-name rows:             "
        f"{duplicate_name_rows:,}\n"
    )

    report.write(
        f"Blank Relation_Names:            "
        f"{blank_names:,}\n"
    )

    def samples(title, values):
        report.write(f"\n{title}\n")

        if not values:
            report.write("  None detected.\n")
        else:
            for value in values:
                report.write(
                    f"  {value}\n"
                )

    samples(
        "Sample duplicate Relation_IDs",
        sample_duplicate_ids,
    )

    samples(
        "Sample malformed Relation_IDs",
        sample_malformed_ids,
    )

    samples(
        "Sample blank Relation_Names",
        sample_blank_names,
    )

    return {
        "total": total,
        "ids": ids,
        "duplicate_ids": duplicate_ids,
        "blank_ids": blank_ids,
        "malformed_ids": malformed_ids,
        "duplicate_name_rows": duplicate_name_rows,
        "blank_names": blank_names,
    }


# ============================================================
# TRIPLE AUDIT
# ============================================================

def audit_triples(
    report,
    entity_ids,
    relation_ids,
):

    write_section(
        report,
        "TRIPLE AUDIT",
    )

    total = 0

    triple_set = set()

    duplicate_triples = 0
    self_loops = 0
    malformed = 0

    missing_entity_refs = 0
    missing_relation_refs = 0

    invalid_entity_ids = 0
    invalid_relation_ids = 0

    used_entities = set()
    used_relations = set()

    relation_usage = Counter()

    sample_duplicates = []
    sample_self_loops = []
    sample_malformed = []
    sample_missing_entities = []
    sample_missing_relations = []

    with open(
        TRIPLES_FILE,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(
                "triples.csv has no header."
            )

        missing_columns = (
            TRIPLE_COLUMNS
            - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "triples.csv missing columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:

            total += 1

            entity_1 = (
                row["Entity_1_ID"] or ""
            ).strip()

            relation = (
                row["Relation_ID"] or ""
            ).strip()

            entity_2 = (
                row["Entity_2_ID"] or ""
            ).strip()

            # ------------------------------------------------
            # Empty fields
            # ------------------------------------------------

            if (
                not entity_1
                or not relation
                or not entity_2
            ):

                malformed += 1

                add_sample(
                    sample_malformed,
                    (
                        f"{entity_1} | "
                        f"{relation} | "
                        f"{entity_2}"
                    ),
                )

                continue

            # ------------------------------------------------
            # ID format
            # ------------------------------------------------

            if not looks_like_wikidata_id(
                entity_1
            ):
                invalid_entity_ids += 1

            if not looks_like_wikidata_id(
                entity_2
            ):
                invalid_entity_ids += 1

            if not looks_like_relation_id(
                relation
            ):
                invalid_relation_ids += 1

            # ------------------------------------------------
            # Duplicate triple
            # ------------------------------------------------

            triple = (
                entity_1,
                relation,
                entity_2,
            )

            if triple in triple_set:

                duplicate_triples += 1

                add_sample(
                    sample_duplicates,
                    (
                        f"{entity_1} | "
                        f"{relation} | "
                        f"{entity_2}"
                    ),
                )

            else:
                triple_set.add(triple)

            # ------------------------------------------------
            # Self-loop
            # ------------------------------------------------

            if entity_1 == entity_2:

                self_loops += 1

                add_sample(
                    sample_self_loops,
                    (
                        f"{entity_1} | "
                        f"{relation} | "
                        f"{entity_2}"
                    ),
                )

            # ------------------------------------------------
            # Entity references
            # ------------------------------------------------

            if entity_1 not in entity_ids:

                missing_entity_refs += 1

                add_sample(
                    sample_missing_entities,
                    entity_1,
                )

            else:
                used_entities.add(entity_1)

            if entity_2 not in entity_ids:

                missing_entity_refs += 1

                add_sample(
                    sample_missing_entities,
                    entity_2,
                )

            else:
                used_entities.add(entity_2)

            # ------------------------------------------------
            # Relation references
            # ------------------------------------------------

            if relation not in relation_ids:

                missing_relation_refs += 1

                add_sample(
                    sample_missing_relations,
                    relation,
                )

            else:
                used_relations.add(relation)

            relation_usage[relation] += 1

            if total % 1_000_000 == 0:

                print(
                    f"Triples processed: {total:,}",
                    flush=True,
                )

    isolated_entities = (
        entity_ids - used_entities
    )

    unused_relations = (
        relation_ids - used_relations
    )

    # --------------------------------------------------------
    # Degree statistics
    # --------------------------------------------------------

    degree_counter = Counter()

    for entity_1, relation, entity_2 in triple_set:

        degree_counter[entity_1] += 1
        degree_counter[entity_2] += 1

    if degree_counter:

        min_degree = min(
            degree_counter.values()
        )

        max_degree = max(
            degree_counter.values()
        )

        avg_degree = (
            sum(degree_counter.values())
            / len(degree_counter)
        )

    else:

        min_degree = 0
        max_degree = 0
        avg_degree = 0.0

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report.write(
        f"Triples processed:              {total:,}\n"
    )

    report.write(
        f"Unique triples:                 "
        f"{len(triple_set):,}\n"
    )

    report.write(
        f"Duplicate triples:             "
        f"{duplicate_triples:,}\n"
    )

    report.write(
        f"Self-loops:                    "
        f"{self_loops:,}\n"
    )

    report.write(
        f"Malformed triples:             "
        f"{malformed:,}\n"
    )

    report.write(
        f"Invalid entity IDs:            "
        f"{invalid_entity_ids:,}\n"
    )

    report.write(
        f"Invalid relation IDs:          "
        f"{invalid_relation_ids:,}\n"
    )

    report.write(
        f"Missing entity references:     "
        f"{missing_entity_refs:,}\n"
    )

    report.write(
        f"Missing relation references:   "
        f"{missing_relation_refs:,}\n"
    )

    report.write("\nGRAPH COVERAGE\n")

    report.write(
        f"Entities referenced by triples:"
        f" {len(used_entities):,}\n"
    )

    report.write(
        f"Isolated entities:             "
        f"{len(isolated_entities):,}\n"
    )

    report.write(
        f"Relations used:                "
        f"{len(used_relations):,}\n"
    )

    report.write(
        f"Unused relations:              "
        f"{len(unused_relations):,}\n"
    )

    report.write("\nDEGREE STATISTICS\n")

    report.write(
        f"Minimum degree:                "
        f"{min_degree:,}\n"
    )

    report.write(
        f"Maximum degree:                "
        f"{max_degree:,}\n"
    )

    report.write(
        f"Average degree:                "
        f"{avg_degree:.2f}\n"
    )

    report.write("\nTOP RELATIONS BY USAGE\n")

    for relation, count in (
        relation_usage.most_common(20)
    ):

        report.write(
            f"  {relation}: {count:,}\n"
        )

    def samples(title, values):

        report.write(
            f"\n{title}\n"
        )

        if not values:

            report.write(
                "  None detected.\n"
            )

        else:

            for value in values:

                report.write(
                    f"  {value}\n"
                )

    samples(
        "Sample duplicate triples",
        sample_duplicates,
    )

    samples(
        "Sample self-loops",
        sample_self_loops,
    )

    samples(
        "Sample malformed triples",
        sample_malformed,
    )

    samples(
        "Sample missing entity references",
        sample_missing_entities,
    )

    samples(
        "Sample missing relation references",
        sample_missing_relations,
    )

    return {
        "total": total,
        "unique": len(triple_set),
        "duplicate": duplicate_triples,
        "self_loops": self_loops,
        "malformed": malformed,
        "invalid_entity_ids": invalid_entity_ids,
        "invalid_relation_ids": invalid_relation_ids,
        "missing_entity_refs": missing_entity_refs,
        "missing_relation_refs": missing_relation_refs,
        "used_entities": len(used_entities),
        "isolated_entities": len(isolated_entities),
        "used_relations": len(used_relations),
        "unused_relations": len(unused_relations),
    }


# ============================================================
# CROSS-FILE CONSISTENCY
# ============================================================

def cross_file_checks(
    report,
    entities,
    relations,
    triples,
):

    write_section(
        report,
        "CROSS-FILE CONSISTENCY",
    )

    checks = []

    checks.append(
        (
            "All triple entity references exist",
            triples["missing_entity_refs"] == 0,
        )
    )

    checks.append(
        (
            "All triple relation references exist",
            triples["missing_relation_refs"] == 0,
        )
    )

    checks.append(
        (
            "No duplicate entity IDs",
            entities["duplicate_ids"] == 0,
        )
    )

    checks.append(
        (
            "No duplicate relation IDs",
            relations["duplicate_ids"] == 0,
        )
    )

    checks.append(
        (
            "No duplicate triples",
            triples["duplicate"] == 0,
        )
    )

    checks.append(
        (
            "No self-loops",
            triples["self_loops"] == 0,
        )
    )

    checks.append(
        (
            "No malformed triples",
            triples["malformed"] == 0,
        )
    )

    checks.append(
        (
            "No malformed entity IDs",
            entities["malformed_ids"] == 0,
        )
    )

    checks.append(
        (
            "No malformed relation IDs",
            relations["malformed_ids"] == 0,
        )
    )

    all_pass = True

    for description, passed in checks:

        if passed:

            report.write(
                f"PASS  {description}\n"
            )

        else:

            report.write(
                f"FAIL  {description}\n"
            )

            all_pass = False

    return all_pass


# ============================================================
# FINAL VERDICT
# ============================================================

def final_verdict(
    report,
    entities,
    relations,
    triples,
):

    write_section(
        report,
        "FINAL VERDICT",
    )

    hard_failures = []
    warnings = []

    # --------------------------------------------------------
    # HARD FAILURES
    # --------------------------------------------------------

    if entities["duplicate_ids"]:
        hard_failures.append(
            f"Duplicate Entity_IDs: "
            f"{entities['duplicate_ids']:,}"
        )

    if entities["blank_ids"]:
        hard_failures.append(
            f"Blank Entity_IDs: "
            f"{entities['blank_ids']:,}"
        )

    if entities["malformed_ids"]:
        hard_failures.append(
            f"Malformed Entity_IDs: "
            f"{entities['malformed_ids']:,}"
        )

    if entities["blank_names"]:
        hard_failures.append(
            f"Blank Entity_Names: "
            f"{entities['blank_names']:,}"
        )

    if relations["duplicate_ids"]:
        hard_failures.append(
            f"Duplicate Relation_IDs: "
            f"{relations['duplicate_ids']:,}"
        )

    if relations["blank_ids"]:
        hard_failures.append(
            f"Blank Relation_IDs: "
            f"{relations['blank_ids']:,}"
        )

    if relations["malformed_ids"]:
        hard_failures.append(
            f"Malformed Relation_IDs: "
            f"{relations['malformed_ids']:,}"
        )

    if relations["blank_names"]:
        hard_failures.append(
            f"Blank Relation_Names: "
            f"{relations['blank_names']:,}"
        )

    if triples["duplicate"]:
        hard_failures.append(
            f"Duplicate triples: "
            f"{triples['duplicate']:,}"
        )

    if triples["self_loops"]:
        hard_failures.append(
            f"Self-loops: "
            f"{triples['self_loops']:,}"
        )

    if triples["malformed"]:
        hard_failures.append(
            f"Malformed triples: "
            f"{triples['malformed']:,}"
        )

    if triples["invalid_entity_ids"]:
        hard_failures.append(
            f"Invalid entity IDs in triples: "
            f"{triples['invalid_entity_ids']:,}"
        )

    if triples["invalid_relation_ids"]:
        hard_failures.append(
            f"Invalid relation IDs in triples: "
            f"{triples['invalid_relation_ids']:,}"
        )

    if triples["missing_entity_refs"]:
        hard_failures.append(
            f"Missing entity references: "
            f"{triples['missing_entity_refs']:,}"
        )

    if triples["missing_relation_refs"]:
        hard_failures.append(
            f"Missing relation references: "
            f"{triples['missing_relation_refs']:,}"
        )

    # --------------------------------------------------------
    # WARNINGS
    # --------------------------------------------------------

    if entities["empty_descriptions"]:
        warnings.append(
            f"Entities without descriptions: "
            f"{entities['empty_descriptions']:,}"
        )

    if entities["duplicate_name_rows"]:
        warnings.append(
            f"Duplicate entity-name rows: "
            f"{entities['duplicate_name_rows']:,}"
        )

    if entities["navigation_descriptions"]:
        warnings.append(
            f"Navigation/list descriptions: "
            f"{entities['navigation_descriptions']:,}"
        )

    if entities["wiki_artifact_descriptions"]:
        warnings.append(
            f"Possible wiki artifacts: "
            f"{entities['wiki_artifact_descriptions']:,}"
        )

    if entities["repetitive_descriptions"]:
        warnings.append(
            f"Possible repetitive descriptions: "
            f"{entities['repetitive_descriptions']:,}"
        )

    if entities["malformed_alias_rows"]:
        warnings.append(
            f"Malformed alias rows: "
            f"{entities['malformed_alias_rows']:,}"
        )

    if relations["duplicate_name_rows"]:
        warnings.append(
            f"Duplicate relation-name rows: "
            f"{relations['duplicate_name_rows']:,}"
        )

    if triples["unused_relations"]:
        warnings.append(
            f"Unused relations: "
            f"{triples['unused_relations']:,}"
        )

    if triples["isolated_entities"]:
        warnings.append(
            f"Isolated entities: "
            f"{triples['isolated_entities']:,}"
        )

    # --------------------------------------------------------
    # Print failures
    # --------------------------------------------------------

    report.write("STRUCTURAL FAILURES\n")

    if not hard_failures:

        report.write(
            "  None detected.\n"
        )

    else:

        for failure in hard_failures:

            report.write(
                f"  FAIL: {failure}\n"
            )

    # --------------------------------------------------------
    # Print warnings
    # --------------------------------------------------------

    report.write("\nWARNINGS\n")

    if not warnings:

        report.write(
            "  None detected.\n"
        )

    else:

        for warning in warnings:

            report.write(
                f"  WARNING: {warning}\n"
            )

    # --------------------------------------------------------
    # Verdict
    # --------------------------------------------------------

    passed = not hard_failures

    report.write("\n")

    if passed:

        report.write(
            "FINAL STATUS: PASS\n"
        )

    else:

        report.write(
            "FINAL STATUS: FAIL\n"
        )

    return passed


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FINAL DATASET SANITY CHECK")
    print("=" * 60)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    print("\nChecking final dataset files...")

    required_files = [
        ENTITIES_FILE,
        RELATIONS_FILE,
        TRIPLES_FILE,
    ]

    for path in required_files:

        if not os.path.isfile(path):

            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

        print(
            f"  OK: {path}"
        )

    os.makedirs(
        os.path.dirname(REPORT_FILE),
        exist_ok=True,
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
            "FINAL DATASET SANITY CHECK\n"
        )

        report.write(
            "=" * 70 + "\n"
        )

        report.write(
            "This audit checks the actual canonical "
            "dataset/final files.\n"
        )

        report.write(
            "\nINPUT FILES\n"
        )

        report.write(
            f"Entities:  {ENTITIES_FILE}\n"
        )

        report.write(
            f"Relations: {RELATIONS_FILE}\n"
        )

        report.write(
            f"Triples:   {TRIPLES_FILE}\n"
        )

        # ----------------------------------------------------
        # Entity audit
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("AUDITING ENTITIES")
        print("=" * 60)

        entities = audit_entities(
            report
        )

        # ----------------------------------------------------
        # Relation audit
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("AUDITING RELATIONS")
        print("=" * 60)

        relations = audit_relations(
            report
        )

        # ----------------------------------------------------
        # Triple audit
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("AUDITING TRIPLES")
        print("=" * 60)

        triples = audit_triples(
            report,
            entities["ids"],
            relations["ids"],
        )

        # ----------------------------------------------------
        # Cross-file checks
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("CROSS-FILE CONSISTENCY")
        print("=" * 60)

        cross_file_passed = cross_file_checks(
            report,
            entities,
            relations,
            triples,
        )

        # ----------------------------------------------------
        # Final verdict
        # ----------------------------------------------------

        passed = final_verdict(
            report,
            entities,
            relations,
            triples,
        )

        # Cross-file check is included explicitly
        if not cross_file_passed:
            passed = False

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL DATASET SANITY CHECK COMPLETE")
    print("=" * 60)

    print("\nFinal dataset:")
    print(
        f"  Entities:  {entities['total']:,}"
    )
    print(
        f"  Relations: {relations['total']:,}"
    )
    print(
        f"  Triples:   {triples['total']:,}"
    )

    print("\nQuality:")
    print(
        f"  Duplicate entities: "
        f"{entities['duplicate_ids']:,}"
    )
    print(
        f"  Duplicate relations: "
        f"{relations['duplicate_ids']:,}"
    )
    print(
        f"  Duplicate triples: "
        f"{triples['duplicate']:,}"
    )
    print(
        f"  Self-loops: "
        f"{triples['self_loops']:,}"
    )
    print(
        f"  Missing entity refs: "
        f"{triples['missing_entity_refs']:,}"
    )
    print(
        f"  Missing relation refs: "
        f"{triples['missing_relation_refs']:,}"
    )

    print("\nDescriptions:")
    print(
        f"  Non-empty: "
        f"{entities['total'] - entities['empty_descriptions']:,}"
    )
    print(
        f"  Empty: "
        f"{entities['empty_descriptions']:,}"
    )

    print("\nReport:")
    print(
        f"  {REPORT_FILE}"
    )

    if passed:

        print(
            "\nFINAL STATUS: PASS"
        )

    else:

        print(
            "\nFINAL STATUS: FAIL"
        )


if __name__ == "__main__":
    main()
