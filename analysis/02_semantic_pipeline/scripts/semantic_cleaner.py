import csv
import json
import os
import re
import subprocess
import sys

csv.field_size_limit(sys.maxsize)

INPUT_FILE = "dataset/cleaned/entities.csv"
OUTPUT_FILE = "dataset/cleaned/entities.csv"
BACKUP_FILE = "dataset/cleaned/entities_structural_backup.csv"

MODEL = "llama3.1:8b"

ARCHIVE_KEYWORDS = (
    "archive",
    "archives",
    "help desk",
    "reference desk",
    "wikiproject",
    "talk page",
    "discussion",
    "template:",
    "category:",
)

MAX_ALIAS_COUNT_FOR_LLM = 50


def normalize(text):
    return " ".join(text.casefold().split())


def parse_aliases(raw):
    if not raw:
        return []

    return [
        alias.strip()
        for alias in raw.split("|")
        if alias.strip()
    ]


def looks_like_archive(alias):
    text = normalize(alias)

    return any(
        keyword in text
        for keyword in ARCHIVE_KEYWORDS
    )


def is_symbol_only(alias):
    return bool(alias) and not any(
        character.isalnum()
        for character in alias
    )


def ask_ollama(entity_name, description, alias):
    prompt = f"""
You are cleaning aliases in a knowledge graph.

Entity name:
{entity_name}

Entity description:
{description[:1500]}

Candidate alias:
{alias}

Determine whether the candidate alias is a valid alternative name,
identifier, spelling variant, abbreviation, transliteration, or
commonly used reference to the SAME entity.

Do NOT judge whether it is merely related to the entity.
It must refer to the same entity.

Reply with ONLY valid JSON:

{{"keep": true}}

or

{{"keep": false}}
"""

    result = subprocess.run(
        [
            "ollama",
            "run",
            MODEL,
            prompt,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Ollama failed:\n{result.stderr}"
        )

    output = result.stdout.strip()

    # ---------------------------------------------------------
    # Extract JSON if the model wrapped it in extra text.
    # ---------------------------------------------------------

    match = re.search(
        r"\{.*\}",
        output,
        re.DOTALL,
    )

    if not match:
        return None

    try:
        decision = json.loads(match.group())

    except json.JSONDecodeError:
        return None

    keep = decision.get("keep")

    if isinstance(keep, bool):
        return keep

    return None


def main():
    print("=" * 60)
    print("LLM-ASSISTED SEMANTIC CLEANING")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(INPUT_FILE)

    if os.path.exists(BACKUP_FILE):
        print(
            f"\nBackup already exists:"
            f"\n  {BACKUP_FILE}"
        )
    else:
        print("\nCreating structural backup...")

        with open(
            INPUT_FILE,
            "r",
            encoding="utf-8",
        ) as src, open(
            BACKUP_FILE,
            "w",
            encoding="utf-8",
        ) as dst:

            for line in src:
                dst.write(line)

    temp_file = OUTPUT_FILE + ".tmp"

    total_entities = 0
    total_aliases = 0
    clean_aliases = 0

    deterministic_removed = 0
    llm_candidates = 0
    llm_kept = 0
    llm_removed = 0
    llm_failed = 0

    archive_candidates = 0
    symbol_candidates = 0
    large_entity_candidates = 0

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
        newline="",
    ) as src, open(
        temp_file,
        "w",
        encoding="utf-8",
        newline="",
    ) as dst:

        reader = csv.DictReader(src)

        writer = csv.DictWriter(
            dst,
            fieldnames=[
                "Entity_ID",
                "Entity_Name",
                "Entity_Aliases",
                "Entity_Description",
            ],
        )

        writer.writeheader()

        for row in reader:
            total_entities += 1

            entity_id = row["Entity_ID"].strip()
            name = row["Entity_Name"].strip()
            aliases = parse_aliases(
                row["Entity_Aliases"]
            )
            description = row["Entity_Description"].strip()

            total_aliases += len(aliases)

            seen = set()
            deterministic_aliases = []

            suspicious_aliases = []

            # -------------------------------------------------
            # Deterministic cleanup
            # -------------------------------------------------

            normalized_name = normalize(name)

            for alias in aliases:
                normalized_alias = normalize(alias)

                if not normalized_alias:
                    deterministic_removed += 1
                    continue

                # Remove canonical name duplicated as alias.
                if normalized_alias == normalized_name:
                    deterministic_removed += 1
                    continue

                # Remove exact normalized duplicates.
                if normalized_alias in seen:
                    deterministic_removed += 1
                    continue

                seen.add(normalized_alias)
                deterministic_aliases.append(alias)

            # -------------------------------------------------
            # Identify suspicious aliases
            # -------------------------------------------------

            for alias in deterministic_aliases:

                suspicious = False

                if looks_like_archive(alias):
                    archive_candidates += 1
                    suspicious = True

                if is_symbol_only(alias):
                    symbol_candidates += 1
                    suspicious = True

                if suspicious:
                    suspicious_aliases.append(alias)

            if len(deterministic_aliases) >= MAX_ALIAS_COUNT_FOR_LLM:
                large_entity_candidates += 1

                # For large alias sets, inspect all aliases.
                suspicious_aliases = list(
                    dict.fromkeys(
                        deterministic_aliases
                    )
                )

            # -------------------------------------------------
            # LLM review
            # -------------------------------------------------

            final_aliases = []

            for alias in deterministic_aliases:

                if alias not in suspicious_aliases:
                    final_aliases.append(alias)
                    continue

                llm_candidates += 1

                decision = ask_ollama(
                    entity_name=name,
                    description=description,
                    alias=alias,
                )

                if decision is True:
                    final_aliases.append(alias)
                    llm_kept += 1

                elif decision is False:
                    llm_removed += 1

                else:
                    # Fail-safe:
                    # preserve the alias when the LLM is uncertain.
                    final_aliases.append(alias)
                    llm_failed += 1

            clean_aliases += len(final_aliases)

            row["Entity_Aliases"] = "|".join(
                final_aliases
            )

            writer.writerow(row)

            if total_entities % 10000 == 0:
                print(
                    f"Processed {total_entities:,} entities | "
                    f"LLM candidates {llm_candidates:,}"
                )

    # ---------------------------------------------------------
    # Replace file only after successful completion.
    # ---------------------------------------------------------

    os.replace(
        temp_file,
        OUTPUT_FILE,
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    removed_total = (
        total_aliases
        - clean_aliases
    )

    report = f"""
SEMANTIC CLEANING REPORT
============================================================

Entities processed:                 {total_entities:,}

ALIASES
------------------------------------------------------------
Original aliases:                   {total_aliases:,}
Final aliases:                      {clean_aliases:,}
Total aliases removed:              {removed_total:,}

Deterministic removals:             {deterministic_removed:,}

LLM candidates:                     {llm_candidates:,}
LLM kept:                           {llm_kept:,}
LLM removed:                        {llm_removed:,}
LLM uncertain/failures:             {llm_failed:,}

SUSPICIOUS CANDIDATES
------------------------------------------------------------
Archive/page-like aliases:          {archive_candidates:,}
Symbol-only aliases:                {symbol_candidates:,}
Large alias-set entities:           {large_entity_candidates:,}

MODEL
------------------------------------------------------------
{MODEL}

OUTPUT
------------------------------------------------------------
{OUTPUT_FILE}

BACKUP
------------------------------------------------------------
{BACKUP_FILE}
"""

    print(report)


if __name__ == "__main__":
    main()
