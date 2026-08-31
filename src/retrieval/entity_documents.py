import json

ENTITY_QUERY = """
MATCH (e:Entity)
RETURN
    e.Entity_ID AS id,
    e.Entity_Name AS name,
    e.Entity_Aliases AS aliases,
    e.Entity_Description AS description
"""


def normalize_aliases(aliases):
    if aliases is None:
        return []

    if isinstance(aliases, list):
        return [str(a).strip() for a in aliases if a]

    if isinstance(aliases, str):
        try:
            parsed = json.loads(aliases)

            if isinstance(parsed, list):
                return [str(a).strip() for a in parsed if a]

        except json.JSONDecodeError:
            pass

        return [aliases.strip()] if aliases.strip() else []

    return [str(aliases).strip()]


def entity_to_document(entity):
    aliases = normalize_aliases(entity.get("aliases"))

    name = entity.get("name") or ""
    description = entity.get("description") or ""

    alias_text = ", ".join(aliases)

    text = (
        f"Name: {name}\n"
        f"Aliases: {alias_text}\n"
        f"Description: {description}"
    )

    return {
        "id": entity["id"],
        "name": name,
        "text": text,
    }
