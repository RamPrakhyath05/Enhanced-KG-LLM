NODES = {
    "Albert Einstein": {"type": "Person"},
    "Nobel Prize": {"type": "Award"},
    "Physics": {"type": "Field"},
    "Germany": {"type": "Country"},
}

# data/graph_data.py

EDGES = [

    # =========================
    # PEOPLE → FIELDS
    # =========================
    ("Albert Einstein", "FIELD", "Physics", {}),
    ("Isaac Newton", "FIELD", "Physics", {}),
    ("Marie Curie", "FIELD", "Physics", {}),
    ("Marie Curie", "FIELD", "Chemistry", {}),
    ("Niels Bohr", "FIELD", "Physics", {}),
    ("Alan Turing", "FIELD", "Computer Science", {}),
    ("Ada Lovelace", "FIELD", "Computer Science", {}),
    ("Nikola Tesla", "FIELD", "Electrical Engineering", {}),

    # =========================
    # PEOPLE → AWARDS
    # =========================
    ("Albert Einstein", "WON_AWARD", "Nobel Prize", {"year": 1921}),
    ("Marie Curie", "WON_AWARD", "Nobel Prize", {"year": 1903}),
    ("Marie Curie", "WON_AWARD", "Nobel Prize", {"year": 1911}),
    ("Niels Bohr", "WON_AWARD", "Nobel Prize", {"year": 1922}),
    ("Alan Turing", "WON_AWARD", "Turing Award", {"year": 1966}),
    ("Ada Lovelace", "WON_AWARD", "Computer History Award", {}),
    ("Nikola Tesla", "WON_AWARD", "IEEE Edison Medal", {}),

    # =========================
    # AWARDS → CATEGORIES
    # =========================
    ("Nobel Prize", "CATEGORY", "Physics", {}),
    ("Nobel Prize", "CATEGORY", "Chemistry", {}),
    ("Turing Award", "CATEGORY", "Computer Science", {}),
    ("IEEE Edison Medal", "CATEGORY", "Electrical Engineering", {}),

    # =========================
    # PEOPLE → BIRTH / DEATH
    # =========================
    ("Albert Einstein", "BORN_IN", "Germany", {}),
    ("Isaac Newton", "BORN_IN", "England", {}),
    ("Marie Curie", "BORN_IN", "Poland", {}),
    ("Niels Bohr", "BORN_IN", "Denmark", {}),
    ("Alan Turing", "BORN_IN", "England", {}),
    ("Ada Lovelace", "BORN_IN", "England", {}),
    ("Nikola Tesla", "BORN_IN", "Croatia", {}),

    # =========================
    # PEOPLE → INSTITUTIONS
    # =========================
    ("Albert Einstein", "AFFILIATED_WITH", "Princeton University", {}),
    ("Alan Turing", "AFFILIATED_WITH", "University of Manchester", {}),
    ("Marie Curie", "AFFILIATED_WITH", "University of Paris", {}),
    ("Isaac Newton", "AFFILIATED_WITH", "University of Cambridge", {}),
    ("Niels Bohr", "AFFILIATED_WITH", "University of Copenhagen", {}),

    # =========================
    # INSTITUTIONS → COUNTRIES
    # =========================
    ("Princeton University", "LOCATED_IN", "United States", {}),
    ("University of Manchester", "LOCATED_IN", "United Kingdom", {}),
    ("University of Paris", "LOCATED_IN", "France", {}),
    ("University of Cambridge", "LOCATED_IN", "United Kingdom", {}),
    ("University of Copenhagen", "LOCATED_IN", "Denmark", {}),

    # =========================
    # FIELDS → RELATED FIELDS
    # =========================
    ("Physics", "RELATED_TO", "Mathematics", {}),
    ("Computer Science", "RELATED_TO", "Mathematics", {}),
    ("Computer Science", "RELATED_TO", "Artificial Intelligence", {}),
    ("Electrical Engineering", "RELATED_TO", "Physics", {}),
    ("Chemistry", "RELATED_TO", "Physics", {}),

    # =========================
    # PEOPLE → CONTRIBUTIONS
    # =========================
    ("Albert Einstein", "KNOWN_FOR", "Theory of Relativity", {}),
    ("Isaac Newton", "KNOWN_FOR", "Laws of Motion", {}),
    ("Marie Curie", "KNOWN_FOR", "Radioactivity", {}),
    ("Alan Turing", "KNOWN_FOR", "Turing Machine", {}),
    ("Nikola Tesla", "KNOWN_FOR", "Alternating Current", {}),

]
