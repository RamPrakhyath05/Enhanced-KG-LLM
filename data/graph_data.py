NODES = {
    "Albert Einstein": {"type": "Person"},
    "Nobel Prize": {"type": "Award"},
    "Physics": {"type": "Field"},
    "Germany": {"type": "Country"},
}

EDGES = [
    ("Albert Einstein", "WON_AWARD", "Nobel Prize", {"year": 1921}),
    ("Albert Einstein", "FIELD", "Physics", {}),
    ("Albert Einstein", "BORN_IN", "Germany", {}),
    ("Nobel Prize", "CATEGORY", "Physics", {}),
] 