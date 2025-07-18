# jin_dong_tctl_patterns.py
# All patterns from Dong et al. (2008) with canonical TCTL formulas

jin_dong_tctl_patterns = [
    # === Existence Patterns ===
    # "p occurs at least once within [a, b]"
    ("Existence", "EF_{[a,b]} p", True),

    # "p occurs at least once after q within [a, b]"
    ("ExistenceAfterQ", "AG(q -> EF_{[a,b]} p)", True),

    # "p occurs at least once before r within [a, b]"
    ("ExistenceBeforeR", "AG(¬r U_{[a,b]} p)", True),

    # "p occurs at least once between q and r within [a, b]"
    ("ExistenceBetweenQandR", "AG(q -> (¬r U_{[a,b]} p))", True),

    # === Absence Patterns ===
    # "p does not occur within [a, b]"
    ("Absence", "AG_{[a,b]} ¬p", True),

    # "p does not occur after q within [a, b]"
    ("AbsenceAfterQ", "AG(q -> AG_{[a,b]} ¬p)", True),

    # "p does not occur before r within [a, b]"
    ("AbsenceBeforeR", "AG(¬r U_{[a,b]} ¬p)", True),

    # "p does not occur between q and r within [a, b]"
    ("AbsenceBetweenQandR", "AG(q -> (¬r U_{[a,b]} ¬p))", True),

    # === Universality Patterns ===
    # "φ holds at all times within [a, b]"
    ("Universality", "AG_{[a,b]} φ", True),

    # "φ holds at all times after q within [a, b]"
    ("UniversalityAfterQ", "AG(q -> AG_{[a,b]} φ)", True),

    # "φ holds at all times before r within [a, b]"
    ("UniversalityBeforeR", "AG(¬r U_{[a,b]} φ)", True),

    # "φ holds at all times between q and r within [a, b]"
    ("UniversalityBetweenQandR", "AG(q -> (¬r U_{[a,b]} φ))", True),

    # === Bounded Existence Patterns ===
    # "p occurs at most once within [a, b]"
    ("BoundedExistence", "A[true U_{[a,b]} (p ∧ AG_{[a,b]} ¬p)]", True),

    # === Precedence Patterns ===
    # "q must not occur before p within [a, b]"
    ("Precedence", "AG(q -> (¬p U_{[a,b]} q))", True),

    # "q must not occur before p after r within [a, b]"
    ("PrecedenceAfterR", "AG(r -> AG(q -> (¬p U_{[a,b]} q)))", True),

    # "q must not occur before p before s within [a, b]"
    ("PrecedenceBeforeS", "AG(¬s U_{[a,b]} (q -> (¬p U_{[a,b]} q)))", True),

    # === Response Patterns ===
    # "Whenever p occurs, q must follow within [a, b]"
    ("Response", "AG(p -> AF_{[a,b]} q)", True),

    # "Whenever p occurs after r, q must follow within [a, b]"
    ("ResponseAfterR", "AG(r -> AG(p -> AF_{[a,b]} q))", True),

    # "Whenever p occurs before s, q must follow within [a, b]"
    ("ResponseBeforeS", "AG(¬s U_{[a,b]} (p -> AF_{[a,b]} q))", True),

    # === Bounded Response Patterns ===
    # "Whenever p occurs, q must follow within [a, b] and before r"
    ("BoundedResponseBeforeR", "AG(p -> (¬r U_{[a,b]} q))", True),

    # === Invariance Patterns ===
    # "Whenever ψ holds, φ must also hold within [a, b]"
    ("Invariance", "AG_{[a,b]} (ψ -> φ)", True),

    # === Stability Patterns ===
    # "If p holds at time t, then it holds for at least the next d time units"
    ("Stability", "AG(p -> AG_{[0,d]} p)", True),

    # === Recurrence Patterns ===
    # "p recurs at least every d time units"
    ("Recurrence", "AG(AF_{[0,d]} p)", True),
]

if __name__ == "__main__":
    for name, formula, obs in jin_dong_tctl_patterns:
        print(f"{name}: {formula} | Observer: {obs}")
