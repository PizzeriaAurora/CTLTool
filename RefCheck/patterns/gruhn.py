# gruhn_laue_tctl_patterns.py
# All patterns from Gruhn & Laue (2006/2008) with TCTL formulas

gruhn_laue_tctl_patterns = [
    # === Absence Patterns ===
    # "Event p does not occur in [t0, t1]"
    ("Absence", "AG_{[t0,t1]} ¬p", False),

    # "Event p does not occur after event q for [t0, t1]"
    ("AbsenceAfterQ", "AG(q -> AG_{[t0,t1]} ¬p)", False),

    # "Event p does not occur before event r in [t0, t1]"
    ("AbsenceBeforeR", "AG(¬r U_{[t0,t1]} ¬p)", False),

    # === Existence Patterns ===
    # "Event p occurs at least once in [t0, t1]"
    ("Existence", "EF_{[t0,t1]} p", False),

    # "Event p occurs at least once after event q in [t0, t1]"
    ("ExistenceAfterQ", "AG(q -> EF_{[t0,t1]} p)", False),

    # "Event p occurs at least once before event r in [t0, t1]"
    ("ExistenceBeforeR", "AG(¬r U_{[t0,t1]} p)", False),

    # === Universality Patterns ===
    # "Property φ holds at all times in [t0, t1]"
    ("Universality", "AG_{[t0,t1]} φ", False),

    # "Property φ holds at all times after event q in [t0, t1]"
    ("UniversalityAfterQ", "AG(q -> AG_{[t0,t1]} φ)", False),

    # "Property φ holds at all times before event r in [t0, t1]"
    ("UniversalityBeforeR", "AG(¬r U_{[t0,t1]} φ)", False),

    # === Bounded Existence ===
    # "Event p occurs at most once in [t0, t1]"
    ("BoundedExistence", "A[true U_{[t0,t1]} (p ∧ AG_{[t0,t1]} ¬p)]", False),

    # === Precedence Patterns ===
    # "Event q must not occur before event p in [t0, t1]"
    ("Precedence", "AG(q -> (¬p U_{[t0,t1]} q))", False),

    # === Response Patterns ===
    # "Whenever event p occurs, event q must follow within [t0, t1]"
    ("Response", "AG(p -> AF_{[t0,t1]} q)", False),

    # === Bounded Response Patterns ===
    # "Whenever event p occurs, event q must follow within [t0, t1] and before r"
    ("BoundedResponseBeforeR", "AG(p -> (¬r U_{[t0,t1]} q))", False),

    # === Invariance Patterns ===
    # "Whenever condition ψ holds, property φ must also hold in [t0, t1]"
    ("Invariance", "AG_{[t0,t1]} (ψ -> φ)", False),

    # === Stability Patterns ===
    # "If p holds at time t, then it holds for at least the next d time units"
    ("Stability", "AG(p -> AG_{[0,d]} p)", False),

    # === Recurrence Patterns ===
    # "Event p recurs at least every d time units"
    ("Recurrence", "AG(AF_{[0,d]} p)", False),
]

if __name__ == "__main__":
    for name, formula, obs in gruhn_laue_tctl_patterns:
        print(f"{name}: {formula} | Observer: {obs}")
