# vogel_patterns.py
# All patterns from Vogel et al. (2023) catalogue
# Source: https://github.com/ThomasVogel/pattern-catalog-uppaal

vogel_patterns = [
    # Existence Patterns
    ("Existence", "EF[0,T] p", True),
    ("ExistenceAfterEvent", "event1 -> EF[0,T] p", True),
    ("ExistenceBeforeEvent", "EF[0,T] (p & before event2)", True),
    ("ExistenceBetweenEvents", "between(event1, event2): EF[0,T] p", True),

    # Absence Patterns
    ("Absence", "AG[0,T] ~p", True),
    ("AbsenceAfterEvent", "event1 -> AG[0,T] ~p", True),
    ("AbsenceBeforeEvent", "AG[0,T] (~p & before event2)", True),
    ("AbsenceBetweenEvents", "between(event1, event2): AG[0,T] ~p", True),

    # Universality Patterns
    ("Universality", "AG[0,T] p", True),
    ("UniversalityAfterEvent", "event1 -> AG[0,T] p", True),
    ("UniversalityBeforeEvent", "AG[0,T] (p & before event2)", True),
    ("UniversalityBetweenEvents", "between(event1, event2): AG[0,T] p", True),

    # Response Patterns
    ("Response", "AG(trigger -> AF[0,T] response)", True),
    ("ResponseAfterEvent", "event1 -> AG(trigger -> AF[0,T] response)", True),
    ("ResponseBeforeEvent", "AG(trigger -> AF[0,T] (response & before event2))", True),
    ("ResponseBetweenEvents", "between(event1, event2): AG(trigger -> AF[0,T] response)", True),

    # Precedence Patterns
    ("Precedence", "AG(event2 -> A[~event1 U event2])", True),
    ("PrecedenceAfterEvent", "event3 -> AG(event2 -> A[~event1 U event2])", True),
    ("PrecedenceBeforeEvent", "AG(event2 -> A[~event1 U (event2 & before event4)])", True),
    ("PrecedenceBetweenEvents", "between(event3, event4): AG(event2 -> A[~event1 U event2])", True),

    # Bounded Existence Patterns
    ("BoundedExistence", "A[true U[0,T] (p & AG[0,T] ~p)]", True),
    ("BoundedExistenceAfterEvent", "event1 -> A[true U[0,T] (p & AG[0,T] ~p)]", True),
    ("BoundedExistenceBeforeEvent", "A[true U[0,T] (p & AG[0,T] ~p & before event2)]", True),
    ("BoundedExistenceBetweenEvents", "between(event1, event2): A[true U[0,T] (p & AG[0,T] ~p)]", True),

    # Bounded Response Patterns
    ("BoundedResponse", "AG(trigger -> AF[0,T] response)", True),
    ("BoundedResponseAfterEvent", "event1 -> AG(trigger -> AF[0,T] response)", True),
    ("BoundedResponseBeforeEvent", "AG(trigger -> AF[0,T] (response & before event2))", True),
    ("BoundedResponseBetweenEvents", "between(event1, event2): AG(trigger -> AF[0,T] response)", True),

    # Bounded Universality Patterns
    ("BoundedUniversality", "AG[0,T] p", True),
    ("BoundedUniversalityAfterEvent", "event1 -> AG[0,T] p", True),
    ("BoundedUniversalityBeforeEvent", "AG[0,T] (p & before event2)", True),
    ("BoundedUniversalityBetweenEvents", "between(event1, event2): AG[0,T] p", True),

    # Bounded Precedence Patterns
    ("BoundedPrecedence", "AG(event2 -> A[~event1 U[0,T] event2])", True),
    ("BoundedPrecedenceAfterEvent", "event3 -> AG(event2 -> A[~event1 U[0,T] event2])", True),
    ("BoundedPrecedenceBeforeEvent", "AG(event2 -> A[~event1 U[0,T] (event2 & before event4)])", True),
    ("BoundedPrecedenceBetweenEvents", "between(event3, event4): AG(event2 -> A[~event1 U[0,T] event2])", True),

    # Invariance Patterns
    ("Invariance", "AG[0,T] (cond -> prop)", True),
    ("InvarianceAfterEvent", "event1 -> AG[0,T] (cond -> prop)", True),
    ("InvarianceBeforeEvent", "AG[0,T] ((cond -> prop) & before event2)", True),
    ("InvarianceBetweenEvents", "between(event1, event2): AG[0,T] (cond -> prop)", True),

    # Stability Patterns
    ("Stability", "AG(p -> AG[0,T] p)", True),
    ("StabilityAfterEvent", "event1 -> AG(p -> AG[0,T] p)", True),
    ("StabilityBeforeEvent", "AG(p -> AG[0,T] (p & before event2))", True),
    ("StabilityBetweenEvents", "between(event1, event2): AG(p -> AG[0,T] p)", True),

    # Recurrence Patterns
    ("Recurrence", "AG(p -> AF[0,T] p)", True),
    ("RecurrenceAfterEvent", "event1 -> AG(p -> AF[0,T] p)", True),
    ("RecurrenceBeforeEvent", "AG(p -> AF[0,T] (p & before event2))", True),
    ("RecurrenceBetweenEvents", "between(event1, event2): AG(p -> AF[0,T] p)", True),
]

# For convenience, you can access all pattern names, formulas, and observer status:
if __name__ == "__main__":
    for name, formula, obs in vogel_patterns:
        print(f"{name}: {formula} | Observer: {obs}")
