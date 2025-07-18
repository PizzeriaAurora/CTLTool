# konrad_cheng_tctl_patterns.py
# All patterns from Konrad & Cheng (2005) with TCTL formulas

konrad_cheng_tctl_patterns = [
    # === Existence Patterns ===
    ("Existence", "EF p", True),
    ("BoundedExistence", "EF_{[d1,d2]} p", True),
    ("ExistenceAfterQ", "A G (q -> EF p)", True),
    ("BoundedExistenceAfterQ", "A G (q -> EF_{[d1,d2]} p)", True),
    ("ExistenceBeforeR", "A G (¬r U p)", True),
    ("BoundedExistenceBeforeR", "A G (¬r U_{[d1,d2]} p)", True),
    ("ExistenceBetweenQandR", "A G (q -> (¬r U p))", True),
    ("BoundedExistenceBetweenQandR", "A G (q -> (¬r U_{[d1,d2]} p))", True),

    # === Absence Patterns ===
    ("Absence", "A G ¬p", True),
    ("BoundedAbsence", "A G_{[d1,d2]} ¬p", True),
    ("AbsenceAfterQ", "A G (q -> A G ¬p)", True),
    ("BoundedAbsenceAfterQ", "A G (q -> A G_{[d1,d2]} ¬p)", True),
    ("AbsenceBeforeR", "A G (¬r U ¬p)", True),
    ("BoundedAbsenceBeforeR", "A G (¬r U_{[d1,d2]} ¬p)", True),
    ("AbsenceBetweenQandR", "A G (q -> (¬r U ¬p))", True),
    ("BoundedAbsenceBetweenQandR", "A G (q -> (¬r U_{[d1,d2]} ¬p))", True),

    # === Universality Patterns ===
    ("Universality", "A G p", True),
    ("BoundedUniversality", "A G_{[d1,d2]} p", True),
    ("UniversalityAfterQ", "A G (q -> A G p)", True),
    ("BoundedUniversalityAfterQ", "A G (q -> A G_{[d1,d2]} p)", True),
    ("UniversalityBeforeR", "A G (¬r U p)", True),
    ("BoundedUniversalityBeforeR", "A G (¬r U_{[d1,d2]} p)", True),
    ("UniversalityBetweenQandR", "A G (q -> (¬r U p))", True),
    ("BoundedUniversalityBetweenQandR", "A G (q -> (¬r U_{[d1,d2]} p))", True),

    # === Response Patterns ===
    ("Response", "A G (q -> A F p)", True),
    ("BoundedResponse", "A G (q -> A F_{[d1,d2]} p)", True),
    ("ResponseAfterQ", "A G (r -> A G (q -> A F p))", True),
    ("BoundedResponseAfterQ", "A G (r -> A G (q -> A F_{[d1,d2]} p))", True),
    ("ResponseBeforeR", "A G (¬r U (q -> A F p))", True),
    ("BoundedResponseBeforeR", "A G (¬r U_{[d1,d2]} (q -> A F_{[d1,d2]} p))", True),
    ("ResponseBetweenQandR", "A G (q -> (¬r U (s -> A F p)))", True),
    ("BoundedResponseBetweenQandR", "A G (q -> (¬r U_{[d1,d2]} (s -> A F_{[d1,d2]} p)))", True),

    # === Precedence Patterns ===
    ("Precedence", "A G (p -> (¬q U p))", True),
    ("BoundedPrecedence", "A G (p -> (¬q U_{[d1,d2]} p))", True),
    ("PrecedenceAfterQ", "A G (r -> A G (p -> (¬q U p)))", True),
    ("BoundedPrecedenceAfterQ", "A G (r -> A G (p -> (¬q U_{[d1,d2]} p)))", True),
    ("PrecedenceBeforeR", "A G (¬r U (p -> (¬q U p)))", True),
    ("BoundedPrecedenceBeforeR", "A G (¬r U_{[d1,d2]} (p -> (¬q U_{[d1,d2]} p)))", True),
    ("PrecedenceBetweenQandR", "A G (q -> (¬r U (s -> (¬p U s))))", True),
    ("BoundedPrecedenceBetweenQandR", "A G (q -> (¬r U_{[d1,d2]} (s -> (¬p U_{[d1,d2]} s))))", True),

    # === Invariant Patterns ===
    ("Invariant", "A G (q -> p)", True),
    ("BoundedInvariant", "A G_{[d1,d2]} (q -> p)", True),
    ("InvariantAfterQ", "A G (r -> A G (q -> p))", True),
    ("BoundedInvariantAfterQ", "A G (r -> A G_{[d1,d2]} (q -> p))", True),
    ("InvariantBeforeR", "A G (¬r U (q -> p))", True),
    ("BoundedInvariantBeforeR", "A G (¬r U_{[d1,d2]} (q -> p))", True),
    ("InvariantBetweenQandR", "A G (q -> (¬r U (s -> p)))", True),
    ("BoundedInvariantBetweenQandR", "A G (q -> (¬r U_{[d1,d2]} (s -> p)))", True),

    # === Recurrence Patterns ===
    ("Recurrence", "A G (A F p)", True),
    ("BoundedRecurrence", "A G (A F_{[0,T]} p)", True),
    ("RecurrenceAfterQ", "A G (q -> A G (A F p))", True),
    ("BoundedRecurrenceAfterQ", "A G (q -> A G (A F_{[0,T]} p))", True),
    ("RecurrenceBeforeR", "A G (¬r U (A F p))", True),
    ("BoundedRecurrenceBeforeR", "A G (¬r U_{[0,T]} (A F_{[0,T]} p))", True),
    ("RecurrenceBetweenQandR", "A G (q -> (¬r U (A F p)))", True),
    ("BoundedRecurrenceBetweenQandR", "A G (q -> (¬r U_{[0,T]} (A F_{[0,T]} p)))", True),

    # === Stability Patterns ===
    ("Stability", "A G (p -> A G p)", True),
    ("BoundedStability", "A G (p -> A G_{[d1,d2]} p)", True),
    ("StabilityAfterQ", "A G (q -> A G (p -> A G p))", True),
    ("BoundedStabilityAfterQ", "A G (q -> A G (p -> A G_{[d1,d2]} p))", True),
    ("StabilityBeforeR", "A G (¬r U (p -> A G p))", True),
    ("BoundedStabilityBeforeR", "A G (¬r U_{[d1,d2]} (p -> A G_{[d1,d2]} p))", True),
    ("StabilityBetweenQandR", "A G (q -> (¬r U (s -> A G s)))", True),
    ("BoundedStabilityBetweenQandR", "A G (q -> (¬r U_{[d1,d2]} (s -> A G_{[d1,d2]} s)))", True),

    # === Duration Patterns ===
    ("MaximumDuration", "A G (p -> A G_{[0,T]} p)", True),
    ("MinimumDuration", "A G (p -> A F_{[T,∞)} ¬p)", True),
]

if __name__ == "__main__":
    for name, formula, obs in konrad_cheng_tctl_patterns:
        print(f"{name}: {formula} | Observer: {obs}")
