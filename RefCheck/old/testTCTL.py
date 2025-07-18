from RefCheck import Parser
from RefCheck.Parser import PARSER, NNFTransformer
from RefCheck import tabta  # Assuming your TimedABTA is named TABTA in tabta.py
def build(formula: str):
        parsed = PARSER.parse(formula)
        nnf = NNFTransformer().to_nnf(parsed)
        return tabta.TABTA(nnf)

def run_tabta_tests():
    # Basic Self-simulation Tests
    basic_formulas = [
        "EF[0,1] p",
        "EF[0,2] q",
        "EF[1,3] (p & q)",
        "AF[0,1] r",
        "AF[2,4] (p | r)",
        "AG[0,2] s",
        "AG[1,5] (s -> p)",
        "EG[0,3] t",
        "EG[2,6] (~t)",
        "EF[0,5] (p & q & r)",
        "AF[1,5] (p -> EF[0,2] q)",
        "AG[0,4] (p | ~q)",
        "EG[3,7] (p & (q | r))",
        "EF[0,0] (~p | q)",
        "AF[0,3] (p & q & ~r)",
        "AG[2,8] ((p -> q) & r)",
        "EG[1,4] (EF[0,1] p)",
        "EF[0,6] (AG[0,2] p)",
        "AF[2,7] (EG[1,3] q)",
        "AG[0,5] (EF[0,2] (p | q))",
    ]

    for f in basic_formulas:
        print(f"Self-simulation: {f}")
        a = build(f)
        ok, size = a.simulate(a)
        print(f"  Simulation relation size: {size}")
        assert ok, f"{f} should simulate itself"
        print("  ✔ Passed")

    # Negation, conjunction, disjunction
    combination_tests = [
        "EF[0,10] (p & q)",
        "EF[2,8] (p | q)",
        "EF[0,5] ~(p & q)",
        "AF[1,5] (p -> EF[0,3] q)",
        "AG[2,4] (p & (q | r))",
        "EG[0,6] (~p | q)",
        "EF[1,7] ((p & q) | r)",
        "AF[0,4] ((p | q) -> r)",
        "AG[3,5] (p & ~q)",
        "EG[2,9] ((p -> r) & (q | s))",
        "EF[0,3] (p | (q & r))",
        "AF[4,8] (~p -> EG[0,2] q)",
        "AG[1,6] (EF[0,3] p & AG[0,2] q)",
        "EG[3,7] ((p & q) -> r)",
        "EF[0,9] (AF[1,3] p)",
        "AF[2,5] (EG[1,2] q)",
        "AG[0,3] (EF[0,1] (p & q))",
        "EG[2,4] (AG[1,2] r)",
        "EF[1,8] (~(p | q))",
        "AF[0,10] (p & (q -> r))",
    ]

    for f in combination_tests:
        print(f"Combo formula: {f}")
        a = build(f)
        ok, _ = a.simulate(a)
        assert ok, f"{f} should simulate itself"
        print("  ✔ Passed")

    # Cross Negative Tests
    negative_pairs = [
        ("EF[0,5] p", "EF[0,5] q"),
        ("AF[1,10] p", "AF[1,10] q"),
        ("EG[2,8] p", "AG[2,8] p"),
        ("EF[0,1] (p & q)", "EF[0,3] p"),
        ("AF[0,5] p", "EF[0,5] p"),
        ("AG[0,4] p", "EG[0,4] p"),
        ("EG[2,3] q", "EG[1,3] (p | q)"),
        ("EF[2,6] r", "EF[2,6] (p & r)"),
        ("AF[0,2] (p | q)", "AF[0,2] p"),
        ("AG[1,5] (~p)", "AG[1,5] p"),
        ("EG[0,5] (p -> q)", "EG[0,5] q"),
        ("EF[0,4] (p & ~q)", "EF[0,4] (p | q)"),
        ("AF[2,7] (q & r)", "AF[2,7] q"),
        ("AG[3,5] (p | ~r)", "AG[3,5] r"),
        ("EG[1,4] (p & q)", "EG[1,4] p"),
        ("EF[0,10] (p -> q)", "EF[0,10] q"),
        ("AF[1,3] (p & (q | r))", "AF[1,3] q"),
        ("AG[0,6] (p & r)", "AG[0,6] (p | q)"),
        ("EG[2,8] (p | q)", "AG[2,8] (p | q)"),
        ("EF[1,5] s", "EF[1,5] r"),
    ]
    for f1, f2 in negative_pairs:
        print(f"Negative simulation: {f1} vs {f2}")
        a1 = build(f1)
        a2 = build(f2)
        ok, _ = a1.simulate(a2)
        assert not ok, f"{f1} should NOT simulate {f2}"
        print("  ✔ Passed")

    # Cross Positive Simulation
    positive_pairs = [
        ("EF[0,10] p", "EF[0,10] (p | q)"),
        ("AG[1,5] (p & q)", "AG[1,5] p"),
        ("EF[0,5] (p & q)", "EF[0,5] p"),
        ("AF[0,10] p", "AF[0,10] (p | q)"),
        ("AG[2,6] (p -> q)", "AG[2,6] (~p | q)"),
        ("EF[1,4] (p | (q & r))", "EF[1,4] p"),
        ("AF[0,3] (p & (q | r))", "AF[0,3] (p | q)"),
        ("EG[2,5] (p & q)", "EG[2,5] p"),
        ("EF[3,7] (AG[0,2] p)", "EF[3,7] p"),
        ("AF[1,2] (EG[0,1] q)", "AF[1,2] q"),
        ("AG[0,8] (EF[0,2] p)", "AG[0,8] p"),
        ("EG[1,3] (p -> EF[0,1] q)", "EG[1,3] (~p | q)"),
        ("EF[0,9] ((p & q) | r)", "EF[0,9] (p | q)"),
        ("AF[2,5] (~p | q)", "AF[2,5] q"),
        ("AG[1,4] (p & q & r)", "AG[1,4] (p & q)"),
        ("EG[0,6] (p | q | r)", "EG[0,6] (p | q)"),
        ("EF[2,8] (AF[0,3] p)", "EF[2,8] p"),
        ("AF[1,5] (EG[1,3] q)", "AF[1,5] q"),
        ("AG[0,4] (EF[0,1] (p | q))", "AG[0,4] (p | q)"),
        ("EG[2,7] (AG[1,2] r)", "EG[2,7] r"),
    ]

    for f1, f2 in positive_pairs:
        print(f"Positive simulation: {f1} simulates {f2}")
        a1 = build(f1)
        a2 = build(f2)
        ok, _ = a1.simulate(a2)
        assert ok, f"{f1} SHOULD simulate {f2}"
        print("  ✔ Passed")

    # Tautology and Contradiction
    taut_contra = [
        "AG[0,10] (p | ~p)",  # tautology
        "AG[0,10] (p & ~p)",  # contradiction
        "EF[0,0] (p | ~p)",   # always true eventually
        "AF[1,1] (p & ~p)",   # always false eventually
    ]

    for f in taut_contra:
        print(f"Taut/Contra test: {f}")
        a = build(f)
        ok, _ = a.simulate(a)
        assert ok, f"{f} should simulate itself"
        print("  ✔ Passed")

    # Nested Timed Logic
    nested_formulas = [
        "EF[0,3] EF[1,2] p",
        "AF[0,5] AG[0,5] p",
        "EF[1,4] AF[0,3] p",
        "AG[0,6] (p -> EF[0,2] q)",
        "EG[2,7] (q & AF[0,1] r)",
    ]

    for f in nested_formulas:
        print(f"Nested: {f}")
        a = build(f)
        ok, _ = a.simulate(a)
        assert ok, f"{f} should simulate itself"
        print("  ✔ Passed")

    print("\n🎉 All TABTA tests passed successfully.")

def debug_one(f):
    a=build(f)
    print(a)
    ok, _ = a.simulate(a)
    assert ok, f"{f} should simulate itself"
    print("  ✔ Passed")
    


def debug_negative(f1,f2):
        print(f"Negative simulation: {f1} vs {f2}")
        a1 = build(f1)
        a2 = build(f2)
        print(a1, a2)
        ok, _ = a1.simulate(a2)
        assert not ok, f"{f1} should NOT simulate {f2}"
        print("  ✔ Passed")



if __name__ == "__main__":
   # run_tabta_tests()
    debug_one("EF[2,6] (p & r)")
    debug_negative("EF[2,6] r", "EF[2,6] (p & r)")
    