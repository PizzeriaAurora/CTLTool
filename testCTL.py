# ==============================================================================
#  IMPORTS
# ==============================================================================
# This script relies on the ABTA implementation and parser utilities from the
# local 'RefCheck' package.
from RefCheck import abta
from RefCheck.Parser import PARSER,NNFTransformer, TRANSF


# ==============================================================================
#  HELPER FUNCTIONS
# ==============================================================================

def build(formula: str) -> abta.ABTA:
    """
    A simple factory function that encapsulates the creation of an ABTA
    from a given temporal logic formula string.

    Args:
        formula: The string representation of the formula.

    Returns:
        An initialized ABTA object corresponding to the formula.
    """
    return abta.ABTA(formula)


# ==============================================================================
#  MAIN TEST SUITE
# ==============================================================================

def run_tests():
    """
    Runs a comprehensive suite of unit tests for ABTA construction and simulation.
    The tests are structured into three primary categories to ensure correctness:

    1. Self-Simulation: Verifies that every automaton correctly simulates itself.
       This is a fundamental sanity check for the entire construction and
       simulation pipeline.

    2. Negative Simulation: Tests pairs of automata (A, B) where the language
       of A is NOT a subset of the language of B. The simulation check
       A.is_simulated_by(B) should correctly return False.

    3. Positive Simulation: Tests pairs of automata (A, B) where the language
       of A IS a subset of the language of B (i.e., A implies B). The simulation
       check A.is_simulated_by(B) should correctly return True.
    """
    print("=============================================")
    print("=         RUNNING COMPREHENSIVE SUITE       =")
    print("=============================================")

    # ------------------------------------------------------------------------------
    # Category 1: Self-Simulation Tests (Sanity Check)
    # Every validly constructed automaton should simulate itself. This provides
    # a baseline confirmation that the parser, NNF transformer, ABTA constructor,
    # and simulator are functioning correctly for a wide range of formulas.
    # ------------------------------------------------------------------------------
    self_simulation_tests = [
        ("Atomic", "p"),
        ("True","true"),
        ("Negated Atomic", "!p"),
        ("Simple Conjunction", "p & q"),
        ("Simple Disjunction", "p | q"),
        ("Simple Implication", "p -> q"),
        ("Basic EF", "EF p"),
        ("Basic EG", "EG p"),
        ("Basic AF", "AF p"),
        ("Basic AG", "AG p"),
        ("Basic AG", "AG p>9"),
        ("Basic EU", "E(p U q)"),
        ("Nested EF", "EF EF p"),
        ("Nested AG", "AG AG p"),
        ("Nested AF of AG", "AF AG p"),
        ("Nested EG of EF", "EG EF p"),
        ("Alternating AG EF", "AG EF p"), # Important: "infinitely often"
        ("Alternating EF AG", "EF AG p"), # Important: "stabilization"
        ("Complex Propositional", "AG (p & q -> r)"),
        ("Complex Nested Temporal", "AG (p -> EF (q | r))"),
        ("De Morgan's Law 1", "!(p & q)"),
        ("De Morgan's Law 2", "!(p | q)"),
        ("Tautology", "p | !p"),
        ("Contradiction", "p & !p"),
        ("Long Conjunction", "p & q & r & s"),
        ("Long Disjunction", "p | q | r | s"),
        ("Deeply Nested EF", "EF (p & EF(q & EF r))"),
        ("Deeply Nested AG", "AG (p -> AG(q -> AG r))"),
        ("Complex Until", "E((p|q) U (r&s))"),
        #("Complex Release", "A[(p&q) W (r|s)]"), # Assuming 'W' is Week Until
    ]

    print("\n--- Category 1: Self-Simulation Tests ---")
    for name, formula in self_simulation_tests:
        print(f"Testing (Self): {name} - `{formula}`")
        automaton = build(formula)
        # An automaton must always be simulated by an identical copy of itself.
        b, l = automaton.is_simulated_by(automaton)
        assert b, f"FAIL: Automaton for `{formula}` should simulate itself!"
        #print(f"  ✔ Passed (size: {l})")

 # ------------------------------------------------------------------------------
    # Category 2: Negative Simulation Tests (L(A) ⊈ L(B))
    # These tests verify that the simulation correctly fails when the language of
    # the first formula is not a subset of the second. This ensures the check is
    # not overly permissive.
    # ------------------------------------------------------------------------------
    negative_simulation_tests = [
        ("Different Atoms", "p", "q"),
        ("Disjunction vs Conjunction", "p | q", "p & q"),
        ("Simple vs. Stricter", "AG p", "AG (p & q)"),
        ("Existential vs. Universal", "EF p", "AG p"),
        ("Universal vs. Stricter Universal", "AF p", "AG p"),
        ("EF Specific vs. EF General", "EF p | EF q", "EF p"),
        ("Order Matters in Until", "E(p U q)", "E(q U p)"),       
        ("Stronger Consequent", "AG (p -> q)", "AG (p -> (q&r))"),
        ("EF (p&q) vs EF p & EF q", "EF p & EF q", "EF (p & q)"), # Two paths are not one path
        ("Different Nested Paths", "EF (p & EF q)", "EF (q & EF p)"),
        ("Tautology vs. Non-Tautology", "p | !p", "q"),
        ("Subset is wrong way (Prop)", "p", "p & q"),
        ("Subset is wrong way (Temp)", "EF p", "EF (p & q)"),
    ]

    print("\n--- Category 2: Negative Simulation Tests ---")
    for name, f1, f2 in negative_simulation_tests:
        print(f"Testing (Negative): {name} - `{f1}` vs `{f2}`")
        auto1 = build(f1)
        auto2 = build(f2)
        b, _ = auto2.is_simulated_by(auto1)
        assert not b, f"FAIL: {name} `{f1}` should NOT be implied by `{f2}`"
        #b, _ = auto1.simulate_delayed(auto2)
        #assert not b, f"FAIL: `{f1}` should NOT simulate (delayed) `{f2}`"
        ##print("  ✔ Passed")


 # ------------------------------------------------------------------------------
    # Category 3: Positive Simulation Tests (L(A) ⊆ L(B))
    # These tests verify that the simulation correctly succeeds for valid logical
    # implications and equivalences.
    # ------------------------------------------------------------------------------
    positive_simulation_tests = [
        #("Conjunction vs Disjunction", "p & q", "p | q"),
        ("Stricter vs. Simple", "AG (p & q)", "AG p"),
        ("Conjunction implies one part", "AG(p & q)", "AG p"),
        #("Weaker Precedent", "AG (p -> r)", "AG (p & q -> r)"),
        ("AG implies AF", "AG p", "AF p"),
        ("AG implies EF", "AG p", "EF p"),
        ("AG>2 implies AGp>5", "AG p>5", "AG p>2"),
        ("AG>2 implies AGp>5", "EF p>5", "EF p>2"),
        #("EF Specific vs. EF General", "EF p", "EF (p | q)"),
        ("Logical Equivalence (Implication)", "p -> q", "!p | q"),
        ("Logical Equivalence (De Morgan)", "!(p | q)", "!p & !q"),
        ("Stronger Precedent", "AG ((p&q) -> r)", "AG (p -> r)"),
        ("Weaker Consequent", "AG (p -> (q&r))", "AG (p -> q)"),
        #("Stabilization implies Inf. Often", "EF AG p", "AG EF p"),
        ("Double Negation", "p", "!!p"),
        #("Idempotence (AND)", "p & p", "p"),
        #("Idempotence (OR)", "p | p", "p"),
        ("Nested implication", "AG (p & q)", "AG p & AG q"), # AG distributes over &
        #("Nested implication EF", "EF (p | q)", "(EF p | EF q)"), # EF distributes over |
        ("Until Implication", "E(p U (q&r))", "E(p U q)"),
        #("EF(p&q) implies EF p & EF q", "EF (p & q)", "EF(EF p & EF q)"),
        ("Very specific implies very general", "AG(p & q & r)", "AF(p | q | r)"),
        ("Universal path vs Existential path", "AF p", "EF p"),
        ("Semantic Transitivity", "AG (p > 10)", "AG (p > 5)"),
    ("Semantic Conjunction", "AG (p > 5 & p < 15)", "AG (p != 20)"),
    ("Semantic Disjunction", "EF (p > 10 | p < 0)", "EF (p != 5)"),
    ("Nested AG", "AG AG p", "AG p"),
    ("Until Implies Finally", "E(p U q)", "EF q"),
    ("Stronger Until Path Condition", "E((p & r) U q)", "E(p U q)"),
    ("AF AG implies EF AG", "AF AG p", "EF AG p"),
    ("AG Distributivity over Implication", "AG (p -> q)", "(AG p -> AG q)"),
    ("Temporal De Morgan Law 1", "!(EG !p)", "AF p"),
    ("Temporal De Morgan Law 2", "!(AF !p)", "EG p"),
    ("Absorption Law 1", "p & (p | q)", "p"),
    ("Absorption Law 2", "p | (p & q)", "p"),
    ("AG over Semantic AND", "AG (p>10 & q<0)", "AG(p>5) & AG(q<5)"),
    #("EF over Semantic OR", "EF (p==1 | q==2)", "EF(p==1) | EF(q==2)"),
    ("Contraction", "AG (p -> (p -> q))", "AG (p -> q)"),
    ("Monotonicity of EF", "AG (p -> q)", "EF p -> EF q"),
    ("Monotonicity of AG", "AG (p -> q)", "AG p -> AG q"),
    #("Existential vs Universal Until", "A[p U q]", "E[p U q]"),
    ("Simplification of Until", "E( true U p )", "EF p"),
    ("Simplification of Globally", "AG(p & true)", "AG p"),
    ("Introduction of Disjunction", "EF p", "EF (p | (q & !q))"),
    ("Complex Path Implication", "AG(p -> AF q)", "AG p -> AF q"),
    ("Semantic Equality Implication", "AG (p == 10)", "AG (p >= 10 & p <= 10)"),
    ("Trivial Path Condition", "E((p & q) U r)", "E(p U r)"),
    ("Trivial Goal Condition", "E(p U (q & r))", "E(p U q)"),
    ("Nested EF", "EF EF p", "EF p"),
    ("AG implies self", "AG p", "p"),
    ("EF and AG interaction", "EF AG (p & q)", "EF AG p & EF AG q"),
    ("Implication within EF", "EF (p & (p->q))", "EF q"),
    ("Semantic exclusion", "AG (p > 5)", "AG (p != 4)"),
    ("Universal to Existential Final", "AF p", "EF p"),
    ("Universal to Existential Global", "AG p", "EG p"), # Only if model is non-empty
    #("Expanding Disjunction", "AG (p | q)", "AG p | AF q"), # This is a more complex theorem
    ("Self-fulfilling Until", "E(q U p)", "p"),
    ("Strengthening antecedent", "AG (q -> r)", "AG ((p & q) -> r)"),
    #("Combining goals", "AF p & AF q", "AF (p & AF q) | AF (q & AF p)"),
    #("Redundant Global", "AG p & AG q", "AG (p & AG q)"),
    ("Existential path strengthening", "EF(p & q)", "EF p"),

    ]

    print("\n--- Category 3: Positive Simulation Tests (A SHOULD simulate B) ---")
    for name, f1, f2 in positive_simulation_tests:
        print(f"Testing (Positive): {name} - `{f1}` vs `{f2}`")
        auto1 = build(f1)
        auto2 = build(f2)
        b, _ = auto2.is_simulated_by(auto1)
        assert b, f"FAIL:{name} `{f1}` SHOULD simulate `{f2}`"

    print("\n=============================================")
    print("=          ALL COMPREHENSIVE TESTS PASSED          =")
    print("=============================================")

# ==============================================================================
#  DEBUGGING AND AD-HOC TEST FUNCTIONS
# ==============================================================================


def debug_one(f: str):
    """Helper to debug the self-simulation of a single formula."""
    a = build(f)
    print(a)
    ok, _ = a.is_simulated_by(a)
    assert ok, f"{f} should simulate itself"
    print("  ✔ Passed")
    

def debug_negative(f1: str, f2: str):
    """Helper to debug a single negative simulation case."""
    print(f"Negative simulation: {f1} vs {f2}")
    a1 = build(f1)
    a2 = build(f2)
    print(a1, a2)
    ok, _ = a1.is_simulated_by(a2)
    assert not ok, f"{f1} should NOT be simulated by {f2}"
    print("  ✔ Passed")



def debug_positive(f1: str, f2: str):
    """Helper to debug a single positive simulation case."""
    print(f"Positive simulation: {f1} vs {f2}")
    a1 = build(f1)
    a2 = build(f2)
    print(a1, a2)
    # Check if a1 is simulated by a2, which corresponds to L(a1) ⊆ L(a2).
    ok, _ = a1.is_simulated_by(a2)
    assert ok, f"{f1} should be simulated by {f2}"
    print("  ✔ Passed")

def runtestT2():
    """
    Runs a large-scale pairwise refinement (simulation) check among a predefined
    set of formulas coming from T". This is useful for finding non-obvious relationships
    in a batch of properties.
    """
    # A dictionary of example formulas for batch testing.
    EXAMPLES = {
        1:'AG(varA != 1 | AF(varR == 1))',
        2:'EF(varA == 1 & EG(varR != 5))',
        3:'AG(varA != 1 | EF(varR == 1))',
        4:'EF(varA == 1 & AG(varR != 1))',
        5:'AG(varS != 1 | AF(varU == 1))',
        6:'EF(varS == 1 | EG(varU != 1))',
        7:'AG(varS != 1 | EF(varU == 1))',
        8:'EF(varS == 1 & AG(varU != 1))',
        9:'AG(varA != 1 | AF(varR == 1))',
        10:'EF(varA == 1 & EG(varR != 1))',
        11:'AG(varA != 1 | EF(varR == 1))',
        12:'EF(varA == 1 & AG(varR != 1))',
        13:'EG(varP1 != 1) | EG(varP2 != 1)',
        14:'EG(varP1 != 1) | EG(varP2 != 1)',
        15:'EF(varP1 == 1) & EF(varP2 == 1)',
        16:'AG(varP1 != 1) | AG(varP2 != 1)',
        17:'AG(AF(varW >= 1))',
        18:'EF(EG(varW < 1))',
        19:'AG(EF(varW >=1))',
        20:'EF(AG(varW < 1))',
        21:'AG(AF(varW == 1))',
        22:'EF(EG(varW != 1))',
        23:'AG(EF(varW == 1))',
        24:'EF(AG(varW != 1))',
        25:'(varC <= 5) | (AF(varR >= 6))',
        26:'(varC >= 6) & EG(varR <= 5)',
        27:'(varC <= 5) | EF(varR >= 6)',
        28:'(varC >= 6) & AG(varR <= 5)',
    }

    print("\n--- Running Pairwise Simulation Tests ---")
    def compute_refinements(EXAMPLES):
        """
        Compute refinement relation: EXAMPLES[i] simulates EXAMPLES[j] ⇨ L(i) ⊆ L(j).
        Returns a dict: refinements[i] = list of js such that i ⊆ j.
        """
        from collections import defaultdict
        from tqdm import tqdm

        refinements = defaultdict(list)

        print("\n--- Computing Refinement Relations ---")

        for i in tqdm(EXAMPLES):
            f1 = EXAMPLES[i]
            a1 = build(f1)

            for j in EXAMPLES:
                if i == j:
                    continue
                f2 = EXAMPLES[j]
                a2 = build(f2)

                ok, _ = a1.is_simulated_by(a2)
                if ok:
                    refinements[i].append(j)

        print("\n========== REFINEMENT SUMMARY ==========")
        for i in sorted(refinements):
            refined_ids = refinements[i]
            refined_strs = ", ".join(f"{j}: {EXAMPLES[j]}" for j in refined_ids)
            print(f"EXAMPLES[{i}] ⊆ {{ {refined_strs} }}")

        return refinements
    return compute_refinements(EXAMPLES)


# ==============================================================================
#  SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Execute the main, comprehensive test suite.
    run_tests()
    
    # Run the large-scale pairwise refinement test.
    runtestT2()
    #debug_negative_delayed("EF (p|q)", "EF p")

