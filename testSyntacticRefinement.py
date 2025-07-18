
import re
from typing import Set, Dict, Any
from z3 import Solver, Real, And, Not, sat,unsat,  Bool, Or
from Parser import PARSER_PROP
TOKEN_REGEX = re.compile(r'\s*(=>|==|!=|<=|>=|[()&|!<>]|\w+|\d+\.\d+|\d+)\s*')
Symbol = str
def parse_prop(prop: str):
    # Parse atomic proposition like "x > 5" into (var, op, val)
    pattern = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(==|!=|<=|>=|<|>)\s*(-?\d+\.?\d*)")
    match = pattern.match(prop)
    if not match:
        return (prop, None, None)  # Boolean variable with no comparison
    var, op, val = match.groups()
    return (var, op, float(val))

def check_entailment(a_props: set, b_prop: str) -> bool:
    # Check if conjunction of a_props implies b_prop
    b_var, b_op, b_val = parse_prop(b_prop)
    
    # If b_prop is simple boolean variable (no comparison)
    if b_op is None:
        # Then A must contain this boolean variable for entailment
        return b_prop in a_props
    
    # Collect all A constraints on the same variable
    relevant_a = [p for p in a_props if parse_prop(p)[0] == b_var and parse_prop(p)[1] is not None]
    if not relevant_a:
        return False  # No info on this variable in A

    # Check if all constraints in A imply b_prop
    # For simplicity, combine intervals and equalities manually:
    # We'll check if all a_props constraints on var imply b_prop

    # Let's build simple bounds from a_props
    lower_bound = -float('inf')
    upper_bound = float('inf')
    equals = None
    not_equals = set()

    for a_p in relevant_a:
        _, a_op, a_val = parse_prop(a_p)
        if a_op == '==':
            if equals is not None and equals != a_val:
                # Contradiction in A: no entailment
                return False
            equals = a_val
        elif a_op == '!=':
            not_equals.add(a_val)
        elif a_op == '<':
            if a_val < upper_bound:
                upper_bound = a_val - 1e-9  # slightly less to be strict
        elif a_op == '<=':
            if a_val < upper_bound:
                upper_bound = a_val
        elif a_op == '>':
            if a_val > lower_bound:
                lower_bound = a_val + 1e-9
        elif a_op == '>=':
            if a_val > lower_bound:
                lower_bound = a_val

    # Now check if these bounds/equalities imply b_prop
    if equals is not None:
        # If A says var == equals, then check if equals satisfies b_prop
        if b_op == '==': return equals == b_val
        if b_op == '!=': return equals != b_val
        if b_op == '<': return equals < b_val
        if b_op == '<=': return equals <= b_val
        if b_op == '>': return equals > b_val
        if b_op == '>=': return equals >= b_val

    # If no equals constraint, check bounds
    if b_op == '==':
        # To imply var == b_val, bounds must be exactly b_val and b_val not in not_equals
        return lower_bound <= b_val <= upper_bound and b_val not in not_equals

    if b_op == '!=':
        # To imply var != b_val, check if b_val is outside bounds or excluded by not_equals
        if b_val < lower_bound or b_val > upper_bound:
            return True
        if b_val in not_equals:
            return True
        # Otherwise cannot guarantee
        return False

    if b_op == '<':
        # All A's values must be < b_val
        return upper_bound < b_val

    if b_op == '<=':
        return upper_bound <= b_val

    if b_op == '>':
        return lower_bound > b_val

    if b_op == '>=':
        return lower_bound >= b_val

    return False

def check_semantic_subset_manual(set_A: set, set_B: set) -> bool:
    if not set_B:
        return True

    for b_prop in set_B:
        if not check_entailment(set_A, b_prop):
            return False
    return True



def build_z3_expr(expr: str, vars: dict):
    # Replace `!` with `Not`, careful with token boundaries
    expr = re.sub(r'!\s*(\w+)', r'Not(\1)', expr)

    # Replace infix & and | with Python-friendly equivalents
    expr = expr.replace("&", ",").replace("|", ",")

    # Handle And/Or function application
    expr = re.sub(r'(\w+)\s*,\s*(\w+)', r'And(\1, \2)', expr)  # Simplified for 2 terms

    # Ensure all variables are defined in Z3
    tokens = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expr))
    for var in tokens:
        if var not in vars:
            vars[var] = Bool(var)

    # Evaluate the transformed expression
    return eval(expr, {"And": And, "Or": Or, "Not": Not, **vars})

Symbol = str
def prop_to_z3(prop: Symbol):
    """Converts a string proposition into a Z3 expression."""
    #if prop in _z3_prop_cache:
    #    return _z3_prop_cache[prop]

    pattern = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(==|!=|<=|>=|<|>|&|)\s*(-?\d+\.?\d*)")
    match = pattern.match(prop)

    if not match:
        #if prop not in _z3_var_cache:
        #    _z3_var_cache[prop] = Bool(prop)
        return Bool(prop)
    var_name, op, val_str = match.groups()
    name = var_name +op+ val_str
    val = float(val_str)

    # Get or create the Z3 variable for this name
    #if name not in _z3_var_cache:
        #_z3_var_cache[name] = Real(var_name)
    z3_var = Real(var_name)
    
    # Create the Z3 expression
    if op == '==': expr = (z3_var == val)
    elif op == '!=': expr = (z3_var != val)
    elif op == '<':  expr = (z3_var < val)
    elif op == '<=': expr = (z3_var <= val)
    elif op == '>':  expr = (z3_var > val)
    elif op == '>=': expr = (z3_var >= val)
    else: raise ValueError(f"Unknown operator: {op}")

    #_z3_prop_cache[prop] = expr
    return expr


def tokenize(expr: str):
    tokens = TOKEN_REGEX.findall(expr)
    return [t for t in tokens if t.strip()]



# High-level interface
def parse_expression_to_z3(expr_str: str):
    tokens = tokenize(expr_str)
    z3_expr = PARSER_PROP.parse(tokens)
    return z3_expr#, parser.variables




def check_semantic_subset_z3(set_A: Set[Symbol], set_B: Set[Symbol]) -> bool:
    """
    Checks if the conjunction of propositions in set_A semantically implies
    the conjunction of propositions in set_B using a Z3 SMT solver.
    
    This correctly handles complex interactions between propositions.
    """
    if not set_B:
        return True # Anything implies True

    # Create Z3 expressions for all propositions in A and B
    z3_props_A = [parse_expression_to_z3(p) for p in set_A]
    z3_props_B = [parse_expression_to_z3(p) for p in set_B]
    # The formula we want to check for unsatisfiability is:
    # (prop_A1 AND prop_A2 AND ...) AND NOT (prop_B1 AND prop_B2 AND ...)
    # which is equivalent to:
    # (prop_A1 AND prop_A2 AND ...) AND (NOT prop_B1 OR NOT prop_B2 OR ...)
    s = Solver()
    
    # Add all premises from set_A
    if z3_props_A:
        s.add(And(z3_props_A))

    # Add the negation of the conclusion (set_B)
    s.add(Not(And(z3_props_B)))
    # If (A AND NOT B) is unsatisfiable, then A => B is a tautology.
    # The check() method returns unsat, sat, or unknown.
    return s.check() == unsat




if __name__ == "__main__":
    expr = "!p | q"
    z3_expr = parse_expression_to_z3(expr)

    print("Z3 expression:", z3_expr)

    result = check_semantic_subset_z3(set("q"), set("p | !p "))
    print("Satisfiability of (expression):", result)