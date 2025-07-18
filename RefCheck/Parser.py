# ==============================================================================
#  IMPORTS
# ==============================================================================
from lark import Lark, Visitor, Tree, Token
from lark import Transformer, v_args
from z3 import Solver, Real, And, Not, sat, unsat, Bool, Or

# ==============================================================================
#  TCTL GRAMMAR DEFINITION
# ==============================================================================

# This Lark grammar defines a Time-bounded Computation Tree Logic (TCTL).
# It supports standard CTL operators, weak until, time bounds, and arithmetic comparisons.

CUSTOMGRAMMAR = """
    start: expr

    ?expr: or_

    ?or_: and_
        | or_ "|" and_     -> or_

    ?and_: unary
        | and_ "&" unary   -> and_

    ?unary: imply_
          | "!" unary      -> neg

    ?imply_: primary "->" primary -> imply_
            | primary

    ?primary: comparison
        | atom
        | "EF" time_expr unary          -> ef
        | "AF" time_expr unary          -> af
        | "EG" time_expr unary          -> eg
        | "AG" time_expr unary          -> ag
        | "E(" expr "U" expr ")" -> eu
        | "A(" expr "U" expr ")" -> au
        | "A(" expr "W" expr ")" -> aw
        | "E(" expr "W" expr ")" -> ew
        | "A" expr "U" expr             -> au_no_time
        | "A" expr "W" expr             -> aw_no_time
        | "E" expr "W" expr             -> ew_no_time
        | "(" expr ")"

    comparison: CNAME COMPARATOR (CNAME|SIGNED_NUMBER)
    atom: CNAME

    COMPARATOR: "==" | "!=" | "<=" | ">=" | "<" | ">"
    ?time_expr: time_constraint?
    time_constraint: /\[\d+,\d+\]/ 

    %import common.WS
    %import common.CNAME
    %import common.SIGNED_NUMBER
    %ignore WS
"""



# ==============================================================================
#  LARK VISITORS (For Tree Traversal and Collection)
# ==============================================================================

class AtomCollector(Visitor):
    """
    A Lark Visitor that traverses a parsed formula tree to collect all unique
    atomic propositions and comparisons, treating them as indivisible units.
    """
    def __init__(self):
        self.atoms = set()
    
    def atom(self, tree):
        # tree.children[0] is a Token holding the atom string
        self.atoms.add(tree.children[0].value)
    def neg(self, tree):
        # tree.children[0] is a Token holding the atom string
        self.atoms.add(tree.children[0].value)
    def comparison(self, tree):
        # tree.children[0] is a Token holding the atom string
        self.atoms.add(tree.children[0].value)




class SubformulaCollector(Visitor):
    """
    A Lark Visitor to collect all top-level temporal subformulas from a parsed tree.
    It reconstructs the string representation of each temporal formula it encounters.
    """
    def __init__(self):
        self.subformulas = set()

    def af(self, tree):
        self.subformulas.add(self._tree_to_formula(tree))
    def eg(self, tree):
        self.subformulas.add(self._tree_to_formula(tree))
    def ef(self, tree):
        self.subformulas.add(self._tree_to_formula(tree))
    def ag(self, tree):
        self.subformulas.add(self._tree_to_formula(tree))

    def _tree_to_formula(self,t: Tree) -> str:
        if isinstance(t, Token):
            return t.value
        if t.data == 'atom':
            return t.children[0].value
        elif t.data == 'neg':
            return f"!{self._tree_to_formula(t.children[0])}"
        elif t.data == 'and_':
            return f"({self._tree_to_formula(t.children[0])} & {self._tree_to_formula(t.children[1])})"
        elif t.data == 'or_':
            return f"({self._tree_to_formula(t.children[0])} | {self._tree_to_formula(t.children[1])})"
        elif t.data == 'imply_':
            return f"({self._tree_to_formula(t.children[0])} -> {self._tree_to_formula(t.children[1])})"
        elif t.data == 'ef':
            return f"EF {self._tree_to_formula(t.children[0])} {self._tree_to_formula(t.children[1])}"
        elif t.data == 'eg':
            return f"EG {self._tree_to_formula(t.children[0])} {self._tree_to_formula(t.children[1])}"
        elif t.data == 'af':
            return f"AF {self._tree_to_formula(t.children[0])} {self._tree_to_formula(t.children[1])}"
        elif t.data == 'ag':
            return f"AG {self._tree_to_formula(t.children[0])} {self._tree_to_formula(t.children[1])}"
        elif t.data == 'eu':
            return f"E[{self._tree_to_formula(t.children[0])} U {self._tree_to_formula(t.children[1])}]"
        elif t.data == 'expr':
            return self._tree_to_formula(t.children[0])
        else:
            if len(t.children) == 1:
                return self._tree_to_formula(t.children[0])
            else:
                return " ".join(self._tree_to_formula(c) if isinstance(c, Tree) else str(c) for c in t.children)
# ==============================================================================
#  LARK TRANSFORMERS (For Tree Manipulation)
# ==============================================================================
class NNFTransformer:
    """
    Transforms a parsed TCTL formula tree into its Negation Normal Form (NNF).
    In NNF, negations are pushed inwards so they only apply directly to atomic
    propositions, using logical dualities (e.g., De Morgan's laws).
    """
    def to_nnf(self, tree: Tree, negated=False) -> Tree:
        """
        Recursively converts the tree to NNF.

        Args:
            tree: The Lark Tree to transform.
            negated: A boolean flag indicating if the current sub-formula is under a negation.
        """
        if isinstance(tree, Token):
            return Tree("neg", [tree]) if negated else Tree("atom", [Token("CNAME",tree)])

        if tree.data == "neg":
            return self.to_nnf(tree.children[0], not negated)

        elif tree.data == "and_":
            left, right = tree.children
            if negated:
                # ¬(φ ∧ ψ) → ¬φ ∨ ¬ψ
                return Tree("or_", [self.to_nnf(left, True), self.to_nnf(right, True)])
            else:
                return Tree("and_", [self.to_nnf(left), self.to_nnf(right)])

        elif tree.data == "or_":
            left, right = tree.children
            if negated:
                # ¬(φ ∨ ψ) → ¬φ ∧ ¬ψ
                return Tree("and_", [self.to_nnf(left, True), self.to_nnf(right, True)])
            else:
                return Tree("or_", [self.to_nnf(left), self.to_nnf(right)])

        elif tree.data == "imply_":
            # (φ -> ψ) ≡ (¬φ ∨ ψ), so handle it first
            left, right = tree.children
            if left.data not in {"ef", "ag", "af", "ag"}:
                return self.to_nnf(Tree("or_", [Tree("neg", [left]), right]), negated)
            else:
                temp = self.to_nnf(Tree("or_", [Tree("neg", [left.children[1]]), right]), negated)
                value = ""
                if len(left.children[0].children)>0:
                    value = left.children[0].children[0]
                return Tree(left.data, [left.children[0],temp])
        elif tree.data in {"ef", "af", "eg", "ag"}:
            op = tree.data
            time_expr = tree.children[0]
            phi = tree.children[1]

            # Modal duals
            if negated:
                dual_map = {"ef": "ag", "af": "eg", "eg": "af", "ag": "ef"}
                new_op = dual_map[op]
                return Tree(new_op, [time_expr, self.to_nnf(phi, True)])
            else:
                return Tree(op, [time_expr, self.to_nnf(phi)])

        elif tree.data == "eu":
            e1,e2 = tree.children
            # Optional: handle ¬E[φ U ψ] with dual R operator or approximate
            if negated:
                return Tree("au", [ self.to_nnf(e2, True), self.to_nnf(e1, True)])
                #raise NotImplementedError("NNF for ¬E[φ U ψ] is not yet implemented.")
            else:

                return Tree("eu", [self.to_nnf(e1),  self.to_nnf(e2)])

        elif tree.data == "atom":
            return Tree("neg", [tree.children[0]]) if negated else tree

        elif tree.data == "start":
            return Tree("start", [self.to_nnf(tree.children[0])])

        else:
            # Default: recurse into children
            return Tree(tree.data, [self.to_nnf(c, negated) if isinstance(c, Tree) else c for c in tree.children])

def propositional_to_str(tree):
    """Helper to convert a purely propositional subtree back into a single string."""
    if tree.data == 'atom':
        return tree.children[0].value
    elif tree.data == 'comparison':
        return ''.join(str(child) for child in tree.children)
    elif tree.data == 'neg':
        return f"!({propositional_to_str(tree.children[0])})"
    elif tree.data == 'and_':
        return f"({propositional_to_str(tree.children[0])} & {propositional_to_str(tree.children[1])})"
    elif tree.data == 'or_':
        return f"({propositional_to_str(tree.children[0])} | {propositional_to_str(tree.children[1])})"
    elif tree.data == 'imply_':
        return f"(!{propositional_to_str(tree.children[0])} | {propositional_to_str(tree.children[1])})"
    return str(tree)

def is_propositional(tree):
    if isinstance(tree, Tree):
        if tree.data in ('atom', 'comparison'):
            return True
        elif tree.data in ('and_', 'or_', 'neg', 'imply_'):
            return all(is_propositional(child) for child in tree.children)
    return False

class Simplifier(Transformer):
    """
    A Lark Transformer that simplifies TCTL formulas by:
    1. Collapsing purely propositional sub-formulas into single atomic propositions.
    2. Distributing temporal operators over logical connectives where valid
       (e.g., EF(a|b) <=> EF(a)|EF(b)).
    """
    def atom(self, items):
        return Tree('atom', items)

    def comparison(self, items):
        return Tree('comparison', items)

    def and_(self, items):
        left, right = items
        if is_propositional(left) and is_propositional(right):
            return Tree('atom', [Token("CNAME",propositional_to_str(Tree('and_', [left, right])))])
        return Tree('and_', [left, right])

    def or_(self, items):
        left, right = items
        if is_propositional(left) and is_propositional(right):
            return Tree('atom', [Token("CNAME",propositional_to_str(Tree('or_', [left, right])))])
        return Tree('or_', [left, right])

    def neg(self, items):
        child = items[0]
        if is_propositional(child):
            return Tree('atom', [Token("CNAME",propositional_to_str(Tree('neg', [child])))])
        return Tree('neg', [child])

    def imply_(self, items):
        left, right = items
        if is_propositional(left) and is_propositional(right):
            return Tree('atom', [Token("CNAME",propositional_to_str(Tree('imply_', [left, right])))])
        return Tree('imply_', [left, right])

    def ef(self, items):
        time_expr, formula = items
        if isinstance(formula, Tree) and formula.data == 'or_':
            left, right = formula.children
            return Tree('or_', [
                Tree('ef', [time_expr, left]),
                Tree('ef', [time_expr, right])
            ])
        return Tree('ef', [time_expr, formula])

    def af(self, items):
        time_expr, formula = items
        if isinstance(formula, Tree) and formula.data == 'and_':
            left, right = formula.children
            return Tree('and_', [
                Tree('af', [time_expr, left]),
                Tree('af', [time_expr, right])
            ])
        return Tree('af', [time_expr, formula])

    def eg(self, items):
        time_expr, formula = items
        if isinstance(formula, Tree) and formula.data == 'and_':
            left, right = formula.children
            return Tree('and_', [
                Tree('eg', [time_expr, left]),
                Tree('eg', [time_expr, right])
            ])
        return Tree('eg', [time_expr, formula])

    def ag(self, items):
        time_expr, formula = items
        if isinstance(formula, Tree) and formula.data == 'or_':
            left, right = formula.children
            return Tree('or_', [
                Tree('ag', [time_expr, left]),
                Tree('ag', [time_expr, right])
            ])
        return Tree('ag', [time_expr, formula])

    def eu(self, items):
        # No general valid distribution for E[φ U ψ] — return as-is
        return Tree('eu', items)
    def start(self, items): return items[0]

# ==============================================================================
#  MANUAL RECURSIVE DESCENT PARSER (For Propositional Logic)
# ==============================================================================
# Parser (recursive descent)
class ParserS:
    """
    A manual recursive descent parser for a simple propositional logic with comparisons.
    Unlike the Lark parser, this one directly builds a Z3 SMT formula.
    """
    def __init__(self):
        pass
        
          # Store Z3 variables
    
    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self):
        tok = self.peek()
        self.pos += 1
        return tok

    def parse(self, tokens = None):
        if tokens is not None:
            self.tokens = tokens
            self.pos = 0
            self.variables = {}
        expr =  self.parse_or()
        if tokens is not None:
            self.pos = 0
            self.variables = {}
        return expr

    def parse_or(self):
        left = self.parse_and()
        while self.peek() == '|':
            self.consume()
            right = self.parse_and()
            left = Or(left, right)
        return left

    def parse_and(self):
        left = self.parse_not()
        while self.peek() == '&':
            self.consume()
            right = self.parse_not()
            left = And(left, right)
        return left

    def parse_not(self):
        if self.peek() == '!':
            self.consume()
            return Not(self.parse_atom())
        return self.parse_atom()

    def parse_atom(self):
        tok = self.peek()
        if tok == '(':
            self.consume()
            expr = self.parse()
            assert self.consume() == ')'
            return expr
        else:
            return self.parse_comparison()

    def parse_comparison(self):
        left = self.consume()
        op = self.peek()

        if op in ['==', '!=', '<', '>', '<=', '>=']:
            self.consume()
            right = self.consume()
            try:
                val = float(right)
                var = self.get_or_create_real(left)
                return self.make_comparison(var, op, val)
            except ValueError:
                # right is a variable
                var1 = self.get_or_create_real(left)
                var2 = self.get_or_create_real(right)
                return self.make_comparison(var1, op, var2)
        else:
            # Boolean variable
            return self.get_or_create_bool(left)

    def make_comparison(self, var, op, val):
        if op == '==': return var == val
        if op == '!=': return var != val
        if op == '<': return var < val
        if op == '>': return var > val
        if op == '<=': return var <= val
        if op == '>=': return var >= val
        raise ValueError(f"Unknown operator: {op}")

    def get_or_create_real(self, name):
        if name not in self.variables:
            self.variables[name] = Real(name)
        return self.variables[name]

    def get_or_create_bool(self, name):
        if name not in self.variables:
            self.variables[name] = Bool(name)
        return self.variables[name]


# ==============================================================================
#  TRANSFORMERS FOR SPECIFIC OUTPUT FORMATS
# ==============================================================================
# This Transformer class walks the parse tree and converts it to NuSMV syntax.
@v_args(inline=True) # Makes the transformer methods receive children directly
class NuSMVTransformer(Transformer):
    """
    A Lark Transformer that converts a parsed TCTL formula tree into a string
    that is syntactically correct for the NuSMV model checker.
    """

    def __init__(self):
        super().__init__()
        self.time_bound_warning_issued = False

    def _check_time(self, time_expr):
        if time_expr is not None and not self.time_bound_warning_issued:
            print("Warning: Time-bounded CTL properties are not supported by standard NuSMV. The time bounds will be ignored.")
            self.time_bound_warning_issued = True

    # --- Pass-through rules ---
    def start(self, expr): return expr
    def expr(self, or_): return or_
    def unary(self, primary): return primary
    def primary(self, item): return f"({item})" if ' ' in str(item) else str(item)
    def atom(self, name): return str(name)

    # --- Logical operators ---
    def or_(self, left, right): return f"({left} | {right})"
    def and_(self, left, right): return f"({left} & {right})"
    def neg(self, unary): return f"!({unary})"
    def imply_(self, left, right): return f"({left} -> {right})"

    # --- Temporal operators ---
    def ef(self, time_expr, unary):
        self._check_time(time_expr)
        return f"EF {unary}"

    def af(self, time_expr, unary):
        self._check_time(time_expr)
        return f"AF {unary}"

    def eg(self, time_expr, unary):
        self._check_time(time_expr)
        return f"EG {unary}"

    def ag(self, time_expr, unary):
        self._check_time(time_expr)
        return f"AG {unary}"
        
    def eu(self, left, right):
        #self._check_time(time_expr)
        return f"E [ {left} U {right} ]"

    def au(self, left, right):
        #self._check_time(time_expr)
        return f"A [ {left} U {right} ]"
    
    # Handle rules without explicit time_expr for completeness
    def au_no_time(self, left, right): return f"A [ {left} U {right} ]"
    def aw_no_time(self, left, right): return self.aw(left, None, right)
    def ew_no_time(self, left, right): return self.ew(left, None, right)

    # --- Weak Until Translation ---
    def ew(self, left, right):
        # E[p W q]  <=>  E[p U q] | EG p
        eu_part = f"E [ {left} U {right} ]"
        eg_part = f"EG {left}"
        return f"({eu_part} | {eg_part})"

    def aw(self, left, right):
        # A[p W q]  <=>  A[p U q] | AG p
        au_part = f"A [ {left} U {right} ]"
        ag_part = f"AG {left}"
        return f"({au_part} | {ag_part})"

    # Catch-all for time constraints (they return their string value)
    def time_constraint(self, tc): return str(tc)
    # None is returned if an optional rule doesn't match
    def time_expr(self, tc=None): return tc











class CTLSATtransformer(Transformer):
    """
    A Lark Transformer that converts the parsed tree into a format presumably
    compatible with the specific CTL-SAT solver (https://github.com/nicolaprezza/CTLSAT), using symbols like 'v', '^', '~'.
    """
    def start(self, args):
        return args[0]

    def or_(self, args):
        return f"({args[0]}v{args[1]})"

    def and_(self, args):
        return f"({args[0]}^{args[1]})"

    def neg(self, args):
        return f"~({args[0]})"

    def imply_(self, args):
        return f"({args[0]}->{args[1]})"

    def ef(self, args):
        return f"EF({args[1]})"

    def af(self, args):
        return f"AF({args[1]})"

    def eg(self, args):
        return f"EG({args[1]})"

    def ag(self, args):
        return f"AG({args[1]})"

    def eu(self, args):
        return f"E({args[0]}U{args[1]})"

    def au(self, args):
        return f"A({args[0]}U{args[1]})"

    def aw(self, args):
        return f"A({args[0]}~({args[1]}))"

    def ew(self, args):
        return f"E({args[0]}~({args[1]}))"

    def au_no_time(self, args):
        return f"A({args[0]}U{args[1]})"

    def aw_no_time(self, args):
        return f"A({args[0]}~({args[1]}))"

    def ew_no_time(self, args):
        return f"E({args[0]}~({args[1]}))"

    def comparison(self, args):
        # This will collapse all comparisons into a dummy 'T'
        return "T"

    def atom(self, args):
        return str(args[0])

    def time_expr(self, args):
        return ""  # Time is ignored in target grammar

    def expr(self, args):
        return args[0]

    def primary(self, args):
        return args[0]

    def imply(self, args):
        return f"({args[0]}->{args[1]})"

    def __default__(self, data, children, meta):
        if len(children) == 1:
            return children[0]
        return f"({' '.join(map(str, children))})"




# ==============================================================================
#  GLOBAL INSTANCES
# ==============================================================================

# Create global instances of the parsers and transformers for easy access


PARSER_PROP = ParserS()

PARSER = Lark(CUSTOMGRAMMAR)

NNF = NNFTransformer()

TRANSF = Simplifier()

NUSMV = NuSMVTransformer()


CTLSATTRANS = CTLSATtransformer()


# ==============================================================================
#  EXAMPLE USAGE
# ==============================================================================
if __name__ == "__main__":
    # Example demonstrating the parsing and transformation pipeline.
    tree = PARSER.parse("p & q ")
    transformed = TRANSF.transform(tree)
    print("Original Parse Tree:")
    print(tree.pretty())
    print("\nTransformed Tree (after collapsing propositional part):")
    print(transformed.pretty())
