# ==============================================================================
#  IMPORTS AND TYPE DEFINITIONS
# ==============================================================================

from typing import Set, Dict, Union, Tuple, List, Any
from dataclasses import dataclass
from collections import deque
from . import Parser  # Local import of the formula parser module.
import re
from z3 import Solver, Real, And, Not, sat, unsat, Bool, BoolVal

# --- Type Aliases for Clarity ---
# Defines a state within the automaton, typically represented as a string.
State = str
# Defines a symbol from the automaton's alphabet, representing an atomic proposition.
Symbol = str
# Defines a direction for a transition in a tree structure (e.g., 'left', 'right').
Direction = str


# ==============================================================================
#  PROPOSITIONAL LOGIC PARSER AND Z3 INTERFACE
# ==============================================================================

# Regular expression to tokenize expressions containing logical and arithmetic operators.
# Captures operators (=>, ==, !=, <=, >=), symbols (&, |, !, <, >), parentheses,
# words (variable names), and numbers (including decimals).
TOKEN_REGEX = re.compile(r'\s*(=>|==|!=|<=|>=|[()&|!<>]|\w+|\d+\.\d+|\d+)\s*')

def tokenize(expr: str) -> List[str]:
    """
    Splits a given expression string into a list of tokens based on TOKEN_REGEX.

    Args:
        expr: The string expression to tokenize.

    Returns:
        A list of string tokens, with surrounding whitespace removed.
    """
    tokens = TOKEN_REGEX.findall(expr)
    return [t for t in tokens if t.strip()]


def parse_expression_to_z3(expr_str: str) -> Any:
    """
    High-level interface to parse a string expression and convert it into a Z3 object.
    Handles simple boolean constants directly and uses the main parser for complex expressions.

    Args:
        expr_str: The expression string (e.g., "x > 5 & y < 10", "true").

    Returns:
        A Z3 expression object representing the input string.
    """
    # Handle boolean literals as special cases for efficiency.
    if expr_str == "true":
        return BoolVal(True)
    if expr_str == "false":
        return BoolVal(False)
    
    # For all other expressions, tokenize and parse.
    tokens = tokenize(expr_str)
    z3_expr = Parser.PARSER_PROP.parse(tokens)
    return z3_expr


# ==============================================================================
#  TRANSITION EXPRESSION DATA CLASSES (AST NODES)
# ==============================================================================

class TransitionExpr:
    """
    Base class for all transition expression types. This forms the Abstract
    Syntax Tree (AST) for the logic governing automaton transitions.
    """
    pass

@dataclass(frozen=True, eq=True)
class TrueExpr(TransitionExpr):
    """Represents the boolean constant 'true'. A transition that is always enabled."""
    symbol= "true"

@dataclass(frozen=True, eq=True)
class FalseExpr(TransitionExpr):
    """Represents the boolean constant 'false'. A transition that is never enabled."""
    symbol= "false"

@dataclass(frozen=True, eq=True)
class AtomicExpr(TransitionExpr):
    """Represents a simple atomic proposition (e.g., 'p')."""
    symbol: Symbol

@dataclass(frozen=True, eq=True)
class AndExpr(TransitionExpr):
    """Represents a logical conjunction (AND) of two sub-expressions."""
    left: TransitionExpr
    right: TransitionExpr

@dataclass(frozen=True, eq=True)
class OrExpr(TransitionExpr):
    """Represents a logical disjunction (OR) of two sub-expressions."""
    left: TransitionExpr
    right: TransitionExpr

@dataclass(frozen=True, eq=True)
class NextExpr(TransitionExpr):
    """Represents a temporal move to a successor state in a specific direction."""
    direction: Direction
    state: State

@dataclass(frozen=True, eq=True)
class StateExpr(TransitionExpr):
    """Represents a reference to another state's transition logic."""
    state: State

@dataclass(frozen=True, eq=True)
class NotExpr(TransitionExpr):
    """Represents a logical negation (NOT) of a sub-expression."""
    expr: TransitionExpr

# --- CTL Operator Expressions ---
# These dataclasses represent the structure of common CTL temporal operators.

@dataclass(frozen=True, eq=True)
class EUExpr(TransitionExpr):
    """Represents the CTL operator E[φ U ψ] (Existential Until)."""
    left: TransitionExpr
    right: TransitionExpr

@dataclass(frozen=True, eq=True)
class EGExpr(TransitionExpr):
    """Represents the CTL operator EG(φ) (Existential Globally)."""
    expr: TransitionExpr

@dataclass(frozen=True, eq=True)
class AGExpr(TransitionExpr):
    """Represents the CTL operator AG(φ) (Universal Globally)."""
    expr: TransitionExpr

@dataclass(frozen=True, eq=True)
class ComparisonExpr(TransitionExpr):
    """
    Represents an arithmetic comparison (e.g., 'x > 5'). This allows the
    automaton to handle propositions over real or integer variables.
    """
    variable: Symbol
    operator: str
    value: Union[int, float]

    def __str__(self):
        """Provides a clean string representation of the comparison."""
        return f"{self.variable} {self.operator} {self.value}"


# A "Move" represents a single clause in the Disjunctive Normal Form (DNF) of a transition.
# It is a tuple containing:
#   1. A set of atomic propositions that must hold at the current node.
#   2. A set of required successor moves, each defined by a (direction, state) tuple.
Move = Tuple[Set[Symbol], Set[Tuple[Direction, State]]]


# ==============================================================================
#  SEMANTIC SUBSET CHECKING (ENTAILMENT)
# ==============================================================================

# --- Caches for Z3 variables and propositions (for potential optimization) ---
_z3_prop_cache: Dict[Symbol, Any] = {}
_z3_var_cache: Dict[Symbol, Any] = {}

def parse_prop(prop: str) -> Tuple[Symbol, Union[str, None], Union[float, None]]:
    """
    Parses a string proposition like "x > 5" into its components.

    Args:
        prop: The proposition string.

    Returns:
        A tuple (variable, operator, value). If the proposition is a simple
        boolean atom (e.g., "door_open"), returns (prop, None, None).
    """
    # Regex to find a variable, a comparison operator, and a numeric value.
    pattern = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(==|!=|<=|>=|<|>)\s*(-?\d+\.?\d*)")
    match = pattern.match(prop)
    if not match:
        return (prop, None, None)  # It's a simple boolean variable.
    var, op, val = match.groups()
    return (var, op, float(val))

def check_entailment(a_props: set, b_prop: str) -> bool:
    """
    Manually checks if a set of propositions `a_props` logically entails `b_prop`.
    This implementation handles simple numeric interval logic but is less robust
    than an SMT solver for complex interactions.

    Args:
        a_props: A set of premise proposition strings.
        b_prop: The conclusion proposition string.

    Returns:
        True if the entailment holds, False otherwise.
    """
    b_var, b_op, b_val = parse_prop(b_prop)

    # If b_prop is a simple boolean variable, entailment holds only if it's in a_props.
    if b_op is None:
        return b_prop in a_props
    
    # Collect all constraints from `a_props` that relate to the same variable as `b_prop`.
    relevant_a = [p for p in a_props if parse_prop(p)[0] == b_var and parse_prop(p)[1] is not None]
    if not relevant_a:
        return False  # No information about this variable in the premises.

    # Build a simple model of the constraints from the premises (lower/upper bounds, equalities).
    lower_bound = -float('inf')
    upper_bound = float('inf')
    equals = None
    not_equals = set()

    for a_p in relevant_a:
        _, a_op, a_val = parse_prop(a_p)
        if a_op == '==':
            if equals is not None and equals != a_val:
                return False # Contradiction in premises (e.g., x==5 and x==6).
            equals = a_val
        elif a_op == '!=':
            not_equals.add(a_val)
        elif a_op == '<':
            upper_bound = min(upper_bound, a_val - 1e-9) # Strict inequality.
        elif a_op == '<=':
            upper_bound = min(upper_bound, a_val)
        elif a_op == '>':
            lower_bound = max(lower_bound, a_val + 1e-9) # Strict inequality.
        elif a_op == '>=':
            lower_bound = max(lower_bound, a_val)

    # Check if the derived model (bounds/equalities) satisfies the conclusion `b_prop`.
    if equals is not None:
        # If premises imply `var == value`, check if that value satisfies the conclusion.
        if b_op == '==': return equals == b_val
        if b_op == '!=': return equals != b_val
        if b_op == '<':  return equals < b_val
        if b_op == '<=': return equals <= b_val
        if b_op == '>':  return equals > b_val
        if b_op == '>=': return equals >= b_val

    # If no equality constraint, check using the derived bounds.
    if b_op == '==':
        return lower_bound <= b_val <= upper_bound and b_val not in not_equals
    if b_op == '!=':
        return b_val < lower_bound or b_val > upper_bound or b_val in not_equals
    if b_op == '<':
        return upper_bound < b_val
    if b_op == '<=':
        return upper_bound <= b_val
    if b_op == '>':
        return lower_bound > b_val
    if b_op == '>=':
        return lower_bound >= b_val

    return False

def check_semantic_subset_manual(set_A: set, set_B: set) -> bool:
    """
    Checks if set_A semantically implies set_B by checking entailment for each
    proposition in set_B.

    Args:
        set_A: A set of premise propositions.
        set_B: A set of conclusion propositions.

    Returns:
        True if the conjunction of set_A implies the conjunction of set_B.
    """
    if not set_B:
        return True # An empty conclusion (True) is implied by anything.

    for b_prop in set_B:
        if not check_entailment(set_A, b_prop):
            return False
    return True

# --- Z3-based Implementation ---

def prop_to_z3(prop: Symbol) -> Any:
    """
    Converts a single string proposition into a corresponding Z3 expression.
    Handles both simple boolean atoms and arithmetic comparisons.
    Note: Caching is commented out but shows a potential optimization path.
    """
    # Regex to parse `variable operator value` structure.
    pattern = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(==|!=|<=|>=|<|>|&|\|)\s*(-?\d+\.?\d*)")
    match = pattern.match(prop)

    if not match:
        # If no match, treat as a simple uninterpreted boolean variable.
        return Bool(prop)
    
    var_name, op, val_str = match.groups()
    val = float(val_str)

    # Create a Z3 real variable for the parsed variable name.
    z3_var = Real(var_name)
    
    # Create the corresponding Z3 comparison expression.
    if op == '==':   expr = (z3_var == val)
    elif op == '!=': expr = (z3_var != val)
    elif op == '<':  expr = (z3_var < val)
    elif op == '<=': expr = (z3_var <= val)
    elif op == '>':  expr = (z3_var > val)
    elif op == '>=': expr = (z3_var >= val)
    else: raise ValueError(f"Unknown operator: {op}")

    return expr

def check_semantic_subset_z3(set_A: Set[Symbol], set_B: Set[Symbol]) -> bool:
    """
    Checks if the conjunction of propositions in set_A semantically implies
    the conjunction of propositions in set_B using the Z3 SMT solver.
    This is equivalent to checking if the formula (A_1 && A_2 && ...) => (B_1 && B_2 && ...)
    is a tautology.
    """
    if not set_B:
        return True # Anything implies True.

    # Convert all proposition strings in both sets to Z3 expressions.
    z3_props_A = [parse_expression_to_z3(p) for p in set_A]
    z3_props_B = [parse_expression_to_z3(p) for p in set_B]
    
    # To check if (A => B) is a tautology, we check if its negation, (A AND NOT B),
    # is unsatisfiable.
    s = Solver()
    
    # Add all premises from set_A to the solver's context.
    if z3_props_A:
        s.add(And(z3_props_A))

    # Add the negation of the conclusion (set_B).
    # NOT (B1 AND B2 AND ...) is equivalent to (NOT B1 OR NOT B2 OR ...)
    s.add(Not(And(z3_props_B)))
    
    # If the solver finds the formula unsatisfiable, the original entailment holds.
    return s.check() == unsat


# ==============================================================================
#  ALTERNATING BÜCHI TREE AUTOMATON (ABTA) CLASS
# ==============================================================================

class ABTA:
    """
    Represents an Alternating Büchi Tree Automaton (ABTA).

    This class provides the structure and methods to convert a CTL-like temporal
    logic formula into an equivalent ABTA. It also includes the core algorithm
    for checking simulation between two automata, which is used to determine
    language inclusion (i.e., formula entailment).

    Attributes:
        states: A set of all state names in the automaton.
        alphabet: A set of all atomic proposition symbols used.
        initial_state: The starting state of the automaton.
        transitions: A dictionary mapping each state to its transition logic (TransitionExpr).
        accepting: A set of accepting (Büchi) states.
        modality: The overall modality ('universal', 'existential', 'propositional')
                  of the formula, which can influence interpretation.
        formula: The original input formula string.
        _dnf_cache: A cache for memoizing the DNF of state transitions.
        _entailment_cache: A cache for memoizing the results of semantic subset checks.
    """
    states: Set[State]
    alphabet: Set[Symbol]
    initial_state: State
    transitions: Dict[State, TransitionExpr]
    accepting: Set[State]
    modality : str
    formula : str
    _dnf_cache : Dict[State, List[Move]]
    _entailment_cache: Dict[Tuple[frozenset, frozenset], bool]

    def __init__(self, formula: str):
        """
        Initializes the ABTA by parsing and converting a formula string.
        """
        # Initialize caches and state tracking.
        self._dnf_cache = {}
        self._entailment_cache = {}
        self.state_counter = 0
        
        # Initialize core automaton components.
        self.states = set()
        self.alphabet = set()
        self.accepting = set()
        self.transitions = {}
        self.atomic_map: Dict[State, Set[Symbol]] = {}
        self.initial_state = ""
        self._get_assignments_memo: Dict[TransitionExpr, List[Set[TransitionExpr]]] = {}
        self.modality = 'propositional'  # Default modality.
        
        # The core conversion process:
        # 1. Parse the formula string into an intermediate tree representation.
        # 2. Convert the formula to Negation Normal Form (NNF).
        # 3. Apply further transformations if necessary.
        # 4. Recursively convert the transformed tree into the automaton structure.
        nnf = Parser.PARSER.parse(formula)
        nnf = Parser.NNFTransformer().to_nnf(nnf)
        nnf = Parser.TRANSF.transform(nnf)
        self.convert(nnf)
        
        self.formula = formula

    def __repr__(self) -> str:
        """
        Provides a detailed, human-readable string representation of the automaton.
        """
        to_return = f"Constructed ABTA:\n Initial state: {self.initial_state}\n States: {self.states} \n" + \
                    f"Accepting states: {self.accepting} \n Transitions:\n"
        
        for state, trans in self.transitions.items():
            to_return += f"  {state} -> {trans}\n"
        return to_return

    def fresh_state(self, prefix="q") -> State:
        """
        Generates a new, unique state name.

        Args:
            prefix: An optional prefix for the state name for readability.

        Returns:
            A unique string identifier for a new state.
        """
        self.state_counter += 1
        return f"{prefix}{self.state_counter}"

    def convert(self, tree: Parser.Tree):
        """
        Top-level method to start the conversion from a parsed tree to the ABTA.
        It determines the formula's modality and initiates the recursive conversion.
        """
        # The parser may wrap the main expression in a 'start' node.
        if tree.data == "start":
            temp = tree.children[0]
        else:
            temp = tree
        
        # Determine the overall modality based on the top-level operator.
        if temp.data in ('ag', 'af', 'au'):
            self.modality = 'universal'
        elif temp.data in ('eg', 'ef', 'eu'):
            self.modality = 'existential'
        else:
            self.modality = 'propositional'

        # Start the recursive conversion and set the initial state.
        initial = self._convert(tree)
        self.initial_state = initial

    def handle_operator(self, data: str, children: list):
        """
        Handles the conversion of temporal logic operators (AG, EG, etc.).
        This method is a dispatcher called by `_convert`.
        """
        if data == "ef":
            # EF φ  ≡  φ ∨ (EX(true) ∧ EY(EF φ)) -> Simplified to φ ∨ E(next) EF φ
            φ = self._convert(children[1])
            state = self.fresh_state("ef")
            self.states.add(state)
            self.transitions[state] = OrExpr(
                StateExpr(φ),
                OrExpr(NextExpr("left", state), NextExpr("right", state)) # E(next)
            )
            self.accepting.add(state) # Accepting because the loop can continue indefinitely.
            return state

        elif data == "eg":
            # EG φ  ≡  φ ∧ E(next) EG φ
            φ = self._convert(children[1])
            state = self.fresh_state("eg")
            self.states.add(state)
            self.transitions[state] = AndExpr(
                StateExpr(φ),
                OrExpr(NextExpr("left", state), NextExpr("right", state)) # E(next)
            )
            self.accepting.add(state) # Accepting because it must hold globally.
            return state

        elif data == "af":
            # AF φ  ≡  φ ∨ A(next) AF φ
            φ = self._convert(children[1])
            state = self.fresh_state("af")
            self.states.add(state)
            self.transitions[state] = OrExpr(
                StateExpr(φ),
                AndExpr(NextExpr("left", state), NextExpr("right", state)) # A(next)
            )
            self.accepting.add(state) # Accepting due to the eventual nature.
            return state

        elif data == "ag":
            # AG φ  ≡  φ ∧ A(next) AG φ
            φ = self._convert(children[1])
            state = self.fresh_state("ag")
            self.states.add(state)
            self.transitions[state] = AndExpr(
                StateExpr(φ),
                AndExpr(NextExpr("left", state), NextExpr("right", state)) # A(next)
            )
            self.accepting.add(state) # Accepting because it must hold globally.
            return state
    
    def handle_basic(self, data: str, children: list):
        """
        Handles the conversion of basic logical operators and atoms.
        This method is a dispatcher called by `_convert`.
        """
        if data == "atom":
            # Handle an atomic proposition.
            sym = children[0].value if isinstance(children[0], Parser.Token) else children[0]
            state = self.fresh_state(f"{sym}")
            self.states.add(state)
            self.atomic_map[state] = {sym}
            self.alphabet.add(sym)
            if sym == "true":
                self.transitions[state] = TrueExpr()
            elif sym == "false":
                self.transitions[state] = FalseExpr()
            else:
                self.transitions[state] = AtomicExpr(sym)
            self.accepting.add(state)
            return state

        elif data == "neg":
            # Handle negation.
            sub = self._convert(children[0])
            state = self.fresh_state("neg")
            self.states.add(state)
            self.transitions[state] = NotExpr(StateExpr(sub))
            # Acceptance depends on the sub-formula, often non-accepting for simple negation.
            return state

        elif data == "and_":
            # Handle conjunction.
            left = self._convert(children[0])
            right = self._convert(children[1])
            state = self.fresh_state("and")
            self.states.add(state)
            self.transitions[state] = AndExpr(StateExpr(left), StateExpr(right))
            return state

        elif data == "or_":
            # Handle disjunction.
            left = self._convert(children[0])
            right = self._convert(children[1])
            state = self.fresh_state("or")
            self.states.add(state)
            self.transitions[state] = OrExpr(StateExpr(left), StateExpr(right))
            return state

        elif data == "imply_":
            # Handle implication (a => b is equivalent to !a | b).
            left = self._convert(children[0])
            right = self._convert(children[1])
            # Create a new state for the negated left-hand side.
            not_left = self.fresh_state("notleft")
            self.states.add(not_left)
            self.transitions[not_left] = NotExpr(StateExpr(left))

            state = self.fresh_state("imply")
            self.states.add(state)
            self.transitions[state] = OrExpr(StateExpr(not_left), StateExpr(right))
            return state
        
        elif data == "comparison":
            # Handle an arithmetic comparison (e.g., "x > 5").
            variable, op, value_node = children
            var_name = variable.value
            operator = op.value
            try:
                val = float(value_node.value)
            except:
                val = value_node.value
            # Create the comparison expression node.
            expr = ComparisonExpr(var_name, operator, val)

            # Create a unique string representation to treat it like a single atomic symbol.
            # This is key for integrating arithmetic into the propositional structure.
            symbolic_representation = f"{var_name}{operator}{val}"

            state = self.fresh_state(f"cmp_{var_name}")
            self.states.add(state)
            self.alphabet.add(symbolic_representation) # Add the unique symbol to the alphabet.
            self.transitions[state] = expr
            self.accepting.add(state) # Comparisons are atomic facts, thus accepting.
            return state
        
        raise NotImplementedError(f"CTL operator '{data}' not yet implemented.")
        
    def _convert(self, tree: Parser.Tree) -> State:
        """
        Recursively walks the parsed formula tree and builds the ABTA states and transitions.
        """
        if isinstance(tree, Parser.Token):
            # Base case: A token is treated as an atomic proposition value.
            return tree.value
        
        # Descend past wrapper nodes.
        if tree.data == "start" or tree.data == "startN":
            tree = tree.children[0]
        
        data = tree.data
        children = tree.children

        # Dispatch to the appropriate handler based on the node type.
        if data in ("atom", "or_", "and_", "neg", "imply", "comparison"):
            return self.handle_basic(data, children)
        elif data in ("ef", "af", "eg", "ag"):
            return self.handle_operator(data, children)
        
        # Handle Until operators.
        elif data == "eu":
            # E[φ1 U φ2]  ≡  φ2 ∨ (φ1 ∧ E(next) E[φ1 U φ2])
            φ1 = self._convert(children[0])
            φ2 = self._convert(children[1])
            state = self.fresh_state("eu")
            self.states.add(state)
            self.transitions[state] = OrExpr(
                StateExpr(φ2),
                AndExpr(
                    StateExpr(φ1),
                    OrExpr(NextExpr("left", state), NextExpr("right", state)) # E(next)
                )
            )
            self.accepting.add(state) # Accepting loop, as it's a safety property until φ2.
            return state

        elif data == "au":
            # A[φ1 U φ2]  ≡  φ2 ∨ (φ1 ∧ A(next) A[φ1 U φ2])
            
            φ1 = self._convert(children[0])
            φ2 = self._convert(children[1])
            state = self.fresh_state("au")
            self.states.add(state)
            self.transitions[state] = OrExpr(
                StateExpr(φ2),
                AndExpr(
                    StateExpr(φ1),
                    AndExpr(NextExpr("left", state), NextExpr("right", state)) # A(next)
                )
            )
            # AU is a liveness property. The loop state is non-accepting because φ2 *must*
            # eventually be true. The paths fulfilling the obligation lead to φ2's state.
            return state
            
        elif data == "ew":
            # E[φ1 W φ2] (Weak Until) ≡ φ2 ∨ (φ1 ∧ E(next) E[φ1 W φ2])
            φ1 = self._convert(children[0])
            φ2 = self._convert(children[1])
            state = self.fresh_state("ew")
            self.states.add(state)
            self.transitions[state] = OrExpr(
                StateExpr(φ2),
                AndExpr(
                    StateExpr(φ1),
                    OrExpr(NextExpr("left", state), NextExpr("right", state)) # E(next)
                )
            )
            # EW is a safety property. The loop is accepting because φ1 can hold forever.
            self.accepting.add(state)
            return state

        elif data == "aw":
            # A[φ1 W φ2] (Weak Until) ≡ φ2 ∨ (φ1 ∧ A(next) A[φ1 W φ2])
            φ1 = self._convert(children[0])
            φ2 = self._convert(children[1])

            
            state = self.fresh_state("aw")
            self.states.add(state)
            self.transitions[state] = OrExpr(
                StateExpr(φ2),
                AndExpr(
                    StateExpr(φ1),
                    AndExpr(NextExpr("left", state), NextExpr("right", state)) # A(next)
                )
            )
            # AW is a safety property. The loop is accepting because φ1 can hold forever.
            self.accepting.add(state)
            return state
        
        elif data == "expr":
            # Handle generic expression wrapper nodes.
            return self._convert(children[0])

        else:
            raise NotImplementedError(f"CTL operator '{data}' not yet implemented.")

    def get_dnf(self, expr: TransitionExpr) -> List[Move]:
        """
        Converts a transition expression AST into Disjunctive Normal Form (DNF).
        Each clause in the DNF is a "Move", representing one possible way to
        satisfy the transition logic.

        Args:
            expr: The TransitionExpr AST node to convert.

        Returns:
            A list of Moves, where each Move is a tuple of (required_atoms, required_next_states).
        """
        if isinstance(expr, StateExpr):
            # If we encounter a reference to another state, recursively get its DNF.
            # This effectively inlines the definition of the referenced state.
            return self.get_dnf(self.transitions[expr.state])

        elif isinstance(expr, AtomicExpr):
            # An atomic proposition `p` becomes a single DNF clause: ({'p'}, {}).
            # It requires 'p' to be true now and has no temporal requirements.
            return [({expr.symbol}, set())]

        elif isinstance(expr, NextExpr):
            # A next expression `E(dir)q` becomes a clause: ({}, {(dir, q)}).
            # It has a temporal requirement but no atomic proposition requirement.
            return [(set(), {(expr.direction, expr.state)})]

        elif isinstance(expr, OrExpr):
            # DNF of (A ∨ B) is (DNF of A) ∪ (DNF of B).
            # We simply concatenate the lists of moves from both branches.
            return self.get_dnf(expr.left) + self.get_dnf(expr.right)

        elif isinstance(expr, AndExpr):
            # DNF of (A ∧ B) is the pairwise combination of clauses from DNF(A) and DNF(B).
            # This is the most complex part of the conversion.
            left_dnf = self.get_dnf(expr.left)
            right_dnf = self.get_dnf(expr.right)
            
            combined_dnf = []
            # For each move in the left DNF...
            for atoms_l, next_l in left_dnf:
                # ...combine it with each move in the right DNF.
                for atoms_r, next_r in right_dnf:
                    # The new move requires the union of atomic propositions and the union of next moves.
                    combined_dnf.append((atoms_l | atoms_r, next_l | next_r))
            return combined_dnf

        elif isinstance(expr, TrueExpr):
            # 'true' is an empty requirement: no atoms, no next moves.
            return [(set(), set())]
            
        elif isinstance(expr, FalseExpr):
            # 'false' is an impossible requirement; it has no satisfying moves.
            return []

        elif isinstance(expr, ComparisonExpr):
            # A comparison is treated like an atom. Its unique string representation is used.
            symbolic_representation = f"{expr.variable}{expr.operator}{expr.value}"
            return [({symbolic_representation}, set())]
            
        # Default case for any other unhandled expression types.
        return []
    
    def check_subset(self, a: Set[Symbol], b: Set[Symbol]) -> bool:
        """
        A cached wrapper for the semantic subset check. Checks if B => A.
        This is used in the simulation check to see if one set of atomic
        propositions logically entails another.

        Args:
            a: The set of conclusion propositions.
            b: The set of premise propositions.

        Returns:
            True if the propositions in B semantically entail the propositions in A.
        """
        # Create a cache key from the frozensets for immutability.
        cache_key = (frozenset(a), frozenset(b))
        if cache_key in self._entailment_cache:
            return self._entailment_cache[cache_key]
        
        # Note the order: check_semantic_subset_z3 checks if its first argument
        # implies the second. In simulation, we need to check if the Duplicator's
        # move (b) implies the Spoiler's move (a).
        result = check_semantic_subset_z3(b, a)
        self._entailment_cache[cache_key] = result
        return result

    def is_simulated_by(self, other: 'ABTA') -> Tuple[bool, int]:
        """
        Checks if this automaton (self) is simulated by another automaton (other).
        This determines if the language of `self` is a subset of the language of `other`.
        The algorithm computes the greatest fixed point of the simulation relation.

        Args:
            other: The other ABTA (the potential simulator, or "Duplicator").

        Returns:
            A tuple containing:
            - A boolean indicating if the simulation holds.
            - An integer representing the size of the final simulation relation.
        """
        if other == self:
            return True, len(self.states)

        aut_self = self   # The "Spoiler"
        aut_other = other # The "Duplicator"
        
        # Clear and prepare the entailment cache for this simulation check.
        self._entailment_cache = {}

        # Pre-compute the DNF for all states in both automata for efficiency.
        # This avoids redundant calculations inside the main fixed-point loop.
        dnf_self = {q: aut_self.get_dnf(aut_self.transitions[q]) for q in aut_self.states}
        dnf_other = {q: aut_other.get_dnf(aut_other.transitions[q]) for q in aut_other.states}

        # Initialize the simulation relation `R` with all possible state pairs,
        # then perform an initial pruning: a pair (q_s, q_o) is invalid if q_s
        # is accepting but q_o is not (violates the Büchi condition).
        R = set()
        for q_s in aut_self.states:
            for q_o in aut_other.states:
                if not (q_s in aut_self.accepting and q_o not in aut_other.accepting):
                    R.add((q_s, q_o))

        # Iteratively refine the simulation relation `R` until a fixed point is reached.
        while True:
            R_prime = set() # The relation for the next iteration.
            
            # For each pair currently in the relation...
            for q_self, q_other in R:
                is_pair_good = True # Assume the pair is valid until proven otherwise.
                
                moves_self = dnf_self[q_self]   # Spoiler's possible moves from q_self.
                moves_other = dnf_other[q_other] # Duplicator's possible moves from q_other.
                
                # For EVERY move the Spoiler makes...
                for atoms_self, next_self in moves_self:
                    found_matching_move = False
                    # ...the Duplicator must have AT LEAST ONE valid response.
                    for atoms_other, next_other in moves_other:
                        
                        # Condition 1: Propositional Entailment.
                        # The Duplicator's atoms must entail the Spoiler's atoms.
                        if not self.check_subset(atoms_self, atoms_other):
                            continue # This move doesn't work, try the next one.

                        # Condition 2: Successor States.
                        # All required successors of the Spoiler's move must be matched
                        # by the Duplicator's move, and the resulting state pairs
                        # must themselves be in the current simulation relation `R`.
                        other_next_map = {direction: state for direction, state in next_other}
                        successors_are_good = True
                        for dir_self, state_self in next_self:
                            # Duplicator must have a move in the same direction.
                            if dir_self not in other_next_map:
                                successors_are_good = False
                                break
                            
                            # The resulting pair of states must be in R.
                            state_other = other_next_map[dir_self]
                            if (state_self, state_other) not in R:
                                successors_are_good = False
                                break
                        
                        if successors_are_good:
                            # A matching move was found for the Spoiler's current move.
                            found_matching_move = True
                            break # No need to check other moves from the Duplicator.

                    if not found_matching_move:
                        # If the Duplicator has no response to this Spoiler move,
                        # the pair (q_self, q_other) is not in the simulation.
                        is_pair_good = False
                        break # No need to check other moves from the Spoiler.

                # If the Duplicator could counter all of the Spoiler's moves...
                if is_pair_good:
                    R_prime.add((q_self, q_other)) #...keep the pair for the next iteration.

            # If the relation did not shrink, we have reached the greatest fixed point.
            if R == R_prime:
                break
            
            # Otherwise, update R and continue refining.
            R = R_prime

        # The overall simulation holds if the pair of initial states is in the final relation.
        initial_pair = (aut_self.initial_state, aut_other.initial_state)
        # Symmetrically check the inverse, as the relation might be defined either way.
        initial_pair_inv = (aut_other.initial_state, aut_self.initial_state)
        
        return initial_pair in R or initial_pair_inv in R, len(R)