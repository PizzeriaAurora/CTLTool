from typing import Set, Dict, Union, Tuple
from dataclasses import dataclass
from . import Parser
State = str
Symbol = str
Direction = str  # left or right for binary trees, or more for k-ary

class TransitionExpr:
    pass

@dataclass
class TrueExpr(TransitionExpr): pass
@dataclass
class FalseExpr(TransitionExpr): pass
@dataclass
class AtomicExpr(TransitionExpr):
    symbol: Symbol

@dataclass
class AndExpr(TransitionExpr):
    left: TransitionExpr
    right: TransitionExpr

@dataclass
class OrExpr(TransitionExpr):
    left: TransitionExpr
    right: TransitionExpr

@dataclass
class NextExpr(TransitionExpr):
    direction: Direction
    state: State

@dataclass
class StateExpr(TransitionExpr):
    state: State

@dataclass
class NotExpr(TransitionExpr):
    expr: TransitionExpr


@dataclass
class EUExpr(TransitionExpr):
    left: TransitionExpr
    right: TransitionExpr
@dataclass
class EGExpr(TransitionExpr):
    expr: TransitionExpr
@dataclass
class AGExpr(TransitionExpr):
    expr: TransitionExpr







class ABTA:
    states: Set[State]
    alphabet: Set[Symbol]
    initial_state: State
    transitions: Dict[State, TransitionExpr]
    accepting: Set[State]
    mem = Dict
    
    def __init__(self,tree: Parser.Tree = None):
        #self.automaton = ABTA(states=set(), alphabet=set(), initial_state="", transitions={}, accepting=set())
        self.state_counter = 0
        self.states = set()
        self.alphabet = set()
        self.accepting = set()
        self.transitions = {}
        self.memo = {}
        self.atomic_map: Dict[State, Set[Symbol]] = {}
        self.initial_state=""
        if tree is not None:
            self.convert(tree)
    
    def __repr__(self) -> str:  # noqa: D401
        to_return= f"Constructed ABTA:\n Initial state: {self.initial_state}\n States: {self.states} \n"+\
        f"Accepting states: {self.accepting} \n Transitions:\n"
        
        for state, trans in self.transitions.items():
            to_return += f"  {state} -> {trans}\n"
        return to_return

    def fresh_state(self, prefix="q"):
        self.state_counter += 1
        return f"{prefix}{self.state_counter}"
    def convert(self, tree: Parser.Tree):
        initial = self._convert(tree)
        self.initial_state = initial
    def handle_operator(self,data, children):
        if data == "ef":
            φ = self._convert(children[1])
            state = self.fresh_state("ef")
            self.states.add(state)
            self.transitions[state] = OrExpr(
                StateExpr(φ),
                OrExpr(NextExpr("left", state), NextExpr("right", state))
            )
            self.accepting.add(state)
            return state

        elif data == "eg":
            φ = self._convert(children[1])
            state = self.fresh_state("eg")
            self.states.add(state)
            self.transitions[state] = AndExpr(
                StateExpr(φ),
                OrExpr(NextExpr("left", state), NextExpr("right", state))
            )
            self.accepting.add(state)
            return state

        elif data == "af":
            φ = self._convert(children[1])
            state = self.fresh_state("af")
            self.states.add(state)
            self.transitions[state] = OrExpr(
                StateExpr(φ),
                AndExpr(NextExpr("left", state), NextExpr("right", state))
            )
            self.accepting.add(state)
            return state

        elif data == "ag":
            φ = self._convert(children[1])
            state = self.fresh_state("ag")
            self.states.add(state)
            self.transitions[state] = AndExpr(
                StateExpr(φ),
                AndExpr(NextExpr("left", state), NextExpr("right", state))
            )
            self.accepting.add(state)
            return state
    
    def handle_basic(self,data, children):
        if data == "atom":
            sym = children[0].value
            state = self.fresh_state(f"atom_{sym}") # Always create a fresh state
            self.states.add(state)
            self.atomic_map[state] = {sym}
            self.alphabet.add(sym)
            # The transition from an atomic state is simply True, 
            # as its properties are defined by the atomic_map.
            self.transitions[state] = TrueExpr() 
            return state

        elif data == "neg":
            sub = self._convert(children[0])
            state = self.fresh_state("neg")
            self.states.add(state)
            self.transitions[state] = NotExpr(StateExpr(sub))
            return state

        elif data == "and_":
            left = self._convert(children[0])
            right = self._convert(children[1])
            state = self.fresh_state("and")
            self.states.add(state)
            self.transitions[state] = AndExpr(StateExpr(left), StateExpr(right))
            return state

        elif data == "or_":
            left = self._convert(children[0])
            right = self._convert(children[1])
            state = self.fresh_state("or")
            self.states.add(state)
            self.transitions[state] = OrExpr(StateExpr(left), StateExpr(right))
            return state

        elif data == "imply_":
            left = self._convert(children[0])
            right = self._convert(children[1])
            # A → B ≡ ¬A ∨ B
            not_left = self.fresh_state("notleft")
            self.states.add(not_left)
            self.transitions[not_left] = NotExpr(StateExpr(left))

            state = self.fresh_state("imply")
            self.states.add(state)
            self.transitions[state] = OrExpr(StateExpr(not_left), StateExpr(right))
            return state
        
    def _convert(self, tree: Parser.Tree) -> State:
        if isinstance(tree, Parser.Token):
            return tree.value
        if tree.data == "start":
            tree = tree.children[0]
        data = tree.data
        children = tree.children

        if data in ("atom", "or_", "and_", "neg", "imply"):
            return self.handle_basic(data, children)
        elif data in ("ef", "af", "eg", "ag"):
            return self.handle_operator(data, children)
        elif data == "eu":
            φ1 = self._convert(children[0])
            φ2 = self._convert(children[2])
            state = self.fresh_state("eu")
            self.states.add(state)
            self.transitions[state] = OrExpr(
                StateExpr(φ2),
                AndExpr(
                    StateExpr(φ1),
                    OrExpr(NextExpr("left", state), NextExpr("right", state))
                )
            )
            self.accepting.add(state)
            return state
        elif data == "expr":
            return self._convert(children[0])

        else:
            raise NotImplementedError(f"CTL operator '{data}' not yet implemented.")
    # ---- Simulate function with memoization and cycle detection ----
    def _check_atomic_consistency(self, q1: State, q2: State, other: 'ABTA') -> bool:
        """Checks if atoms at q1 are a superset of atoms at q2."""
        atoms1 = self.atomic_map.get(q1, set())
        atoms2 = other.atomic_map.get(q2, set())
        return atoms2.issubset(atoms1)


    
    def _simulate(self, phi1: TransitionExpr, phi2: TransitionExpr, R: Set[Tuple[str, str]], call_stack: Set[Tuple[int, int]]) -> bool:
        """
        Checks if phi1 simulates phi2, given the current simulation relation R.
        This is a co-inductive check, assuming recursion is true (for greatest fixpoint).
        """
        key = (id(phi1), id(phi2))
        
        if key in call_stack:
            # Recursion detected. In a greatest fixpoint calculation, we assume
            # the recursive hypothesis holds until proven otherwise.
            return True
        if key in self.memo:
            return self.memo[key]

        call_stack.add(key)
        
        # --- Base Cases ---
        if isinstance(phi2, TrueExpr):
            result = True
        elif isinstance(phi2, FalseExpr):
            # phi1 can only simulate False if phi1 is itself False.
            result = isinstance(phi1, FalseExpr)
        
        # --- Inductive Cases (Pattern match on phi2 first) ---
        elif isinstance(phi2, AtomicExpr):
            # phi1 can only simulate an atomic proposition if it is the *same* proposition.
            if isinstance(phi1, AtomicExpr):
                result = phi1.symbol == phi2.symbol
            else: # A complex formula cannot be proven to simulate a single atom.
                result = False
        elif isinstance(phi2, StateExpr):
            print(phi1, phi2,"here")
            # To simulate a state q2, phi1 must entail some state q1 where (q1, q2) is in R.
            if isinstance(phi1, StateExpr):
                result = (phi1.state, phi2.state) in R
            elif isinstance(phi1, AndExpr):
                # If phi1 is A & B, either A or B must entail q2.
                result = self._simulate(phi1.left, phi2, R, call_stack) or self._simulate(phi1.right, phi2, R, call_stack)
            elif isinstance(phi1, OrExpr):
                # If phi1 is A | B, either A or B must entail q2. (This is the same logic as And for this direction)
                # This seems counterintuitive, but for phi1 to *guarantee* a property, one of its components must.
                # A better model would be a SAT check, but within this recursive structure, this is a plausible heuristic.
                # A more accurate check for Or is that *both* sides must simulate phi2.
                # If the choice is A or B, and we need to guarantee q2, then both A and B must guarantee it.
                result = self._simulate(phi1.left, phi2, R, call_stack) and self._simulate(phi1.right, phi2, R, call_stack)
            else: # Atomic, Next, etc. cannot simulate a StateExpr
                result = False
        elif isinstance(phi2, NotExpr):
            # To simulate ¬B, phi1 must also be a negation, say ¬A.
            # The rule is: ¬A simulates ¬B iff B simulates A (contraposition).
            if isinstance(phi1, NotExpr):
                # This is the crucial part: the arguments are flipped!
                result = self._simulate(phi2.expr, phi1.expr, R, call_stack)
            else:
                # A positive formula (And, Or, etc.) cannot simulate a negative one
                # in this framework. This highlights the brittleness of not using PNF.
                result = False
        
        elif isinstance(phi2, NextExpr):
            print(phi1, phi2,"here")
            # Similar logic to StateExpr
            if isinstance(phi1, NextExpr):
                result = (phi1.direction == phi2.direction) and ((phi1.state, phi2.state) in R)
            elif isinstance(phi1, AndExpr):
                result = self._simulate(phi1.left, phi2, R, call_stack) or self._simulate(phi1.right, phi2, R, call_stack)
            elif isinstance(phi1, OrExpr):
                result = self._simulate(phi1.left, phi2, R, call_stack) and self._simulate(phi1.right, phi2, R, call_stack)
            else:
                result = False
                
        
                
        elif isinstance(phi2, AndExpr):
            # To simulate A & B, phi1 must simulate A AND phi1 must simulate B.
            result = self._simulate(phi1, phi2.left, R, call_stack) and self._simulate(phi1, phi2.right, R, call_stack)

        elif isinstance(phi2, OrExpr):
            # To simulate A | B, phi1 must simulate A OR phi1 must simulate B.
            if isinstance(phi1, OrExpr):
                # If phi1 is also an OR, we can distribute the check.
                # This is an optimization. (A|B) simulates (C|D) if A simulates (C|D) or B simulates (C|D)
                result = self._simulate(phi1.left, phi2, R, call_stack) or self._simulate(phi1.right, phi2, R, call_stack)
            elif isinstance(phi1, AndExpr):
                # If phi1 is A&B, it must simulate A|B. This is always true if A simulates A|B or B simulates A|B
                # Since this is a check of implication, it's complex. The simplest check is the general one below.
                result = self._simulate(phi1, phi2.left, R, call_stack) or self._simulate(phi1, phi2.right, R, call_stack)
            else:
                # General case
                result = self._simulate(phi1, phi2.left, R, call_stack) or self._simulate(phi1, phi2.right, R, call_stack)

        # Note: Negation is tricky. This implementation assumes Positive Normal Form.
        

        else:
            # Fallback for unhandled expressions like NotExpr, EUExpr, etc.
            result = False
        print(result)
        self.memo[key] = result
        call_stack.remove(key)
        return result
    def _simulate_with_fixpoint(self, other: 'ABTA') -> Set[Tuple[str, str]]:
        """
        Compute the greatest simulation relation R ⊆ Q1 x Q2.
        This version correctly integrates atomic checks into the fixpoint loop.
        """
        # R starts with all pairs that satisfy the accepting condition.
        R = {
            (q1, q2)
            for q1 in self.states
            for q2 in other.states
            if (not (q2 in other.accepting) or (q1 in self.accepting))
        }

        # Memo cache for simulate calls
        self.memo = {} 
        
        changed = True
        print(R)
        while changed:
            changed = False
            to_remove = set()
            for (q1, q2) in R:
                # A pair (q1, q2) is invalid if:
                # 1. Atomic propositions are inconsistent.
                # OR
                # 2. The transition from q1 does not simulate the transition from q2.
                
                # Check condition 1: Atomic Consistency
                if not self._check_atomic_consistency(q1, q2, other):
                    to_remove.add((q1, q2))
                    continue # No need to check transitions if atoms already fail

                # Check condition 2: Transition Simulation
                phi1 = self.transitions.get(q1, TrueExpr()) # Default to True if no transition
                phi2 = other.transitions.get(q2, TrueExpr())
                
                # Reset memoization for each top-level state-pair check
                self.memo.clear() 
                
                if not self._simulate(phi1, phi2, R, set()):
                    to_remove.add((q1, q2))

            if to_remove:
                R -= to_remove
                changed = True
        
        self.memo.clear()
        return R


    def simulate(self,other : object) -> Tuple[bool, int]:
        sim_rel = self._simulate_with_fixpoint(other)
        print(sim_rel)
        return ((self.initial_state, other.initial_state) in sim_rel,len(sim_rel))
