from __future__ import annotations

from functools import cached_property, lru_cache
from typing import Tuple, Dict

from . import Parser
from .abta import ABTA
Interval = Tuple[int, int]

_UNARY_ORDER: Dict[str, tuple[str, ...]] = {
    "ag": ("ag", "af", "ax", "a", "eg", "ef", "ex", "e"),
    "eg": ("eg", "ef", "ex", "e"),
    "af": ("af", "ef"),
    "ef": ("ef",),
    "eu":("eu")
}
_INF = 10 ** 9

class CTLProperty:
    """Immutable wrapper around a TCTL formula and its parse tree.

    Instances are cached by formula string so that syntactic equality implies
    object identity.  Caching also helps ensure that repeated parses are not
    performed when the same formula is encountered often (e.g. during model
    checking).
    """

    __slots__ = ("formula", "_tree", "atomic_props", "sub_formulas", "abta")

    _cache: Dict[str, "CTLProperty"] = {}

    # ------------------------------------------------------------------
    # construction & identity
    # ------------------------------------------------------------------
    #def __new__(cls, formula: str) -> "CTLProperty":
    #    try:
    #        return cls._cache[formula]
    #    except KeyError:
    #        instance = super().__new__(cls)
    #        cls._cache[formula] = instance
    #        return instance

    def __init__(self, formula: str) -> None:
        if getattr(self, "_initialized", False):
            return  # already initialised via __new__ path reuse
        self.formula = formula
        parsed = Parser.PARSER.parse(formula)
        if isinstance(parsed, Parser.Tree) and parsed.data == "start":
            first_child = parsed.children[0]
            #if not (isinstance(first_child, Parser.Tree) and first_child.data in _UNARY_ORDER):
            #    # Wrap first child in AG with default time constraint []
            #    default_time = Parser.Tree("time_constraint", [Parser.Token("TIME", "[]")])
            #    new_ag = Parser.Tree("ag", [default_time, first_child])
            #    parsed = Parser.Tree("start", [new_ag])
            #    self.formula = "AG"+self.formula
        self._tree= Parser.NNF.to_nnf(parsed)
        #print(self._tree)
        #print(self._tree.pretty())
        
        temp_1 = Parser.AtomCollector()
        temp_1.visit(self._tree)
        temp_2 = Parser.SubformulaCollector()
        temp_2.visit(self._tree)
       
        self.atomic_props = temp_1.atoms
        #print(self.atomic_props)
        self.sub_formulas = temp_2.subformulas
        self.abta = ABTA(formula)

    def atoms(self):
        return self.atomic_props
    def subformulas(self):
        return self.sub_formulas


    def tree(self) -> Parser.Tree:  # noqa: D401
        """Direct access to the (read‑only) parse tree."""
        return self._tree

    # ------------------------------------------------------------------
    # comparison & human‑friendly representation
    # ------------------------------------------------------------------
    def __hash__(self) -> int:
        return hash(self.formula)

    def __eq__(self, other: object) -> bool:  # noqa: D401
        return isinstance(other, CTLProperty) and self.formula == other.formula

    def __repr__(self) -> str:  # noqa: D401
        return f"{self.formula!r})"

    # ------------------------------------------------------------------
    # syntactic refinement public API
    # ------------------------------------------------------------------
    def refines_s(self, other: "CTLProperty") -> bool:
        """Return *True* if *self* syntactically refines *other*.

        The test is performed over the children of the implicit top‑level
        node inserted by the grammar (typically labelled "start").
        """
        return self._refines(self._tree.children[0], other._tree.children[0])
    def refines(self, other: "CTLProperty", use_syn = False) -> bool:
        if use_syn:
            if self.refines_s(other):
                return True
        return  other.abta.is_simulated_by(self.abta)[0]
    # ------------------------------------------------------------------
    # internal helpers (interval handling, recursion engine)
    # ------------------------------------------------------------------
    @staticmethod
    def _interval(node: Parser.Tree) -> Interval:
        """Extract the closed interval [lo, hi] from *node* (or return default)."""
        try:
            token = node.children[0].children[0]
            lo, hi = map(int, token.value.strip("[]").split(","))
            return lo, hi
        except Exception:
            return (0, _INF)

    @staticmethod
    def _interval_subsumes(inner: Interval, outer: Interval) -> bool:
        lo_i, hi_i = inner
        lo_o, hi_o = outer
        return lo_o <= lo_i and hi_i <= hi_o

    # the core refinement algorithm ------------------------------------

    @lru_cache(maxsize=None)
    def _refines(self, t1: Parser.Tree, t2: Parser.Tree) -> bool:  # noqa: C901
        """Recursive engine deciding syntactic refinement (memoised)."""
        # structural equality — quick exit
        if t1 == t2:
            return True
        # atoms ---------------------------------------------------------
        if t1.data == t2.data == "atom":
            return t1.children[0] == t2.children[0]

        # negation via contrapositive ----------------------------------
        if t1.data == t2.data == "neg":
            if isinstance(t1.children[0], Parser.Token) and isinstance(t2.children[0], Parser.Token):
                return t1.children[0] == t2.children[0]
            #if isinstance(t1.children[0], Parser.Token):
            #    return self._refines(t2.children[0], t1.children[0])

            return self._refines(t2.children[0], t1.children[0])

        # boolean connectives -----------------------------------------
        match t1.data, t2.data:
            case "and_", _:
                left, right = t1.children
                return self._refines(left, t2) or self._refines(right, t2)
            case _, "and_":
                left, right = t2.children
                return self._refines(t1, left) and self._refines(t1, right)
            case "or_", _:
                left, right = t1.children
                return self._refines(left, t2) and self._refines(right, t2)
            case _, "or_":
                left, right = t2.children
                return self._refines(t1, left) or self._refines(t1, right)

        # implication ---------------------------------------------------
        if t1.data == "imply_":
            if(t1.children[0]==t2.children[0]):
                return self._refines(t1.children[1], t2.children[1])
            # conservative — treat implication in antecedent as unknown
            return False
        if t2.data == "imply_":
            antecedent, consequent = t2.children
            conj = Parser.Tree("and_", [t1, antecedent])
            return self._refines(conj, consequent)

        # unary temporal operators -------------------------------------
        if (
            t1.data in _UNARY_ORDER
            and t2.data in _UNARY_ORDER[t1.data]
        ):
            sub1, sub2 = t1.children[-1], t2.children[-1]
            I1, I2 = self._interval(t1), self._interval(t2)

            # For G-type: AG, EG — wider interval is stronger
            if t1.data in {"ag", "eg"}:
                if self._interval_subsumes(I1, I2):
                    return self._refines(sub1, sub2)

            # For F-type: AF, EF — narrower interval is stronger
            elif t1.data in {"af", "ef"}:
                if self._interval_subsumes(I2, I1):  # t1 narrower than t2
                    return self._refines(sub1, sub2)

            # Allow cross-op strengthening if temporal op is stronger AND intervals permit
            if t1.data != t2.data:
                if t1.data in {"ag", "eg"} and self._interval_subsumes(I1, I2):
                    return self._refines(sub1, sub2)
                if t1.data in {"af", "ef"} and self._interval_subsumes(I2, I1):
                    return self._refines(sub1, sub2)

        # until ---------------------------------------------------------
        if t1.data == t2.data == "eu":
            φ1, ψ1 = t1.children
            φ2, ψ2 = t2.children
            return self._refines(φ1, φ2) and self._refines(ψ1, ψ2)

        # fallback — no refinement relationship recognised
        return False








