import random
import subprocess
import time
import argparse
import os
import sys
from typing import List, Tuple, Dict, Any

from .Analyzer import Analyzer
from .Parser import  PARSER, NUSMV, CTLSATTRANS


def to_sat_syntax(custom_prop: str) -> str:
    """Translates a property from custom syntax to NuSMV syntax."""
    try:
        tree = PARSER.parse(custom_prop)
        return CTLSATTRANS.transform(tree)
    except Exception as e:
        print(f"Error parsing or transforming property: {custom_prop}")
        raise e

def to_nusmv_syntax(custom_prop: str) -> str:
    """Translates a property from custom syntax to NuSMV syntax."""
    try:
        tree = PARSER.parse(custom_prop)
        return NUSMV.transform(tree)
    except Exception as e:
        print(f"Error parsing or transforming property: {custom_prop}")
        raise e
import random
from typing import List, Tuple, Dict
from lark import Lark, Transformer, v_args
from .Parser import NUSMV


import random
import re
from typing import List, Dict, Any, Tuple

class SimplePropertyGenerator:
    """
    Generates CTL properties based on equivalence classes of atomic propositions.

    The generator first partitions the given atomic propositions into a specified
    number of "equivalence classes". Each generated property and its subsequent
    refinements will only use atomic propositions from a single, specific class.

    The generator uses a custom, human-readable syntax and provides a translation
    to standard NuSMV syntax.

    Custom Syntax Examples:
    - AG (p & q)
    - A(p U q)
    - (EF r) -> (AG q)

    NuSMV Syntax Examples:
    - AG (p & q)
    - A [ p U q ]
    - (EF r) -> (AG q)
    """
    def __init__(self, atomic_props: List[str], num_equivalence_classes: int):
        if not atomic_props:
            raise ValueError("Atomic propositions list cannot be empty.")
        if not 1 <= num_equivalence_classes <= len(atomic_props):
            raise ValueError(
                f"Number of equivalence classes ({num_equivalence_classes}) must be "
                f"between 1 and the number of atomic props ({len(atomic_props)})."
            )

        self.all_atomic_props = atomic_props
        self.num_classes = num_equivalence_classes
        self.prop_classes = self._partition_props(atomic_props, num_equivalence_classes)

    def _partition_props(self, props: List[str], k: int) -> List[List[str]]:
        """Partitions a list of propositions into k pseudo-random sublists."""
        partitions = [[] for _ in range(k)]
        # Shuffle for random distribution
        shuffled_props = random.sample(props, len(props))
        for i, prop in enumerate(shuffled_props):
            partitions[i % k].append(prop)
        return partitions

    def _translate_to_nusmv(self, prop: str) -> str:
        """Translates a property from custom syntax to NuSMV syntax."""
        # NuSMV uses square brackets for U and W operators.
        # Use non-greedy matching (.*?) to handle nested properties correctly.
        
        return to_nusmv_syntax(prop)

    def _make_atom(self, class_index: int) -> str:
        """Picks a random atomic proposition from a specific equivalence class."""
        return random.choice(self.prop_classes[class_index])

    def _make_primary(self, class_index: int, depth=0) -> str:
        """Generates a primary CTL expression from a specific class."""
        if depth > 2:  # Prevent excessive nesting
            return self._make_atom(class_index)

        # Pass class_index down to recursive calls
        patterns = [
            lambda: self._make_atom(class_index),
            lambda: f"EF {self._make_unary(class_index, depth+1)}",
            lambda: f"AG {self._make_unary(class_index, depth+1)}",
            lambda: f"AF {self._make_unary(class_index, depth+1)}",
            lambda: f"EG {self._make_unary(class_index, depth+1)}",
            lambda: f"A({self._make_expr(class_index, depth+1)} U {self._make_expr(class_index, depth+1)})",
            lambda: f"E({self._make_expr(class_index, depth+1)} U {self._make_expr(class_index, depth+1)})",
            lambda: f"A({self._make_expr(class_index, depth+1)} W {self._make_expr(class_index, depth+1)})",
        ]
        return random.choice(patterns)()

    def _make_unary(self, class_index: int, depth=0) -> str:
        """Generates a unary expression (or a primary one) from a specific class."""
        if random.random() < 0.8 or depth > 3:
            return self._make_primary(class_index, depth)
        else:
            return f"!({self._make_unary(class_index, depth+1)})"

    def _make_expr(self, class_index: int, depth=0) -> str:
        """Generates a binary boolean expression from a specific class."""
        # Generate a simple expression, avoiding deep recursion for base cases
        if random.random() < 0.5:
            op = '&'
        else:
            op = '|'
        return f"({self._make_unary(class_index, depth+1)} {op} {self._make_unary(class_index, depth+1)})"

    def _generate_base_property(self, class_index: int) -> str:
        """Generates a single, random CTL property from a specific class."""
        if random.random() < 0.2:
            # Implication
            left = self._make_primary(class_index)
            right = self._make_primary(class_index)
            return f"({left}) -> ({right})"
        return self._make_primary(class_index)

    def _refine_property(self, prop: str, class_index: int) -> str:
        """Creates a stricter version of a property using atoms from its class."""
        extra_condition = self._make_atom(class_index)
        
        # Atoms from the property's own class
        atoms_in_class = self.prop_classes[class_index]
        # Find which of those atoms appear in the property string
        replaceable_atoms = [p for p in atoms_in_class if re.search(r'\b' + p + r'\b', prop)]
        
        # Strategy 1: Replace an existing atom with a conjunction
        if replaceable_atoms and random.random() < 0.7:
            chosen_atom = random.choice(replaceable_atoms)
            # Use regex to replace whole word to avoid replacing 'p' in 'prop'
            return re.sub(r'\b' + chosen_atom + r'\b', f"({chosen_atom} & {extra_condition})", prop, 1)
        
        # Strategy 2: Add a global constraint
        else:
            return f"({prop}) & AG ({extra_condition})"

    def generate(
        self, num_per_class: int, num_refinements: int
    ) -> Dict[int, List[Dict[str, str]]]:
        """
        Generates properties for each equivalence class.

        Args:
            num_per_class: The number of base properties to generate for each class.
            num_refinements: The number of successive refinements to generate for each base property.

        Returns:
            A dictionary where keys are class indices and values are lists of generated
            properties (each as a dict with 'custom' and 'nusmv' keys).
        """
        properties_by_class = {}
        base_props = {}
        base_props["custom"] = []
        base_props["nusmv"] = []
        all_props = base_props
        for i in range(self.num_classes):
            class_properties = []
            for _ in range(num_per_class):
                # 1. Generate a base property
                current_prop = self._generate_base_property(class_index=i)
                class_properties.append({
                    "custom": current_prop,
                    "nusmv": self._translate_to_nusmv(current_prop)
                })
                base_props["custom"] += [current_prop]
                base_props["nusmv"] += [ self._translate_to_nusmv(current_prop)]
                all_props["custom"] += [current_prop]
                all_props["nusmv"] += [ self._translate_to_nusmv(current_prop)]
                # 2. Generate a chain of refinements from that base property
                for _ in range(num_refinements):
                    current_prop = self._refine_property(current_prop, class_index=i)
                    class_properties.append({
                        "custom": current_prop,
                        "nusmv": self._translate_to_nusmv(current_prop)
                    })
                    all_props["custom"] += [current_prop]
                    all_props["nusmv"] += [ self._translate_to_nusmv(current_prop)]
            properties_by_class[i] = class_properties
        return properties_by_class, all_props, base_props


class PropertyGenerator:
    """
    Generates a mix of simple and computationally difficult CTL properties.

    This generator preserves the original API for seamless integration. It enhances
    the generation logic to create properties known to stress BDD-based model
    checkers by using deep nesting, alternating quantifiers, and complex
    temporal clauses.

    The core logic of partitioning propositions into equivalence classes and
    generating refinements is maintained.
    """
    def __init__(self, atomic_props: List[str], num_equivalence_classes: int):
        if not atomic_props:
            raise ValueError("Atomic propositions list cannot be empty.")
        if not 1 <= num_equivalence_classes <= len(atomic_props):
            raise ValueError(
                f"Number of equivalence classes ({num_equivalence_classes}) must be "
                f"between 1 and the number of atomic props ({len(atomic_props)})."
            )

        self.all_atomic_props = atomic_props
        self.num_classes = num_equivalence_classes
        self.prop_classes = self._partition_props(atomic_props, num_equivalence_classes)

    def _partition_props(self, props: List[str], k: int) -> List[List[str]]:
        """Partitions a list of propositions into k pseudo-random sublists."""
        partitions = [[] for _ in range(k)]
        shuffled_props = random.sample(props, len(props))
        for i, prop in enumerate(shuffled_props):
            partitions[i % k].append(prop)
        return partitions

    def _translate_to_nusmv(self, prop: str) -> str:
        """Translates a property from custom syntax to NuSMV syntax."""
        # This function's body is assumed to exist elsewhere as per the prompt.
        # It's a placeholder for the actual translation call.
        return to_nusmv_syntax(prop)

    def _make_atom(self, class_index: int) -> str:
        """Picks a random atomic proposition from a specific equivalence class."""
        return random.choice(self.prop_classes[class_index])

    def _make_primary(self, class_index: int, depth=0) -> str:
        """Generates a primary CTL expression, now with deeper nesting allowed."""
        # Increased depth limit to allow for more complex properties
        if depth > 4:
            return self._make_atom(class_index)

        # Added alternating quantifiers, which are computationally expensive
        patterns = [
            lambda: self._make_atom(class_index),
            lambda: f"EF {self._make_unary(class_index, depth+1)}",
            lambda: f"AG {self._make_unary(class_index, depth+1)}",
            lambda: f"AF {self._make_unary(class_index, depth+1)}",
            lambda: f"EG {self._make_unary(class_index, depth+1)}",
            lambda: f"AG (EF {self._make_unary(class_index, depth+1)})", # Hard: Alternating
            lambda: f"AF (EG {self._make_unary(class_index, depth+1)})", # Hard: Alternating
            lambda: f"A({self._make_expr(class_index, depth+1)} U {self._make_expr(class_index, depth+1)})",
            #lambda: f"E({self._make_expr(class_index, depth+1)} U {self._make_expr(class_index, depth+1)})",
            lambda: f"A({self._make_expr(class_index, depth+1)} W {self._make_expr(class_index, depth+1)})",
        ]
        return random.choice(patterns)()

    def _make_unary(self, class_index: int, depth=0) -> str:
        """Generates a unary expression (or a primary one)."""
        # Slightly increase chance of negation for more complex boolean logic
        if random.random() < 0.7 or depth > 5:
            return self._make_primary(class_index, depth)
        else:
            return f"!({self._make_unary(class_index, depth+1)})"

    def _make_expr(self, class_index: int, depth=0) -> str:
        """Generates a binary boolean expression."""
        op = random.choice(['&', '|'])
        # Allow expressions to be composed of other expressions, not just primaries
        if random.random() < 0.4 and depth < 3:
             return f"({self._make_expr(class_index, depth+1)} {op} {self._make_unary(class_index, depth+1)})"
        return f"({self._make_unary(class_index, depth+1)} {op} {self._make_unary(class_index, depth+1)})"

    def _generate_base_property(self, class_index: int) -> str:
        """
        Generates a single, random CTL property, tiered by difficulty.
        """
        rand = random.random()

        # 40% chance of a simple property (good for baseline)
        if rand < 0.4:
            p = self._make_atom(class_index)
            q = self._make_atom(class_index)
            return random.choice([
                p,
                f"AG {p}",
                f"AF {p}",
                f"!({p} & {q})"
            ])

        # 45% chance of a medium-difficulty property (standard patterns)
        elif rand < 0.85:
            p = self._make_primary(class_index, depth=1)
            q = self._make_primary(class_index, depth=1)
            return random.choice([
                f"({p}) -> ({q})",
                f"AG({p} -> {q})",
                f"AG({self._make_atom(class_index)} -> AF {self._make_atom(class_index)})"
            ])

        # 15% chance of a hard property (deeply nested or complex)
        else:
            p = self._make_primary(class_index, depth=2)
            q = self._make_primary(class_index, depth=2)
            r = self._make_expr(class_index, depth=2)
            return random.choice([
                f"AG(AF {p})", # Often non-trivial
                f"A( {p} U {r} )", # Until with a complex expression
                f"AG(({p}) -> AF({q}))", # Classic nested liveness
                f"({self._generate_base_property(class_index)}) & ({self._generate_base_property(class_index)})" # conjunction of two props
            ])


    def _refine_property(self, prop: str, class_index: int) -> str:
        """
        Refines a CTL property making it strictly stronger,
        i.e., any model satisfying the refined property satisfies the original.

        Strategies:
        1. Conjunction with a new atomic proposition (adds stricter condition).
        2. Add temporal safety or liveness constraints conjunctively.
        3. Add nested "until" or "always" style formulas conjunctively.
        """

        atoms = self.prop_classes[class_index]

        # Find atomic propositions from the class in the property (simple heuristic)
        atoms_in_prop = [a for a in atoms if re.search(r'\b' + re.escape(a) + r'\b', prop)]

        # Choose a new atom from the class (prefer not already in prop to strengthen)
        candidate_atoms = [a for a in atoms if a not in atoms_in_prop]
        new_atom = random.choice(candidate_atoms) if candidate_atoms else random.choice(atoms)

        # Create new atoms for temporal formulas
        p = self._make_atom(class_index)
        q = self._make_atom(class_index)

        # Define stronger temporal constraints to conjunct
        temporal_strengtheners = [
            f"AG({p})",               # Always globally p (safety)
            f"AF({p})",               # Always eventually p (liveness)
            f"AG({p} -> AF({q}))",    # Fairness-like constraint: if p holds globally, q eventually holds
            f"A({p} U {q})",          # p until q holds
        ]

        # Choose refinement strategy weighted towards conjunctions (stronger)
        strategy_weights = ['conj_atom'] * 8 + ['conj_temporal'] * 3 + ['conj_until'] * 1
        strategy = random.choice(strategy_weights)

        if strategy == 'conj_atom':
            # Conjoin new atomic proposition
            replaceable_atoms = [atom for atom in atoms if re.search(r'\b' + re.escape(atom) + r'\b', prop)]
            if replaceable_atoms:
                chosen_atom = random.choice(replaceable_atoms)

                # Generate a new atom different from the chosen one to strengthen the condition
                extra_candidates = [a for a in atoms if a != chosen_atom]
                if not extra_candidates:
                    # If no alternative atom available, just reuse chosen_atom (still stricter by conjoining with itself)
                    extra_condition = chosen_atom
                else:
                    extra_condition = random.choice(extra_candidates)

                # Replace chosen atom with a stricter conjunction
                refined_prop = re.sub(
                    r'\b' + re.escape(chosen_atom) + r'\b',
                    f"({chosen_atom} & {extra_condition})",
                    prop,
                    count=1
                )
                return refined_prop

        elif strategy == 'conj_temporal':
            # Conjoin a temporal strengthener formula
            temporal_formula = random.choice(temporal_strengtheners[:-1])  # exclude until here
            return f"({prop}) & ({temporal_formula})"

        else:  # conj_until
            # Conjoin until formula
            until_formula = temporal_strengtheners[-1]
            return f"({prop}) & ({until_formula})"

    def _abstract_property(self, prop: str, class_index: int) -> str:

        atoms = self.prop_classes[class_index]

        # Find atomic propositions from the class in the property (simple heuristic)
        atoms_in_prop = [a for a in atoms if re.search(r'\b' + re.escape(a) + r'\b', prop)]

        # Choose a new atom from the class (prefer not already in prop to strengthen)
        candidate_atoms = [a for a in atoms if a not in atoms_in_prop]
        new_atom = random.choice(candidate_atoms) if candidate_atoms else random.choice(atoms)

        # Create new atoms for temporal formulas
        p = self._make_atom(class_index)
        q = self._make_atom(class_index)

        # Define stronger temporal constraints to conjunct
        temporal_strengtheners = [
            f"AG({p})",               # Always globally p (safety)
            f"AF({p})",               # Always eventually p (liveness)
            f"AG({p} -> AF({q}))",    # Fairness-like constraint: if p holds globally, q eventually holds
            f"A({p} U {q})",          # p until q holds
        ]

        # Choose refinement strategy weighted towards conjunctions (stronger)
        strategy_weights = ['disj_atom'] * 8 + ['relax_temporal'] * 2# + ['conj_until'] * 1
        strategy = random.choice(strategy_weights)

        if strategy == 'disj_atom':
            # Conjoin new atomic proposition
            replaceable_atoms = [atom for atom in atoms if re.search(r'\b' + re.escape(atom) + r'\b', prop)]
            if replaceable_atoms:
                chosen_atom = random.choice(replaceable_atoms)

                # Generate a new atom different from the chosen one to strengthen the condition
                extra_candidates = [a for a in atoms if a != chosen_atom]
                if not extra_candidates:
                    # If no alternative atom available, just reuse chosen_atom (still stricter by conjoining with itself)
                    extra_condition = chosen_atom
                else:
                    extra_condition = random.choice(extra_candidates)

                # Replace chosen atom with a stricter conjunction
                refined_prop = re.sub(
                    r'\b' + re.escape(chosen_atom) + r'\b',
                    f"({chosen_atom} | {extra_condition})",
                    prop,
                    count=1
                )
                return refined_prop

        elif strategy == 'relax_temporal':
            # Replace temporal operators with weaker ones
            # Example: AG(p) → AF(p), AF(p) → EF(p)
            prop = re.sub(r'\bAG\((.*?)\)', r'AF(\1)', prop)
            prop = re.sub(r'\bAF\((.*?)\)', r'EF(\1)', prop)
            return prop
#
        else:  # conj_until
            # Conjoin until formula
            until_formula = temporal_strengtheners[-1]
            return f"({prop}) & ({until_formula})"


    def generate(
        self, num_per_class: int, num_refinements: int,
        translator =None
    ) -> Dict[int, List[Dict[str, str]]]:
        """
        Generates properties for each equivalence class.
        """
        if translator == None:
            translator = self._translate_to_nusmv
        properties_by_class = {}
        all_props = {"custom": [], "translated": []}
        base_props = {"custom": [], "translated": []}

        for i in range(self.num_classes):
            class_properties = []
            for _ in range(num_per_class):
                # 1. Generate a base property using the new tiered logic
                current_prop = self._generate_base_property(class_index=i)
                nusmv_prop = translator(current_prop)

                class_properties.append({"custom": current_prop, "translated": nusmv_prop})
                base_props["custom"].append(current_prop)
                base_props["translated"].append(nusmv_prop)
                all_props["custom"].append(current_prop)
                all_props["translated"].append(nusmv_prop)

                # 2. Generate a chain of refinements using the enhanced logic
                prop_for_refinement = current_prop
                for _ in range(num_refinements):

                    # I should call abstract property. 
                    prop_for_refinement = self._abstract_property(prop_for_refinement, class_index=i)
                    trans_refined_prop = translator(prop_for_refinement)

                    class_properties.append({"custom": prop_for_refinement, "translated": trans_refined_prop})
                    all_props["custom"].append(prop_for_refinement)
                    all_props["translated"].append(trans_refined_prop)

            properties_by_class[i] = class_properties

        return properties_by_class, all_props, base_props