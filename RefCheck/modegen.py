


import random
from typing import List, Dict, Any

import random
from typing import List, Tuple, Dict, Any, Set



class SimpleModelGenerator:
    """
    Generates a random Kripke structure (FSM) with desirable properties
    and exports it to a NuSMV model file.

    This generator ensures:
    1. The transition relation is total (every state has at least one successor).
    2. The requested number of transitions is respected.
    3. All states are likely reachable from the initial state, leading to more
       interesting models for verification.
    """
    def __init__(self, num_states: int, num_transitions: int, atomic_props: List[str]):
        """
        Initializes the model generator with the specified parameters.

        Args:
            num_states: The number of states in the model.
            num_transitions: The total number of unique transitions.
            atomic_props: A list of strings for atomic proposition names.

        Raises:
            ValueError: If parameters are invalid (e.g., too few transitions
                        to ensure every state has a successor).
        """
        if num_states <= 0:
            raise ValueError("Number of states must be positive.")
        # To ensure a total relation (each state has an outgoing edge), we need
        # at least num_states transitions.
        #if num_transitions < num_states:
        #    raise ValueError(
        #        f"Number of transitions ({num_transitions}) must be at least the "
        #        f"number of states ({num_states}) to avoid deadlocks."
        #    )
        ## The maximum number of unique transitions is num_states * num_states
        #if num_transitions > num_states * num_states:
        #    raise ValueError(
        #        f"Number of transitions ({num_transitions}) cannot exceed the "
        #        f"total possible transitions ({num_states * num_states})."
        #    )
#
        self.num_states = num_states
        self.num_transitions = num_transitions
        self.atomic_props = atomic_props
        self.states = list(range(num_states))
        self.initial_state = 0  # Conventionally, the first state is initial
        self.model = self._generate()

    def _generate(self) -> Dict[str, Any]:
        """Generates the random Kripke structure."""
        transitions: Set[Tuple[int, int]] = set()

        # Step 1: Ensure every state has at least one outgoing transition.
        # This makes the transition relation total and avoids deadlocks.
        for state in self.states:
            # Add a transition from the current state to a random state.
            # This might create a self-loop, which is perfectly valid.
            dst = random.choice(self.states)
            transitions.add((state, dst))

        # Step 2: Add remaining random transitions until the desired count is reached.
        # The set `transitions` will automatically handle duplicates.
        while len(transitions) < self.num_transitions:
            src = random.choice(self.states)
            dst = random.choice(self.states)
            transitions.add((src, dst))

        # Step 3: Assign atomic propositions randomly to each state.
        # Each proposition has a 50% chance of being true in any given state.
        props_map = {state: set() for state in self.states}
        for state in self.states:
            for prop in self.atomic_props:
                if random.random() < 0.5:
                    props_map[state].add(prop)

        return {
            "states": self.states,
            "transitions": sorted(list(transitions)), # Sorted for consistent output
            "props_map": props_map,
            "initial_state": self.initial_state
        }

    def to_nusmv_model(self) -> str:
        """Exports the generated model to a string in NuSMV format."""
        lines = ["MODULE main"]
        lines.append("VAR")
        lines.append(f"\tstate: 0..{self.num_states - 1};")
        lines.append("ASSIGN")
        lines.append(f"\tinit(state) := {self.initial_state};")

        # Group transitions by source state for easier processing
        trans_map = {s: [] for s in self.states}
        for src, dst in self.model['transitions']:
            trans_map[src].append(dst)

        lines.append("\tnext(state) := case")
        for state in sorted(trans_map.keys()):
            next_states = trans_map.get(state)
            if next_states:
                # Format for single or multiple next states (nondeterministic choice)
                next_states_str = f"{{{', '.join(map(str, sorted(next_states)))}}}"
                lines.append(f"\t\tstate = {state} : {next_states_str};")
            else:
                # This case should not be reached with the new _generate method,
                # but it's good practice to handle it for robustness.
                # A state with no outgoing transitions deadlocks (stutters).
                lines.append(f"\t\tstate = {state} : {state};")
        
        # A default case is good practice but not strictly necessary if all
        # states are covered.
        lines.append("\t\tTRUE : state;")
        lines.append("\tesac;")

        # Define atomic propositions based on the state variable
        lines.append("DEFINE")
        for prop in self.atomic_props:
            # Find all states where the proposition is true
            true_states = [s for s, props in self.model['props_map'].items() if prop in props]
            if true_states:
                # Create a disjunction of conditions, e.g., (state = 1) | (state = 3)
                conds = " | ".join([f"(state = {s})" for s in true_states])
                lines.append(f"\t{prop} := {conds};")
            else:
                # If the proposition is never true, define it as FALSE
                lines.append(f"\t{prop} := FALSE;")

        return "\n".join(lines)


class ModelGenerator:
    """
    Generates a Kripke structure and exports it to a NuSMV model file.

    This generator supports multiple modes to create models that are intentionally
    difficult for BDD-based model checkers.

    Modes:
    - 'random': (Default) Generates a random Kripke structure with a specified
      number of states and transitions.
      Required args: `num_states`, `num_transitions`, `atomic_props`.

    - 'chain': Generates a model with a very large diameter (a single long cycle),
      which challenges fixpoint-based CTL checking algorithms.
      Required args: `num_states`, `atomic_props`.

    - 'counter': Generates a binary counter with many interacting boolean state
      variables. This structure is known to cause BDD variable ordering issues
      and lead to state-space explosion.
      Required args: `num_bits`, `atomic_props`.
    """
    def __init__(self, mode: str = 'random', **kwargs):
        """
        Initializes the model generator based on the specified mode.

        Args:
            mode (str): The generation mode ('random', 'chain', or 'counter').
            **kwargs: Keyword arguments specific to the chosen mode.
        """
        self.mode = mode
        self.atomic_props = kwargs.get('atomic_props')
        if not self.atomic_props or not isinstance(self.atomic_props, list):
            raise ValueError("'atomic_props' (list) is a required keyword argument.")

        if self.mode == 'random':
            self._init_random(**kwargs)
        elif self.mode == 'chain':
            self._init_chain(**kwargs)
        elif self.mode == 'counter':
            self._init_counter(**kwargs)
        else:
            raise ValueError(f"Unknown generator mode: '{self.mode}'")

    def _init_random(self, **kwargs):
        """Initializes the generator for 'random' mode."""
        self.num_states = kwargs.get('num_states')
        self.num_transitions = kwargs.get('num_transitions')
        if not all([self.num_states, self.num_transitions]):
            raise ValueError("For 'random' mode, 'num_states' and 'num_transitions' are required.")
        if self.num_states <= 0:
            raise ValueError("Number of states must be positive.")

        self.states = list(range(self.num_states))
        self.initial_state = 0
        self.model_data = self._generate_random_model()

    def _init_chain(self, **kwargs):
        """Initializes the generator for 'chain' mode."""
        self.num_states = kwargs.get('num_states')
        if not self.num_states:
            raise ValueError("For 'chain' mode, 'num_states' is required.")
        if self.num_states <= 0:
            raise ValueError("Number of states must be positive.")

        self.states = list(range(self.num_states))
        self.initial_state = 0
        self.model_data = self._generate_chain_model()

    def _init_counter(self, **kwargs):
        """Initializes the generator for 'counter' mode."""
        self.num_bits = kwargs.get('num_bits')
        if not self.num_bits:
            raise ValueError("For 'counter' mode, 'num_bits' is required.")
        if not 1 <= self.num_bits <= 128:
            raise ValueError("Number of bits must be between 1 and 128.")

        self.bit_vars = [f"b{i}" for i in range(self.num_bits)]
        # For this mode, the model is defined by logic, not an explicit graph.
        self.model_data = None

    def _generate_random_model(self) -> Dict[str, Any]:
        """Generates a random Kripke structure."""
        transitions: Set[Tuple[int, int]] = set()
        # Ensure total transition relation
        for state in self.states:
            dst = random.choice(self.states)
            transitions.add((state, dst))

        # Add remaining transitions
        while len(transitions) < self.num_transitions:
            src = random.choice(self.states)
            dst = random.choice(self.states)
            transitions.add((src, dst))

        # Assign props randomly
        props_map = {state: {p for p in self.atomic_props if random.random() < 0.5} for state in self.states}
        
        return {
            "transitions": sorted(list(transitions)),
            "props_map": props_map
        }

    def _generate_chain_model(self) -> Dict[str, Any]:
        """Generates a Kripke structure that is a single long cycle."""
        transitions = [(i, i + 1) for i in range(self.num_states - 1)]
        transitions.append((self.num_states - 1, 0)) # Complete the cycle

        # Assign props sparsely to make reachability checks meaningful
        props_map = {state: set() for state in self.states}
        for state in self.states:
            for prop in self.atomic_props:
                if random.random() < 0.05: # Low probability
                    props_map[state].add(prop)
        
        # Optionally place key props at opposite ends to create "killer" properties
        if len(self.atomic_props) >= 2:
            props_map[0].add(self.atomic_props[0])
            props_map[self.num_states // 2].add(self.atomic_props[1])

        return {
            "transitions": transitions,
            "props_map": props_map
        }

    def to_nusmv_model(self, fairness_props: List[str] = None, prop_definitions: Dict[str, str] = None) -> str:
        """
        Exports the generated model to a string in NuSMV format.

        Args:
            fairness_props (List[str], optional): A list of propositions to use as
              fairness constraints. Defaults to None.
            prop_definitions (Dict[str, str], optional): Required only for 'counter'
              mode. A dictionary mapping atomic prop names to their boolean logic
              definitions (e.g., {'p': 'b0 & !b5'}). Defaults to None.

        Returns:
            str: The complete NuSMV model as a string.
        """
        if self.mode == 'counter':
            return self._to_nusmv_counter(fairness_props, prop_definitions)
        else:
            return self._to_nusmv_state_based(fairness_props)

    def _to_nusmv_state_based(self, fairness_props: List[str]) -> str:
        """Generates NuSMV for 'random' and 'chain' modes."""
        lines = ["MODULE main"]
        lines.append("VAR")
        lines.append(f"\tstate: 0..{self.num_states - 1};")
        lines.append("ASSIGN")
        lines.append(f"\tinit(state) := {self.initial_state};")

        trans_map = {s: [] for s in self.states}
        for src, dst in self.model_data['transitions']:
            trans_map[src].append(dst)

        lines.append("\tnext(state) := case")
        for state in sorted(trans_map.keys()):
            next_states = trans_map.get(state, [state]) # Default to stutter if no trans
            next_states_str = f"{{{', '.join(map(str, sorted(next_states)))}}}" if len(next_states) > 1 else str(next_states[0])
            lines.append(f"\t\tstate = {state} : {next_states_str};")
        lines.append("\t\tTRUE : state;") # Default case
        lines.append("\tesac;")

        lines.append("DEFINE")
        for prop in self.atomic_props:
            true_states = [s for s, props in self.model_data['props_map'].items() if prop in props]
            if true_states:
                conds = " | ".join([f"(state = {s})" for s in true_states])
                lines.append(f"\t{prop} := {conds};")
            else:
                lines.append(f"\t{prop} := FALSE;")
        
        model_string = "\n".join(lines)

        if fairness_props:
            fairness_lines = [f"FAIRNESS {prop};" for prop in fairness_props]
            model_string += "\n" + "\n".join(fairness_lines)

        return model_string

    def _to_nusmv_counter(self, fairness_props: List[str], prop_definitions: Dict[str, str]) -> str:
        """Generates NuSMV for the 'counter' mode."""
        if not prop_definitions:
            raise ValueError("For 'counter' mode, 'prop_definitions' are required in to_nusmv_model().")

        lines = ["MODULE main"]
        lines.append("VAR")
        for var in self.bit_vars:
            lines.append(f"\t{var}: boolean;")

        lines.append("ASSIGN")
        for var in self.bit_vars:
            lines.append(f"\tinit({var}) := FALSE;")

        # Counter logic (next state function)
        lines.append(f"\tnext({self.bit_vars[0]}) := !{self.bit_vars[0]};")
        for i in range(1, self.num_bits):
            carry_in = " & ".join(self.bit_vars[:i])
            lines.append(f"\tnext({self.bit_vars[i]}) := {self.bit_vars[i]} != ({carry_in});")

        lines.append("DEFINE")
        for prop, definition in prop_definitions.items():
            if prop not in self.atomic_props:
                 print(f"Warning: Definition provided for property '{prop}' not in the initial atomic_props list.")
            lines.append(f"\t{prop} := {definition};")

        model_string = "\n".join(lines)
        
        if fairness_props:
            fairness_lines = [f"FAIRNESS {prop};" for prop in fairness_props]
            model_string += "\n" + "\n".join(fairness_lines)

        return model_string

class HybridModelGenerator:
    """
    Generates a hybrid "chain-of-counters" model designed to be extremely
    difficult for BDD-based model checkers.

    This model is initialized with a simple list of atomic proposition names
    and automatically generates complex definitions for them, mixing conditions
    on the high-level chain and the low-level counter.
    """
    def __init__(self, num_chain_states: int, num_bits: int, atomic_props: List[str]):
        """
        Initializes the hybrid model generator.

        Args:
            num_chain_states: The number of states in the high-level chain.
            num_bits: The number of boolean variables for the internal counter.
            atomic_props: A list of strings for the atomic proposition names.
        """
        if not num_chain_states > 1:
            raise ValueError("Number of chain states must be greater than 1.")
        if not 1 <= num_bits <= 128:
            raise ValueError("Number of bits must be between 1 and 128.")
        if not atomic_props or not isinstance(atomic_props, list):
            raise ValueError("'atomic_props' must be a non-empty list of strings.")

        self.num_chain_states = num_chain_states
        self.num_bits = num_bits
        self.atomic_props = atomic_props
        self.bit_vars = [f"b{i}" for i in range(num_bits)]

        # Generate the random trigger values needed for the chain to advance.
        self._triggers = self._generate_triggers()
        # Automatically generate and store the complex definitions for the props.
        self.prop_definitions = self._generate_prop_definitions()

    def _generate_triggers(self) -> List[str]:
        """Generates the complex boolean conditions needed to advance the chain."""
        triggers = []
        for _ in range(self.num_chain_states):
            cond_parts = []
            for i in range(self.num_bits):
                cond_parts.append(f"{self.bit_vars[i]}" if random.random() < 0.5 else f"!{self.bit_vars[i]}")
            triggers.append(" & ".join(cond_parts))
        return triggers

    def _generate_prop_definitions(self) -> Dict[str, str]:
        """
        Creates complex, random definitions for each atomic proposition,
        mixing conditions on the chain state and counter bits.
        """
        definitions = {}
        for prop in self.atomic_props:
            rand_val = random.random()

            # 70% chance: A complex definition involving both chain and bits
            if rand_val < 0.7:
                chain_cond = f"(chain_pos = {random.randint(0, self.num_chain_states - 1)})"
                
                # Create a condition on a small subset of the bits
                num_bits_in_cond = random.randint(1, min(3, self.num_bits))
                chosen_bit_indices = random.sample(range(self.num_bits), num_bits_in_cond)
                bit_conds = [
                    f"({self.bit_vars[i]} {random.choice(['&', '|'])} !{self.bit_vars[i]})" if random.random() > 0.5 else
                    (f"{self.bit_vars[i]}" if random.random() < 0.5 else f"!{self.bit_vars[i]}")
                    for i in chosen_bit_indices
                ]
                bit_part = f"({' & '.join(bit_conds)})"
                definitions[prop] = f"{chain_cond} & {bit_part}"

            # 15% chance: A simpler definition involving only the chain position
            elif rand_val < 0.85:
                chain_state = random.randint(0, self.num_chain_states - 1)
                definitions[prop] = f"(chain_pos = {chain_state})"

            # 15% chance: A definition involving only the counter bits
            else:
                num_bits_in_cond = random.randint(1, min(4, self.num_bits))
                chosen_bit_indices = random.sample(range(self.num_bits), num_bits_in_cond)
                bit_conds = [f"{self.bit_vars[i]}" if random.random() < 0.5 else f"!{self.bit_vars[i]}" for i in chosen_bit_indices]
                definitions[prop] = f"({' & '.join(bit_conds)})"
        
        return definitions

    def to_nusmv_model(self) -> str:
        """
        Exports the hybrid model to a string in NuSMV format.
        This method is now self-contained and requires no arguments.
        """
        lines = ["MODULE main"]
        lines.append("VAR")
        lines.append(f"\tchain_pos: 0..{self.num_chain_states - 1};")
        for var in self.bit_vars:
            lines.append(f"\t{var}: boolean;")

        lines.append("ASSIGN")
        lines.append(f"\tinit(chain_pos) := 0;")
        for var in self.bit_vars:
            lines.append(f"\tinit({var}) := FALSE;")

        # Counter transition logic
        lines.append(f"\tnext({self.bit_vars[0]}) := !{self.bit_vars[0]};")
        for i in range(1, self.num_bits):
            carry_in = " & ".join(self.bit_vars[:i])
            lines.append(f"\tnext({self.bit_vars[i]}) := {self.bit_vars[i]} != ({carry_in});")

        # Chain transition logic
        lines.append("\tnext(chain_pos) := case")
        for i in range(self.num_chain_states):
            current_pos_cond = f"chain_pos = {i}"
            trigger_cond = self._triggers[i]
            next_pos = (i + 1) % self.num_chain_states
            lines.append(f"\t\t({current_pos_cond}) & ({trigger_cond}) : {next_pos};")
        lines.append("\t\tTRUE : chain_pos;")
        lines.append("\tesac;")

        # Definitions for Atomic Propositions (using the pre-generated ones)
        lines.append("DEFINE")
        for prop, definition in self.prop_definitions.items():
            lines.append(f"\t{prop} := {definition};")

        return "\n".join(lines)


