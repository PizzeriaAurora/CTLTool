import random
from typing import List, Dict, Any, Set, Tuple

class ModelGenerator:
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
        if num_transitions < num_states:
            raise ValueError(
                f"Number of transitions ({num_transitions}) must be at least the "
                f"number of states ({num_states}) to avoid deadlocks."
            )
        # The maximum number of unique transitions is num_states * num_states
        if num_transitions > num_states * num_states:
            raise ValueError(
                f"Number of transitions ({num_transitions}) cannot exceed the "
                f"total possible transitions ({num_states * num_states})."
            )

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

# --- Example Usage ---
if __name__ == "__main__":
    try:
        # Generate a model with 5 states, 8 transitions, and 3 atomic props
        generator = ModelGenerator(
            num_states=5,
            num_transitions=8,
            atomic_props=["p", "q", "r"]
        )

        # Get the NuSMV code as a string
        nusmv_code = generator.to_nusmv_model()

        # Print the generated model
        print("--- Generated NuSMV Model ---")
        print(nusmv_code)
        
        # You can save this to a file:
        # with open("random_model.smv", "w") as f:
        #     f.write(nusmv_code)
        # print("\nModel saved to random_model.smv")

    except ValueError as e:
        print(f"Error generating model: {e}")