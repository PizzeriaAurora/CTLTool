 # Now, update the simulate function to use this new structure
    def simulate(self, other: 'ABTA') -> Tuple[bool, int]:
        if other == self:
            return True,0
        visited = set()
        queue = deque([(self.initial_state, other.initial_state)])
        visited.add((self.initial_state, other.initial_state))
        
        # Heuristic: if initial state is from EF or EG, it's existential.
        is_existential = self.modality == 'existential' and other.modality =="existential"
        while queue:
            qA, qB = queue.popleft()

            if qB in other.accepting and qA not in self.accepting:
                # This acceptance check is more subtle now. For EF, both are accepting.
                # It's more critical for mixing types like EF vs AG. Let's keep it.
                return False, len(visited)

            deltaA_expr = self.transitions.get(qA)
            deltaB_expr = other.transitions.get(qB)
            if not deltaA_expr or not deltaB_expr:
                continue

            dnf_A = self.get_dnf(deltaA_expr)
            dnf_B = other.get_dnf(deltaB_expr)
            

            B_dnf, A_dnf = (dnf_B, dnf_A) if not is_existential else (dnf_A, dnf_B)
            
            # For every move in the outer loop...
            for atoms_outer, next_outer in B_dnf:
                # ...we must find a matching move in the inner loop.
                found_match = False
                for atoms_inner, next_inner in A_dnf:
                    dirs_outer = {d for d, _ in next_outer}
                    dirs_inner = {d for d, _ in next_inner}
                    # --- Matching condition also depends on type ---
                    if not is_existential:
                        # Universal: A (inner) must provide everything B (outer) needs
                        # atoms_B ⊆ atoms_A
                        atomic_match = self.check_subset(atoms_outer, atoms_inner)#atoms_outer.issubset(atoms_inner)
                        temporal_match = dirs_outer.issubset(dirs_inner)
                    else:
                        # Existential: A (outer) must only provide what B (inner) accepts
                        # atoms_A ⊆ atoms_B
                        atomic_match = self.check_subset(atoms_inner, atoms_outer)
                        temporal_match = dirs_inner.issubset(dirs_outer)
                    

                    if atomic_match and temporal_match:
                        # Match is found, queue up the next states
                        for dir_o, state_o in next_outer:
                            # Find the corresponding state
                           
                            state_i = next((s for d, s in next_inner if d == dir_o),None)
                            if state_i==None:
                                return False, len(visited)
                            
                            
                            # The pair is always (A, B)
                            pair = (state_i, state_o) if not is_existential else (state_o, state_i)
                            
                            if pair not in visited:
                                visited.add(pair)
                                queue.append(pair)
                        
                        found_match = True
                        break # Found a match, move to the next outer move

                if not found_match:
                    # No matching move was found in the inner DNF. Simulation fails.
                    return False, len(visited)
        
        return True, len(visited)
    
    def _find_delayed_match(self,
                        automaton_to_match: 'ABTA',
                        move_to_match: Move,
                        start_state: State,
                        acceptance_required: bool) -> Union[Tuple[State, Move], None]:

        atoms_outer, next_outer = move_to_match
        dirs_outer = {d for d, _ in next_outer}

        # Queue for BFS: (state_to_visit, path_to_it)
        queue = deque([(start_state, [start_state])])
        visited_search = {start_state}

        while queue:
            current_inner_state, current_path = queue.popleft()
            
            inner_dnf = automaton_to_match.get_dnf(automaton_to_match.transitions[current_inner_state])
            for inner_move in inner_dnf:
                atoms_inner, next_inner = inner_move
                dirs_inner = {d for d, _ in next_inner}

                atomic_match = check_semantic_subset_z3(atoms_outer, atoms_inner)
                temporal_match = dirs_outer.issubset(dirs_inner)

                if atomic_match and temporal_match:
                    # Potential match. Check acceptance.
                    is_run_accepting = (not acceptance_required) or \
                                    any(s in automaton_to_match.accepting for s in current_path)
                    if is_run_accepting:
                        print(current_inner_state,inner_move)
                        return (current_inner_state, inner_move) # SUCCESS

 
            for inner_move in inner_dnf:
                _, next_inner = inner_move
                for _, next_state in next_inner:
                    if next_state not in visited_search:
                        visited_search.add(next_state)
                        queue.append((next_state, current_path + [next_state]))

        return None # Failure
    def simulate_delayed(self, other: 'ABTA') -> Tuple[bool, int]:
        # Checks for L(self) ⊆ L(other) using delayed simulation.
        # `self` is the "outer" automaton (∀ moves).
        # `other` is the "inner" automaton (∃ matching delayed run).
        
        
        if other == self:
            return True, 0

        # The simulation relation R ⊆ States_self x States_other
        simulation_relation = set()
        
        # The queue for the main algorithm checker
        queue = deque([(self.initial_state, other.initial_state)])
        visited_pairs = {(self.initial_state, other.initial_state)}

        while queue:
            q_outer, q_inner_start = queue.popleft()

            # Add the pair we are processing to the simulation relation
            simulation_relation.add((q_outer, q_inner_start))

            outer_is_accepting = q_outer in self.accepting

            outer_dnf = self.get_dnf(self.transitions[q_outer])

            # For every move of the outer automaton...
            for outer_move in outer_dnf:
                # ... we must find a delayed match in the inner automaton.
                
                match_result = self._find_delayed_match(
                    automaton_to_match=other,
                    move_to_match=outer_move,
                    start_state=q_inner_start,
                    acceptance_required=outer_is_accepting 
                )
                print("Martch result",match_result)
                if match_result is None:
                    # No delayed match could be found for this outer_move. Simulation fails.
                    return False, len(visited_pairs)
                
                # A match was found! Queue the successor states.
                matching_inner_state, matching_inner_move = match_result
                _, next_outer = outer_move
                _, next_inner = matching_inner_move

                # Create a lookup map for the successful inner move
                inner_next_map = {d: s for d, s in next_inner}
                for dir_outer, state_outer in next_outer:
                    state_inner = inner_next_map[dir_outer]
                    
                    pair = (state_outer, state_inner)
                    if pair not in visited_pairs:
                        visited_pairs.add(pair)
                        queue.append(pair)
        
        return True, len(visited_pairs)