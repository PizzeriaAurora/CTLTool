# ==============================================================================
#  IMPORTS
# ==============================================================================
from .Property import CTLProperty
from typing import List, Dict, Set, Tuple, Optional, Iterable
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Local import for the Alternating Büchi Tree Automaton, which is core to the semantic check.
from .abta import ABTA


# ==============================================================================
#  WORKER FUNCTION FOR PARALLEL PROCESSING
# ==============================================================================

def analyze_refinement_class(index: int, class_props: List[CTLProperty], use_syn: bool) -> Tuple[int, nx.DiGraph]:
        """
        Analyzes the refinement relationships for a single class of properties.
        This function is designed to be executed in a separate process to enable parallel computation.

        Args:
            index: The index of the equivalence class, used for tracking.
            class_props: A list of CTLProperty objects belonging to this class.
            use_syn: A boolean flag indicating whether to use syntactic checks (if available)
                    or the full semantic check.

        Returns:
            A tuple containing the class index and the resulting refinement graph (nx.DiGraph).
        """
        # Initialize a directed graph to store refinement relationships.
        G = nx.DiGraph()    
         # Create a mapping from the string representation of a property to the object itself
        # for efficient lookups within the loop.
        prop_map = {str(p): p for p in class_props}
        prop_strings = list(prop_map.keys())
        G.add_nodes_from(prop_strings)
        prop_pairs = list(itertools.combinations(prop_strings, 2))
        # Iterate through each pair with a progress bar for monitoring.
        for p_i_str, p_j_str in tqdm(prop_pairs, total=len(prop_pairs), desc=f"Class {index+1} Pairs", leave=False):
            if p_i_str==p_j_str:
                    continue
            p_i = prop_map[p_i_str]
            p_j = prop_map[p_j_str]
            
            
             # Check for refinement: p_i refines p_j means L(p_i) ⊆ L(p_j).
             # This is represented as a directed edge from p_i to p_j.
            if p_i.refines(p_j, use_syn):
                G.add_edge(p_i_str, p_j_str)
            elif p_j.refines(p_i, use_syn):
                    G.add_edge(p_j_str, p_i_str)
        # The function returns the original index and the constructed graph.
        return index, G


def analyze_refinement_class_otherDirection(index: int, class_props: List[CTLProperty], use_syn: bool) -> Tuple[int, nx.DiGraph]:
        """
        Analyzes the refinement relationships for a single class of properties.
        This function is designed to be executed in a separate process to enable parallel computation.

        Args:
            index: The index of the equivalence class, used for tracking.
            class_props: A list of CTLProperty objects belonging to this class.
            use_syn: A boolean flag indicating whether to use syntactic checks (if available)
                    or the full semantic check.

        Returns:
            A tuple containing the class index and the resulting refinement graph (nx.DiGraph).
        """
        # Initialize a directed graph to store refinement relationships.
        G = nx.DiGraph()    
         # Create a mapping from the string representation of a property to the object itself
        # for efficient lookups within the loop.
        prop_map = {str(p): p for p in class_props}
        prop_strings = list(prop_map.keys())
        G.add_nodes_from(prop_strings)
        prop_pairs = list(itertools.combinations(prop_strings, 2))
        # Iterate through each pair with a progress bar for monitoring.
        for p_i_str, p_j_str in tqdm(prop_pairs, total=len(prop_pairs), desc=f"Class {index+1} Pairs", leave=False):
            if p_i_str==p_j_str:
                    continue
            p_i = prop_map[p_i_str]
            p_j = prop_map[p_j_str]
            
            
             # Check for refinement: p_i refines p_j means L(p_i) ⊆ L(p_j).
             # This is represented as a directed edge from p_i to p_j.
            #if p_i.refines(p_j, use_syn):
            #    G.add_edge(p_i_str, p_j_str)
            if p_j.refines(p_i, use_syn):
                    G.add_edge(p_j_str, p_i_str)
        # The function returns the original index and the constructed graph.
        return index, G






# ==============================================================================
#  DATA STRUCTURES
# ==============================================================================
class UnionFind:
    """
    An implementation of the Union-Find (or Disjoint Set Union) data structure.
    It is used to efficiently track and merge sets of elements, which is ideal for
    determining equivalence classes based on shared features.
    This implementation includes path compression and union-by-size optimizations.
    """
    def __init__(self, n: int):
        """
        Initializes the Union-Find structure for `n` elements.

        Args:
            n: The total number of elements, initially in `n` disjoint sets.
        """
        # self.p[i] stores the parent of element i.
        self.p = list(range(n))
        # self.sz[i] stores the size of the set containing element i.
        self.sz = [1] * n

    def find(self, x: int) -> int:
        """
        Finds the representative (root) of the set containing element `x`.
        Implements path compression for optimization.

        Args:
            x: The element to find.

        Returns:
            The root of the set containing `x`.
        """
        # Traverse up to the root.
        while self.p[x] != x:
            # Path compression: set the parent of x to its grandparent.
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        """
        Merges the sets containing elements `a` and `b`.
        Implements union-by-size optimization.

        Args:
            a: An element in the first set.
            b: An element in the second set.
        """
        # Find the roots of the sets for a and b.
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Union-by-size: attach the smaller tree to the root of the larger tree.
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.sz[ra] += self.sz[rb]



# ==============================================================================
#  STANDALONE UTILITY FUNCTION
# ==============================================================================


def build_equiv_classes(forms: List[CTLProperty]) -> Dict[int, List[int]]:
    """
    Groups a list of properties into equivalence classes based on shared atomic literals.
    Two properties are in the same class if they share at least one atom.

    Args:
        forms: A list of CTLProperty objects.

    Returns:
        A dictionary mapping the root of each equivalence class to a list of
        indices of the properties belonging to that class.
    """
    uf = UnionFind(len(forms))
    lit_map: Dict[str, List[int]] = {}
    # Map each atomic literal to the list of properties that contain it.
    for idx, f in enumerate(forms):
        for lit in f.atoms():
            lit_map.setdefault(lit, []).append(idx)
    for lst in lit_map.values():
        for a, b in itertools.combinations(lst, 2):
            uf.union(a, b)
    classes: Dict[int, List[int]] = {}
    for i in range(len(forms)):
        classes.setdefault(uf.find(i), []).append(i)
    return classes

# ==============================================================================
#  MAIN ANALYZER CLASS
# ==============================================================================
class Analyzer:
    """
    Orchestrates the analysis of TCTL properties. It handles loading properties,
    grouping them into equivalence classes, analyzing refinement relationships
    within each class, and generating reports and visualizations.
    """
    def __init__(self, prop_strings: List[str]):
        """
        Initializes the analyzer with a list of property strings.

        Args:
            prop_strings: A list of raw TCTL formula strings to be analyzed.
        """
        self.asts =[]
        self.not_possi = []# Stores properties that failed to parse.
        for idx,s in enumerate(prop_strings):
            #try:
                self.asts += [CTLProperty(s)]
            #except Exception as e:
            #    print(e)
            #    print(s)
            #    self.not_possi +=[(s,str(e))]
        self.classes = {}
        self.graphs = {}

    @classmethod
    def load_from_file(cls, filepath: str) -> 'Analyzer':
        """
        Creates an Analyzer instance by loading property strings from a file.
        Each line in the file is treated as a property. Lines starting with '#'
        and empty lines are ignored.

        Args:
            filepath: The path to the file containing the properties.

        Returns:
            A new instance of the Analyzer class.
        """
        try:
            props =[]
            with open(filepath) as fh:
                for l in fh:
                    if l.strip() and not l.strip().startswith("#"):
                        props += [l.split("#", 1)[0].strip() ]
            print(f"Loaded {len(props)} properties")
            return cls(props)
        except FileNotFoundError:
            print(f"Error: File not found at '{filepath}'")
            return cls([]) # Return an empty analyzer

    def build_equivalence_classes(self):
        """
        Groups the loaded properties into equivalence classes. Two properties are in
        the same class if they share at least one common atomic proposition. This
        is done because properties with no shared atoms cannot have a refinement
        relationship.
        """
        print("== Building Classes==")
        n = len(self.asts)
        if n == 0:
            return
            
        uf = UnionFind(n)
        lit_map: Dict[str, List[int]] = {}
        for idx, f in enumerate(self.asts):
            for lit in f.atoms():
                lit_map.setdefault(lit, []).append(idx)
        
        for lst in lit_map.values():
            for i in range(len(lst) - 1):
                uf.union(lst[i], lst[i+1])
                
        classes_by_root: Dict[int, List[CTLProperty ]] = {}
        for i in range(n):
            root = uf.find(i)
            if root not in classes_by_root:
                classes_by_root[root] = []
            classes_by_root[root].append(self.asts[i])
        
        # Store as a list of lists, which is easier to iterate over
        self.classes = list(classes_by_root.values())

    

    def analyze_refinements_opt(self, use_syn=False):
        """
        Analyzes refinement relationships in parallel for each equivalence class.
        It submits each class to a separate process for analysis, which can
        significantly speed up computation on multi-core systems.
        """
        if not self.classes:
            print("Warning: Equivalence classes have not been built. Call build_equivalence_classes() first.")
            return

        print("== Analyzing refinements (Parallel) ==")
        futures = []
        self.graphs = {}

        with ProcessPoolExecutor() as executor:
            for i, class_props in enumerate(self.classes):
                futures.append(executor.submit(analyze_refinement_class, i, class_props, use_syn))
                #futures.append(executor.submit(analyze_refinement_class_otherDirection, i+len(self.classes)+1, class_props, use_syn))

            for future in tqdm(as_completed(futures), total=len(futures), desc="Classes Done", leave=False):
                index, graph = future.result()
                self.graphs[index] = graph

    def analyze_refinements(self, use_syn = False):
        """
        Analyzes refinement relationships sequentially for each equivalence class.
        It constructs a directed graph where an edge from P1 to P2 means P1 refines P2.
        """
        if not self.classes:
            print("Warning: Equivalence classes have not been built. Call build_equivalence_classes() first.")
            return

        print("== Analyzing refinements==")
        for i, class_props in tqdm(enumerate(self.classes), total=len(self.classes), desc="Analyzing Classes"):
            G = nx.DiGraph()
            
            prop_map = {str(p): p for p in class_props}
            prop_strings = list(prop_map.keys())

            G.add_nodes_from(prop_strings)

            # Add self-loops for every property, as every property refines itself.
            #for p_str in prop_strings:
            #    G.add_edge(p_str, p_str)

            prop_pairs = list(itertools.combinations(prop_strings, 2))
            
            for p_i_str, p_j_str in tqdm(prop_pairs, total=len(prop_pairs), desc=f"Class {i+1} Pairs", leave=False):
                p_i = prop_map[p_i_str]
                p_j = prop_map[p_j_str]
                if p_i_str==p_j_str:
                    continue

    
                
                # Check 1: Does p_i refine p_j?
                if p_i.refines(p_j, use_syn):
                    G.add_edge(p_i_str, p_j_str)
                # Check 1: Does p_i refine p_j?
                elif p_j.refines(p_i, use_syn):
                    G.add_edge(p_j_str, p_i_str)


            
            self.graphs[i] = G
    def get_required_properties(self) -> List[str]:
        """
        Determines a minimal set of properties that must be verified. This set includes
        all "root" properties (those not refined by any other) and any other properties
        that are not reachable from a root (e.g., properties in a separate cycle or
        disconnected component). Verifying this set implies the verification of all
        other properties in the class.

        Returns:
            A list of property strings that form the minimal required verification set.
        """
        required = self.get_synthesis()
        result = set()

        for i, req_set in required.items():
            G = self.graphs[i]
            roots = set(req_set['roots'])

            # The roots are always required as they are the strongest properties.
            result.update(roots)

            # For all other nodes, check if they are not reachable from any root
            for node in G.nodes:
                if node in roots:
                    continue  # already included
                # If no root has a path to this node, it's not implied → must verify it
                if not any(nx.has_path(G, root, node) for root in roots):
                    result.add(node)

        return list(result)

    def get_synthesis(self) -> Dict[int, Dict[str, list]]:
        """
        Analyzes the structure of each refinement graph to identify key properties.
        This includes "roots" (the strongest, most general properties) and "cycles"
        (sets of mutually refining, i.e., equivalent, properties).

        Returns:
            A dictionary where each key is a class index and the value is another
            dictionary containing a list of 'roots' and a list of 'cycles'.
        """
        if not self.graphs:
            print("Warning: Refinement analysis has not been run. Call analyze_refinements() first.")
            return {}

        required_sets = {}
        print("Getting Synthesis")
        for i, G in self.graphs.items():
            class_num = i 
            
            # 1. Roots
            root_properties = [node for node, in_degree in G.in_degree() if in_degree == 0]

            # 2. Cycles via strongly connected components
            sccs = list(nx.strongly_connected_components(G))
            cycles = [list(scc) for scc in sccs if len(scc) > 1]

            required_sets[class_num] = {
                'roots': root_properties,
                'cycles': cycles
            }
        #for i, G in self.graphs.items():
        #    class_num = i 
        #    
        #    # 1. Find properties with no incoming edges (in-degree == 0)
        #    # These are the strongest properties that are not refined by others.
        #    root_properties = [node for node, in_degree in G.in_degree() if in_degree == 0]
#
        #    # 2. Find cycles in the graph
        #    try:
        #        print("Getting Cycles")
        #        # nx.simple_cycles finds all fundamental cycles in a directed graph
        #        cycles = list(nx.simple_cycles(G))
        #    except Exception as e:
        #        print(f"Could not detect cycles for class {class_num+1}: {e}")
        #        cycles = []
#
        #    # 3. Store the results for the current class
        #    required_sets[class_num] = {
        #        'roots': root_properties,
        #        'cycles': cycles
        #    }
#
        return required_sets
    def print_required_props(self):
        required_properties = self.determine_required_properties()
        print("\n--- Minimal Set of Required Properties ---")
        for class_num, req_set in required_properties.items():
            print(f"\nClass {class_num}:")
            if req_set['roots']:
                print("  Strongest Properties (Roots):")
                for prop in req_set['roots']:
                    print(f"    - {prop}")
            
            if req_set['cycles']:
                print("  Found refinement cycles (these are logically equivalent):")
                for cycle in req_set['cycles']:
                    print(f"    - Cycle: {' <=> '.join(cycle)}")
            
            if not req_set['roots'] and not req_set['cycles']:
                print("  No specific root properties or cycles found. All properties might be equivalent or unrelated.")

    def write_report_and_graphs(self, 
                                filename="refinement_analysis.txt",
                                output_folder = "output",
                                graphs_name = "refinement_class_",
                                head_folder = ""
                                ):
        """Generates a text report and graph images for the analysis."""
        if not self.graphs:
            print("Warning: Refinement analysis has not been run. Call analyze_refinements() first.")
            return
        if head_folder !="":
            if not os.path.exists(head_folder):
                os.makedirs(head_folder)
            output_folder = head_folder + "/" + output_folder
            
        if not os.path.exists(f"{output_folder}_{0:03d}"):
            print(f"Warning: Output folder {output_folder} did not exist, I am creating a new one")
            output_folder = f"{output_folder}_{0:03d}"
            os.makedirs(output_folder)
        else:
            output_folder_new = output_folder # Start with the base name
            counter = 1
            while True:
                # Create a new name with a formatted number (e.g., _001, _002)
                output_folder_new = f"{output_folder}_{counter:03d}"
                if not os.path.exists(output_folder_new):
                    # Found an available name, break the loop
                    break
                counter += 1
            os.makedirs(output_folder_new)
            print(f"Next available folder name is: {output_folder_new}")
            output_folder = output_folder_new
        report_filename = output_folder + "/" + filename
        with open(report_filename, "w", encoding="utf-8") as f_out:
            print(f"Writing analysis to {report_filename} and generating graph images...")
            f_out.write("TCTL Refinement Analysis\n========================\n\n")

            for i, class_props in enumerate(self.classes):
                class_num = i + 1
                f_out.write(f"--- Equivalence Class {class_num} ---\n")
                f_out.write("Properties in this class:\n")
                for p in class_props:
                    f_out.write(f"  - {p}\n")
                f_out.write("\n")
                f_out.write("\n Not possible:")
                for p in self.not_possi:
                    f_out.write(f"  - {p[0]}\n")
                    f_out.write(f"Reason:{p[1]} ")


                G = self.graphs[i]
                TR = G
                try:
                    if nx.is_directed_acyclic_graph(G):
                        TR = nx.transitive_reduction(G)
                except Exception:
                    pass # Keep original graph if reduction fails

                f_out.write("Found Refinements (⇒ means 'refines'):\n")
                if not TR.edges():
                    f_out.write("  - No non-trivial refinements found in this class.\n")
                else:
                    for p_i_str, p_j_str in sorted(list(TR.edges())):
                        f_out.write(f"  - {p_i_str}  ⇒  {p_j_str}\n")
                
                # Draw and save the graph
                graph_filename = f"{output_folder}/{graphs_name}_{class_num}.png"
                f_out.write(f"\nRefinement graph saved to: {graph_filename}\n\n\n")
                
                plt.figure(figsize=(12, 8))
                pos = nx.spring_layout(TR, k=0.9, iterations=50, seed=42)
                labels = {n: '\n'.join(n[i:i+25] for i in range(0, len(n), 25)) for n in TR.nodes()}
                nx.draw(TR, pos, labels=labels, with_labels=True, node_size=3000, node_color="skyblue", 
                        font_size=8, arrows=True, arrowstyle='-|>', arrowsize=20, edge_color='gray')
                plt.title(f"Refinement Graph - Class {class_num}", size=15)
                plt.savefig(graph_filename, bbox_inches='tight', dpi=150)
                plt.close()
        
        print("Analysis complete.")