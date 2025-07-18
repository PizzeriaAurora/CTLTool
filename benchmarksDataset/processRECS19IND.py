import os
import re

# --- Configuration ---
# Define source and destination directories for clarity and easy modification.
SOURCE_DIR = "./rawRers2019Ind"
DEST_DIR = "Rers2019Ind"
PROPERTIES_FILENAME = "prop.txt"



from lark import Lark, Transformer, v_args

rers_grammar = r"""
start: expr
?expr: expr "|" expr   -> or_
     | expr "&" expr   -> and_
     | expr "=>" expr  -> imply
     | "AG" "(" expr ")" -> ag
     | "EG" "(" expr ")" -> eg
     | "AF" "(" expr ")" -> af
     | "EF" "(" expr ")" -> ef
     | "(" expr ")"
     | labeled

// A labeled expression is a list of labels followed by an OPTIONAL atom.
labeled: labels (atom)?

// The labels part is one or more <CNAME> blocks.
labels: ("<" CNAME ">")+

atom: "true" | "false" | CNAME

%import common.CNAME
%import common.WS
%ignore WS
"""

@v_args(inline=True)
class CTLTransformer(Transformer):

    def __init__(self):
        super().__init__()
        self.time_bound_warning_issued = False
    def start(self, args):
        return args
    def or_(self, a, b):
        return f"({a} | {b})"
    def and_(self, a, b):
        return f"({a} & {b})"
    def imply(self, a, b):
        return f"({a} -> {b})"
    def ag(self, inner):
        return f"AG {inner}"
    
    def ef(self, inner):
        return f"EF {inner}"
    def eg(self, inner):
        return f"EG {inner}"
    def af(self, inner):
        return f"AF {inner}"
    def af(self, atom):
        return atom
    
    def labeled(self, labels, atoms=[]):
        children = list(reversed(labels.children))
        result = children[0]
        for label in children[1:]:
            result = f"({label} &{result})"
        result = f" {result}"
        return result


parser = Lark(rers_grammar)
trans = CTLTransformer()

def load_and_convert_rers(filepath: str) -> list[str]:
    """
    Loads a RERS problem file, cleans each line, and converts it into a 
    grammar-compatible format.
    """
    converted_formulas = []
    try:
        with open(filepath, 'r') as fh:
            # Read raw lines, remove comments and strip whitespace
            raw_lines = [l.split("#", 1)[0].strip() for l in fh if l.strip() and not l.strip().startswith("#")]
            
            # Convert each valid line
            for line in raw_lines:
                processed_line = re.sub(r'(?<=[)>])\s*(?=\()', ' & ', line)
                converted = parser.parse(processed_line)
                final_string = trans.transform(converted)
                converted_formulas.append(final_string)
        
        return converted_formulas
        
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        return []


def process_files():
    """
    Main function to process the directory.
    """
    # 1. Ensure the destination directory exists. Create it if it doesn't.
    print(f"Ensuring destination directory exists: {DEST_DIR}")
    os.makedirs(DEST_DIR, exist_ok=True)

    files = os.listdir(SOURCE_DIR)
    print(files)
    for filename in files:
        if filename == ".DS_Store":
            continue

        source_filepath = os.path.join(SOURCE_DIR, filename)
        dest_filepath = os.path.join(DEST_DIR, filename)
        properties_to_append = load_and_convert_rers(source_filepath)
        print(f"Processing {source_filepath} -> {dest_filepath}")
        try:
            with open(source_filepath, 'r') as f_in, open(dest_filepath, 'w') as f_out:
                for prop_line in properties_to_append:
                    f_out.write(f"{prop_line}\n")
        except Exception as e:
            print(f"  - Failed to process {filename}: {e}")

    print("\nProcessing complete.")



if __name__ == "__main__":
    process_files()