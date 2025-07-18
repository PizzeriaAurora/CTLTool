import os
import re

# --- Configuration ---
# Define source and destination directories for clarity and easy modification.
SOURCE_DIR = "./benchmarks/rawParallelRers2019Ctl"
DEST_DIR = "benchmarks/ParallelRers2019Ctl"
PROPERTIES_FILENAME = "prop.txt"

def convert_rers_formula(raw_formula: str) -> str:
    """
    Converts a RERS-style formula into a simplified grammar.

    Transformations:
    - Consecutive <...><...><none>true  →  (a & b & none)
    - Consecutive [..][..][none]false  →  ~(a & b & none)
    - => becomes ->
    """
    formula = raw_formula

    # Step 1: Handle grouped <> or [] followed by true
    def group_true(match):
        props = re.findall(r'[<\[]\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*[>\]]', match.group(0))
        return f"({' & '.join(props)})"

    formula = re.sub(r'(?:[<\[]\s*[a-zA-Z_][a-zA-Z0-9_]*\s*[>\]])+\s*true', group_true, formula)

    # Step 2: Handle grouped <> or [] followed by false
    def group_false(match):
        props = re.findall(r'[<\[]\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*[>\]]', match.group(0))
        return f"!({' & '.join(props)})"
    
    # Step 4: Replace adjacent <a><b><c> or [a][b]... → (a & b & c)
    def combine_adjacent_props(match):
        props = re.findall(r'[<\[]([a-zA-Z_][a-zA-Z0-9_]*)[>\]]', match.group(0))
        return '(' + ' & '.join(props) + ')'

    formula = re.sub(r'(?:[<\[][a-zA-Z_][a-zA-Z0-9_]*[>\]]){2,}', combine_adjacent_props, formula)


    formula = re.sub(r'(?:[<\[]\s*[a-zA-Z_][a-zA-Z0-9_]*\s*[>\]])+\s*false', group_false, formula)
    formula = re.sub(r'(?:[<\[]\s*[a-zA-Z_][a-zA-Z0-9_]*\s*[>\]])+\s*none', group_false, formula)



    # Step 3: Replace => with ->
    formula = formula.replace('=>', '->')

    return formula

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
                converted = convert_rers_formula(line)
                converted_formulas.append(converted)
        
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
            # Use 'with' for safe file handling (reading and writing).
            with open(source_filepath, 'r') as f_in, open(dest_filepath, 'w') as f_out:

                for prop_line in properties_to_append:
                    f_out.write(f"{prop_line}\n")
                    
        except Exception as e:
            print(f"  - Failed to process {filename}: {e}")

    print("\nProcessing complete.")


# This standard Python construct ensures the code runs only when the script is executed directly.
if __name__ == "__main__":
    process_files()