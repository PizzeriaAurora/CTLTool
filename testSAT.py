import subprocess
from RefCheck.Parser import PARSER, CTLSATTRANS

# Parse and transform both formulas
tree1 = PARSER.parse("AF (p & !q)")
transformed1 = CTLSATTRANS.transform(tree1)
print("Formula A:", transformed1)

tree2 = PARSER.parse("AF (p)")
transformed2 = CTLSATTRANS.transform(tree2)
print("Formula B:", transformed2)

# Build the implication test: ¬(A ∧ ¬B)
implication_test_formula = f"~({transformed1}^~{transformed2})"

# Run ctl-sat
result = subprocess.run(
    ["./ctl-sat", implication_test_formula],
    capture_output=True,
    text=False
)

