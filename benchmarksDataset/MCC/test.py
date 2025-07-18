import itertools



prop_strings = ["a","b", "c"]
prop_pairs = list(itertools.combinations(prop_strings, 2))
print(prop_pairs)