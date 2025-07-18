
import random
import subprocess
import time
import argparse
import os
import sys
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np

from RefCheck.Analyzer import Analyzer
from RefCheck.Parser import  PARSER, NUSMV,CTLSATTRANS
from RefCheck.propgen import PropertyGenerator, to_sat_syntax
from RefCheck.modegen import HybridModelGenerator
import itertools

from tqdm import tqdm

random.seed(476)
#Input file position - 2
CONTINUE_FROM = 0
PRINT = False
def flatten(xss):
    if len(xss) ==0:
        return []
    if not isinstance(xss[0], list):
        return xss
    return sum(xss, [])


def run_sat(properties: List[str], satpath: str):
    prop_pairs = list(itertools.combinations(properties, 2))
    start_time = time.time()
    for f1, f2 in prop_pairs:
        # Build the implication test: ¬(A ∧ ¬B)
        implication_test_formula = f"~({f1}^~{f2})"
        # Run ctl-sat
        result = subprocess.run(
            [satpath, implication_test_formula],
            capture_output=True,
            text=True
        )
    return time.time() - start_time


def run_once(idx,
            base_props, 
            refined_props,
            num_classes,
            atomic_props,
            saving_dir,
            satpath: str = "/Users/brdlcu/NuSMV/bin/NuSMV"):
    total_props = base_props + refined_props
    tqdm.write(f"--- CTL Refinement Benchmark {idx}---")
    tqdm.write(f"Properties: {total_props*num_classes} total ({base_props} base, {refined_props} refined)")
    tqdm.write("-" * 30)
    prop_dir = f"{saving_dir}/properties"
    try:
        os.mkdir(prop_dir)
    except:
        pass
    atomic_props = [f"p_{i}" for i in range(atomic_props)]
    tqdm.write(f"\n[{idx}:1/4] Generating properties...")
    prop_gen = PropertyGenerator(atomic_props, num_equivalence_classes=num_classes)
    _,all_properties, _ = prop_gen.generate(base_props, refined_props,to_sat_syntax)
    props_path = f"{prop_dir}/benchmark_properties_{idx}.txt"
    with open(props_path, 'w') as f:
        f.write("\n".join(all_properties["custom"]))
    tqdm.write(f"  -> {len(all_properties["custom"])} properties saved to '{props_path}'")

    tqdm.write(f"[{idx}:2/4] Running Scenario B: Brute-force (checking all properties)...")
    time_brute_force = run_sat( all_properties["translated"],satpath)
    tqdm.write(f"  -> Brute-force SAT time: {time_brute_force:.4f} seconds")
    tqdm.write(f"\n[{idx}:3/4] Running Scenario A: Refinement Analysis")
    analyzer = Analyzer.load_from_file(props_path)
    analysis_start_time = time.time()
    analyzer.build_equivalence_classes()
    analyzer.analyze_refinements_opt(True)
    analysis_time = time.time() - analysis_start_time
    bench_dir = f"{saving_dir}/refinmenet_results"
    try:
        os.mkdir(bench_dir)
    except:
        pass

    analyzer.write_report_and_graphs(output_folder="benchmark",
                                     head_folder=bench_dir
                                       )
    required_properties = analyzer.get_required_properties()
    
    tqdm.write(f"  -> Analysis found {len(required_properties)} required properties.")
    tqdm.write(f"  -> Analysis time: {analysis_time:.4f} seconds")

    required_properties_sat=[to_sat_syntax(p[1:-2]) for p in required_properties]
    time_mc_refined = 0#run_sat( required_properties_sat,satpath)
    #tqdm.write(f"  -> Model checking time for refined set: {time_mc_refined:.4f} seconds")
#
    total_time_with_analysis = analysis_time + time_mc_refined
    tqdm.write(f"  -> Total time with analysis: {total_time_with_analysis:.4f} seconds")
#
    ## 4. Report Results
    tqdm.write("\n[4/4] --- BENCHMARK RESULTS ---")
    tqdm.write(f"Time for Brute-Force (check all {len(all_properties["custom"])} props):      {time_brute_force:.4f} s")
    tqdm.write(f"Time with the refinement Method (analyze + check {len(required_properties)} props): {total_time_with_analysis:.4f} s")
    
    tqdm.write("-" * 30)
    diff = 0
    if total_time_with_analysis < time_brute_force:
        speedup = time_brute_force / total_time_with_analysis if total_time_with_analysis > 0 else float('inf')
        tqdm.write(f"CONCLUSION: The refinement method is FASTER by {time_brute_force - total_time_with_analysis:.4f} seconds ({speedup:.2f}x speedup).")
        diff=speedup
        
    else:
        slowdown = total_time_with_analysis / time_brute_force if time_brute_force > 0 else float('inf')
        tqdm.write(f"CONCLUSION: Brute-force is FASTER by {total_time_with_analysis - time_brute_force:.4f} seconds ({slowdown:.2f}x slower).")
        diff = -slowdown
    

    return time_brute_force, total_time_with_analysis, analysis_time,time_mc_refined, diff


def main():
    parser = argparse.ArgumentParser(description="Benchmark CTL property refinement analysis against brute-force model checking.")
    parser.add_argument("--satpath", type=str, default="./extern/ctl-sat", help="CTLSAT Path")
    parser.add_argument("--input", type=str, default="input_files/input_benchmarks_sat.csv", help="Input file.")
    parser.add_argument("--output", type=str, default="resultBenchmark.csv", help="Result output file.")
    parser.add_argument("--output_dir", type=str, default="result/SAT", help="Benchmark results folder.")
    args = parser.parse_args()



    satpath = args.satpath


    input_args_list = ["states", "transitions","atomic_props","base_props","refined_props","num_classes","chain_states"
                 "bit_width"
                 ]
    
    output_args_list = ["time_brute_force", "total_time_with_analysis", "analysis_time","time_mc_refined","diff" ]
    output_folder = args.output_dir
    if not os.path.exists(output_folder):
        tqdm.write(f"Warning: Output folder {output_folder} did not exist, I am creating a new one")
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
        tqdm.write(f"Next available folder name is: {output_folder_new}")
        output_folder = output_folder_new
    
    
   
    input_df = pd.read_csv(args.input)
    df = pd.DataFrame(columns=output_args_list)
    pbar = tqdm(input_df.iterrows(), total=len(list(input_df.iterrows())), position=0)
    for idx, row in pbar:
        if idx<CONTINUE_FROM:
            continue
        atomic_props = row['atomic_props']
        base_props = row['base_props']
        refined_props = row['refined_props']
        num_classes = row['num_classes']
    
        # Print the variables for this row
        tqdm.write(f"Benchmark {idx}:")
        tqdm.write(f"  atomic_props: {atomic_props}")
        tqdm.write(f"  base_props: {base_props}")
        tqdm.write(f"  refined_props: {refined_props}")
        tqdm.write(f"  num_classes: {num_classes}")
        tqdm.write("-" * 40)
    
        output_time = run_once(idx,
                base_props, 
                refined_props,
                num_classes,
                atomic_props,
                output_folder,
                satpath
                )
        df.loc[idx] = output_time
        pbar.set_description(f"Prev. Run {output_time[-1]:.1f}% ")

        df.to_csv(output_folder + "/"+args.output, index=False)
    print(f"Saved results in {output_folder+ "/"+args.output}")
    



if __name__ == '__main__':
    main()