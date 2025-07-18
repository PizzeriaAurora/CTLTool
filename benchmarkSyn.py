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
from RefCheck.Parser import  PARSER, NUSMV
from RefCheck.propgen import PropertyGenerator, to_nusmv_syntax
from RefCheck.modegen import HybridModelGenerator


from tqdm import tqdm

random.seed(476)
#Input file position - 2

PRINT = True
def flatten(xss):
    if len(xss) ==0:
        return []
    if not isinstance(xss[0], list):
        return xss
    return sum(xss, [])
    



# --- Benchmark Runner ---

def run_nusmv_check(model_path: str, properties: List[str], nusmv_path: str = "/Users/brdlcu/NuSMV/bin/NuSMV", i=0, not_print = True) -> float:
    """
    Runs NuSMV on a model with a list of properties and returns the execution time.
    """
    if not properties:
        return 0.0
    # Create a temporary file with the model and specs
    temp_model_path = f"./.temp_model{i}.smv"
    with open(model_path, 'r') as f_model:
        model_content = f_model.read()
    
    with open(temp_model_path, 'w') as f_temp:
        f_temp.write(model_content)
        f_temp.write("\n")
        for prop in properties:
            f_temp.write(f"CTLSPEC {prop};\n")
   
    start_time = time.time()
    try:
        subprocess.run(
            [nusmv_path,temp_model_path],
            check=True,
            capture_output=not_print,
            text=not_print
        )
    except FileNotFoundError:
        tqdm.write(f"Error: '{nusmv_path}' not found. Please ensure NuSMV is installed and in your PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        tqdm.write("NuSMV execution failed.")
        tqdm.write("stdout:", e.stdout)
        tqdm.write("stderr:", e.stderr)
        sys.exit(1)
    finally:
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
        pass    
    return time.time() - start_time



def run_once(idx,
            base_props, 
            refined_props,
            transitions,
            states,
            num_classes,
            atomic_props,
            chain_states,
            bit_width,
            saving_dir, 
            nusmv_path: str = "/Users/brdlcu/NuSMV/bin/NuSMV", id=0, not_print = True):
    total_props = base_props * (refined_props+1)
    tqdm.write(f"--- CTL Refinement Benchmark {idx}---")
    tqdm.write(f"Model: {states} states, {transitions} transitions")
    tqdm.write(f"Properties: {total_props*num_classes} total ({base_props} base, {refined_props} refined)")
    tqdm.write("-" * 30)
    models_dir = f"{saving_dir}/models"
    try:
        os.mkdir(models_dir)
    except:
        pass
    prop_dir = f"{saving_dir}/properties"
    try:
        os.mkdir(prop_dir)
    except:
        pass
    # 1. Generate Model and Properties
    tqdm.write(f"\n[{idx}:0.5/4] Generating model ...")
    atomic_props = [f"p_{i}" for i in range(atomic_props)]
    
    model_gen = HybridModelGenerator(
        num_chain_states=chain_states,
        num_bits=bit_width,
        atomic_props=atomic_props
    )
    model_str = model_gen.to_nusmv_model()
    model_path = f"{models_dir}/benchmark_model_{idx}.smv"
    with open(model_path, 'w') as f:
        f.write(model_str)

    tqdm.write(f"\n[{idx}:1/4] Generating properties...")
    prop_gen = PropertyGenerator(atomic_props, num_equivalence_classes=num_classes)
    
    _,all_properties, _ = prop_gen.generate(base_props, refined_props)
    props_path = f"{prop_dir}/benchmark_properties_{idx}.txt"
    with open(props_path, 'w') as f:
        f.write("\n".join(all_properties["custom"]))
    tqdm.write(f"  -> Model saved to '{model_path}'")
    tqdm.write(f"  -> {len(all_properties["custom"])} properties saved to '{props_path}'")

    # 2. Benchmark Scenario B: Brute-Force Model Checking
    tqdm.write(f"[{idx}:2/4] Running Scenario B: Brute-force (checking all properties)...")
    time_brute_force = run_nusmv_check(model_path, all_properties["translated"],nusmv_path, id, not_print)
    tqdm.write(f"  -> Brute-force time: {time_brute_force:.4f} seconds")
#
    ## 3. Benchmark Scenario A: Analysis + Refined Model Checking
    tqdm.write(f"\n[{idx}:3/4] Running Scenario A: Analysis + Refined Check...")
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

    #for cls in required_properties.values():
    #    req_props+= flatten(cls["roots"])
    #    req_props+= flatten(cls["cycles"])
    required_properties_svm = [to_nusmv_syntax(p[1:-2]) for p in required_properties]

    props_path = f"{prop_dir}/benchmark_properties_svm_{idx}.txt"
    with open(props_path, 'w') as f:
        f.write("\n".join(required_properties_svm))

    #tqdm.write(len(required_properties_svm), len(all_properties["custom"]))
    ## 3b. Time the model checking of the smaller set
    time_mc_refined = run_nusmv_check(model_path, required_properties_svm,nusmv_path, id, not_print)
    tqdm.write(f"  -> Model checking time for refined set: {time_mc_refined:.4f} seconds")
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
    parser.add_argument("--nuvsm", type=str, default="mac", help="Nuvsm Path")
    parser.add_argument("--input", type=str, default="input_files/input_benchmarks.csv", help="Input file.")
    parser.add_argument("--output", type=str, default="resultBenchmark.csv", help="Result output file.")
    parser.add_argument("--output_dir", type=str, default="result_syn", help="Benchmark results folder.")
    parser.add_argument("--id", type=int, default=0, help="Benchmark results folder.")
    parser.add_argument("--continue_from", type=int, default=0, help="Benchmark results folder.")
    parser.add_argument("--end_at", type=int, default=123647890, help="Finish at.")
    parser.add_argument("--not_print", type=int, default=0, help="Not print")
    args = parser.parse_args()


    if args.nuvsm=="win":
        nuvsm_path = r'C:\NuSVM\NuSMV-2.7.0-win64\bin\NuSMV.exe'
    elif args.nuvsm=="mac":
        nuvsm_path = r"/Users/brdlcu/NuSMV/bin/NuSMV"
    else:
        nuvsm_path = args.nuvsm


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
        if idx<args.continue_from:
            continue
        states = row['states']
        transitions = row['transitions']
        atomic_props = row['atomic_props']
        base_props = row['base_props']
        refined_props = row['refined_props']
        num_classes = row['num_classes']
        chain_states = row['chain_states']
        bit_width = row['bit_width']
    
        # Print the variables for this row
        tqdm.write(f"Benchmark {idx}:")
        tqdm.write(f"  states: {states}")
        tqdm.write(f"  transitions: {transitions}")
        tqdm.write(f"  atomic_props: {atomic_props}")
        tqdm.write(f"  base_props: {base_props}")
        tqdm.write(f"  refined_props: {refined_props}")
        tqdm.write(f"  num_classes: {num_classes}")
        tqdm.write(f"  chain_states: {chain_states}")
        tqdm.write(f"  bit_width: {bit_width}")
        tqdm.write("-" * 40)
        output_time = run_once(idx,
                base_props, 
                refined_props,
                transitions,
                states,
                num_classes,
                atomic_props,
                chain_states,
                bit_width,
                output_folder,
                nuvsm_path, 
                args.id, 
                args.not_print
                )
        df.loc[idx] = output_time
        pbar.set_description(f"Prev. Run {output_time[-1]:.1f}% ")

        df.to_csv(output_folder + "/"+args.output, index=False)
    print(f"Saved results in {output_folder+ "/"+args.output}")
    



if __name__ == '__main__':
    main()