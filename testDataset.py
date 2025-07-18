


from RefCheck.Analyzer import Analyzer
import re
import time
import pandas as pd
def test():
    from RefCheck.Property import TCTLProperty
    prop1 = TCTLProperty("AG[0,100] (system_on -> AF[0,10] watchdog_ok)")
    prop = TCTLProperty("AG[0,100] (system_on -> AF[0,20] watchdog_ok)")
    print(prop.tree().pretty())
    print(prop1.tree().pretty())
    print(prop.refines(prop1))
    


COLUMNS = ["TestName", "TotalNumberOfProperties", "SizeMinimalSet", "time"]

def ParallelRers2019Ctl(df):
    import os
    from tqdm import tqdm
    SOURCE_DIR = "./benchmarks/ParallelRers2019Ctl"
    files = os.listdir(SOURCE_DIR)
    for idx,filename in tqdm(enumerate(files),total=len(files)):
        source_filepath = os.path.join(SOURCE_DIR, filename)
        analyzer = Analyzer.load_from_file(source_filepath)
        if analyzer.asts:
            analysis_start_time = time.time()
            analyzer.build_equivalence_classes()
            analyzer.analyze_refinements(True)
            analysis_time = round(time.time() - analysis_start_time,4)
            temp = len(analyzer.get_required_properties())
            print(f"From {len(analyzer.asts)} only {temp} are required checked in {analysis_time}s")
            new_row = pd.DataFrame({ COLUMNS[0]: [f"ParallelRers2019Ctl_{idx}"], 
                                    COLUMNS[1]:[len(analyzer.asts)],
                                    COLUMNS[2]: [temp],
                                    COLUMNS[3]: [analysis_time],

                                    })
            df = pd.concat([df, new_row], ignore_index=True)
            #analyzer.write_report_and_graphs(
            #    filename=filename,
            #    head_folder="result2/PR2019"
            #    )
        else:
            print("No properties loaded, exiting.")
    return df


# not working, need a way to deal with properties like <>() next to parenthesis. 
def INDParallelRers2019Ctl(df = None):
    import os
    from tqdm import tqdm
    SOURCE_DIR = "./benchmarks/Rers2019Ind"
    files = os.listdir(SOURCE_DIR)
    for filename in tqdm(files):
        source_filepath = os.path.join(SOURCE_DIR, filename)
        analyzer = Analyzer.load_from_file(source_filepath)
        if analyzer.asts:
            analyzer.build_equivalence_classes()
            analyzer.analyze_refinements(True)
            print(f"From {len(analyzer.asts)} only {len(analyzer.get_required_properties())} are required")
            #analyzer.write_report_and_graphs(
            #    filename=filename,
            #    head_folder="result2/INDPR2019"
            #    )
        else:
            print("No properties loaded, exiting.")



def NatLangCTL(df):
    #import pandas as pd
    #dataset = pd.read_csv("benchmarks/Natural2CTLclean.csv", delimiter=";")
    #with open("Natural2CTLclean.txt", "w") as output:
    #    for row in list(dataset["CTL"]):
    #        output.write(str(row).replace("stable","") + '\n')
    analyzer=Analyzer.load_from_file("benchmarks/Natural2CTLclean.txt")
    if analyzer.asts:
        analysis_start_time = time.time()
        analyzer.build_equivalence_classes()
        analyzer.analyze_refinements_opt(True)
        analysis_time = round(time.time() - analysis_start_time,4)
        temp = len(analyzer.get_required_properties())
        print(f"From {len(analyzer.asts)} only {temp} are required")
        new_row = pd.DataFrame({ COLUMNS[0]: [f"NatLang"], 
                                    COLUMNS[1]:[len(analyzer.asts)],
                                    COLUMNS[2]: [temp],
                                    COLUMNS[3]: [analysis_time],

                                    })
        df = pd.concat([df, new_row], ignore_index=True)
        #analyzer.write_report_and_graphs(
        #        filename="natlang",
        #        head_folder="result2/NatLangCTL"
        #        )
        return df
    else:
        print("No properties loaded, exiting.")


def T2(df):
    analyzer = Analyzer.load_from_file("benchmarks/T2.txt")

    #print(analyzer.asts)
    if analyzer.asts:
        analysis_start_time = time.time()
        analyzer.build_equivalence_classes()
        analyzer.analyze_refinements(True)
        analysis_time = round(time.time() - analysis_start_time,4)
        temp = len(analyzer.get_required_properties())
        print(f"From {len(analyzer.asts)} only {temp} are required")
        #analyzer.write_report_and_graphs(head_folder="result2/T2")
        new_row = pd.DataFrame({ COLUMNS[0]: [f"T2"], 
                                    COLUMNS[1]:[len(analyzer.asts)],
                                    COLUMNS[2]: [temp],
                                    COLUMNS[3]: [analysis_time],
                                    })
        df = pd.concat([df, new_row], ignore_index=True)
        return df
    else:
        print("No properties loaded, exiting.")
    

if __name__ == "__main__":
    df = pd.DataFrame(columns=COLUMNS)
    df = ParallelRers2019Ctl(df)
    df.to_csv("result2/testDatasetResult.csv")
    df = T2(df)
    df.to_csv("result2/testDatasetResult.csv")
    df = NatLangCTL(df)
    df.to_csv("result2/testDatasetResult.csv")
    #NatLangCTL()
   
