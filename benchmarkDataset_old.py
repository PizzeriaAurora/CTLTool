


from RefCheck.Analyzer import Analyzer
import re
import time
import pandas as pd
def test():
    from RefCheck.Property import CTLProperty
    prop1 = CTLProperty("AG[0,100] (system_on -> AF[0,10] watchdog_ok)")
    prop = CTLProperty("AG[0,100] (system_on -> AF[0,20] watchdog_ok)")
    print(prop.tree().pretty())
    print(prop1.tree().pretty())
    print(prop.refines(prop1))
    


COLUMNS = ["TestName", "TotalNumberOfProperties", "SizeMinimalSet", "time"]
resu_folder="result"
def ParallelRers2019Ctl(df):
    import os
    from tqdm import tqdm
    SOURCE_DIR = "./benchmarksDataset/ParallelRers2019Ctl"
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
            analyzer.write_report_and_graphs(
                filename=filename,
                head_folder=f"{resu_folder}/PR2019"
                )
        else:
            print("No properties loaded, exiting.")
    return df


# not working, need a way to deal with properties like <>() next to parenthesis. 
def INDParallelRers2019Ctl(df = None):
    import os
    from tqdm import tqdm
    SOURCE_DIR = "./benchmarksDataset/Rers2019Ind"
    files = os.listdir(SOURCE_DIR)
    for idx,filename in tqdm(enumerate(files),total=len(files)):
        source_filepath = os.path.join(SOURCE_DIR, filename)
        analyzer = Analyzer.load_from_file(source_filepath)
        if analyzer.asts:
            analysis_start_time = time.time()
            analyzer.build_equivalence_classes()
            analyzer.analyze_refinements_opt(False)
            analysis_time = round(time.time() - analysis_start_time,4)
            temp = len(analyzer.get_required_properties())
            print(f"From {len(analyzer.asts)} only {temp} are required checked in {analysis_time}s")
            new_row = pd.DataFrame({ COLUMNS[0]: [f"ParallelRers2019Ctl_{idx}"], 
                                    COLUMNS[1]:[len(analyzer.asts)],
                                    COLUMNS[2]: [temp],
                                    COLUMNS[3]: [analysis_time],

                                    })
            df = pd.concat([df, new_row], ignore_index=True)
            analyzer.write_report_and_graphs(
                filename=filename,
                head_folder=f"{resu_folder}/INDPR2019"
                )
        else:
            print("No properties loaded, exiting.")
    return df


def NatLangCTL(df):
    #import pandas as pd
    #dataset = pd.read_csv("benchmarksDataset/Natural2CTLclean.csv", delimiter=";")
    #with open("Natural2CTLclean.txt", "w") as output:
    #    for row in list(dataset["CTL"]):
    #        output.write(str(row).replace("stable","") + '\n')
    analyzer=Analyzer.load_from_file("benchmarksDataset/Natural2CTLclean.txt")
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
        analyzer.write_report_and_graphs(
                filename="natlang",
                head_folder=f"{resu_folder}/NatLangCTL"
                )
        return df
    else:
        print("No properties loaded, exiting.")


def T2(df):
    analyzer = Analyzer.load_from_file("benchmarksDataset/T2.txt")

    #print(analyzer.asts)
    if analyzer.asts:
        analysis_start_time = time.time()
        analyzer.build_equivalence_classes()
        analyzer.analyze_refinements(True)
        analysis_time = round(time.time() - analysis_start_time,4)
        temp = len(analyzer.get_required_properties())
        print(f"From {len(analyzer.asts)} only {temp} are required")
        #analyzer.write_report_and_graphs(head_folder=f"{resu_folder}/T2")
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
    
    #df = ParallelRers2019Ctl(df)
    #df.to_csv(f"{resu_folder}/test_test_testDatasetResult.csv")
    #df = T2(df)
    #df.to_csv(f"{resu_folder}/test_test_testDatasetResult.csv")
    #df = NatLangCTL(df)
    df = INDParallelRers2019Ctl(df)
    df.to_csv(f"{resu_folder}/test_test_testDatasetResult.csv")
    #NatLangCTL()
   
