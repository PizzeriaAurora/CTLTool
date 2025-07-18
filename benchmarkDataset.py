


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
resu_folder="result2"
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
            new_row = pd.DataFrame({ COLUMNS[0]: [f"{filename[:-4]}"], 
                                    COLUMNS[1]:[len(analyzer.asts)],
                                    COLUMNS[2]: [temp],
                                    COLUMNS[3]: [analysis_time],

                                    })
            df = pd.concat([df, new_row], ignore_index=True)
            #analyzer.write_report_and_graphs(
            #    filename=filename,
            #    head_folder=f"{resu_folder}/PR2019"
            #    )
        else:
            print("No properties loaded, exiting.")
    return df


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
            analyzer.analyze_refinements_opt(True)
            analysis_time = round(time.time() - analysis_start_time,4)
            temp = len(analyzer.get_required_properties())
            print(f"From {len(analyzer.asts)} only {temp} are required checked in {analysis_time}s")
            new_row = pd.DataFrame({ COLUMNS[0]: [f"{filename[:-4]}"], 
                                    COLUMNS[1]:[len(analyzer.asts)],
                                    COLUMNS[2]: [temp],
                                    COLUMNS[3]: [analysis_time],

                                    })
            df = pd.concat([df, new_row], ignore_index=True)
            #analyzer.write_report_and_graphs(
            #    filename=filename,
            #    head_folder=f"{resu_folder}/INDPR2019"
            #    )
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
        #analyzer.write_report_and_graphs(
        #        filename="natlang",
        #        head_folder=f"{resu_folder}/NatLangCTL"
        #        )
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
    


def MCC(df = None):
    import os
    from tqdm import tqdm
    SOURCE_DIR = "./benchmarksDataset/MCC/extractedFiles_001"
    files = os.listdir(SOURCE_DIR)
    already_done = pd.read_csv("./result2/complete1.csv")
    done_files = list(already_done["TestName"])
    for idx,filename in tqdm(enumerate(files),total=len(files)):
        print(f"== {filename} ==")
        if filename[:-4] in done_files:
            continue
        source_filepath = os.path.join(SOURCE_DIR, filename)
        analyzer = Analyzer.load_from_file(source_filepath)
        if analyzer.asts:
            analysis_start_time = time.time()
            analyzer.build_equivalence_classes()
            analyzer.analyze_refinements_opt(True)
            analysis_time = round(time.time() - analysis_start_time,4)
            print("Getting requ properties")
            temp = len(analyzer.get_required_properties())
            print(f"From {len(analyzer.asts)} only {temp} are required checked in {analysis_time}s")
            new_row = pd.DataFrame({ COLUMNS[0]: [f"{filename[:-4]}"], 
                                    COLUMNS[1]:[len(analyzer.asts)],
                                    COLUMNS[2]: [temp],
                                    COLUMNS[3]: [analysis_time],

                                    })
            df = pd.concat([df, new_row], ignore_index=True)
            #df.to_csv(f"{resu_folder}/complete3.csv")
            #analyzer.write_report_and_graphs(
            #    filename=filename,
            #    head_folder=f"{resu_folder}/mcc"
            #    )
        else:
            print("No properties loaded, exiting.")
    return df

if __name__ == "__main__":
    df = pd.DataFrame(columns=COLUMNS)
    
    df = ParallelRers2019Ctl(df)
    df.to_csv(f"{resu_folder}/complete.csv")
    df = T2(df)
    df.to_csv(f"{resu_folder}/complete.csv")
    df = NatLangCTL(df)
    df.to_csv(f"{resu_folder}/complete.csv")
    df = MCC(df)
    df.to_csv(f"{resu_folder}/complete3.csv")
    #df = INDParallelRers2019Ctl(df)
    #df.to_csv(f"{resu_folder}/complete3.csv")
    
    #NatLangCTL()
   
