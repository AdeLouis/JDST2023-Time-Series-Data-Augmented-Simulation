#Script is used to combine results from the model run for experiment 1

import pandas as pd
import numpy as np
import glob as glob

def real_combine(datasets,first_index, second_index):
    summary_df = pd.DataFrame()
    summary_df["Model"] = first_index
    summary_df["Type"] = second_index

    for dataset in datasets:
        results = []

        for model in model_names:
            df = pd.read_csv(path + dataset + "/" + model + "_results.csv",index_col = 0)
            rmse_values = df["Mean Test RMSE"].to_numpy()
            #results.append(np.min(rmse_values))
            results.append(np.mean(rmse_values)) #average rmse for all subjects over 10 iterations
            #results.append(np.max(rmse_values))

        summary_df[dataset] = results

    summary_df.to_csv(path + "real.csv")

def sim_combine(datasets,first_index, second_index,kind):

    summary_df = pd.DataFrame()
    summary_df["Model"] = first_index
    summary_df["Type"] = second_index

    for dataset in datasets:
        results = []
        flag = "true"

        for model in model_names:
            try:
                df = pd.read_csv(path + dataset + "/" + model + "_results.csv",index_col = 0)
                rmse_values = df["Mean Test RMSE"].to_numpy()
                #results.append(np.min(rmse_values))
                results.append(np.mean(rmse_values)) #average rmse for all subjects over 10 iterations
                #results.append(np.max(rmse_values))
            except:
                pass
                flag = "fail"
        
        if kind in ["rct","oaps","ohio"]:
            if flag == "true":
                if dataset == "sim_adult":
                    summary_df["Sim"] = results
                elif dataset == "sim_adult_dropout":
                    summary_df["Dropout_random"] = results
                elif dataset == "sim_adult_"+kind+"_missing":
                    summary_df["Dropout_predicted"] = results
                elif dataset == "sim_adult_noise":
                    summary_df["Error_gaussian"] = results
                elif dataset == "sim_adult_cgmerror":
                    summary_df["Error_cgm"] = results
                elif dataset == "sim_adult_"+kind+"_error":
                    summary_df["Error_predicted"] = results
                elif dataset == "sim_adult_dropout_noise":
                    summary_df["Error_gaussian-Dropout_random"] = results
                elif dataset == "sim_adult_cgmerror_dropout":
                    summary_df["Error_cgm-Dropout_random"] = results
                elif dataset == "sim_adult_"+kind+"_error_dropout":
                    summary_df["Error_predicted-Dropout_random"] = results
                elif dataset == "sim_adult_"+kind+"_missing_noise":
                    summary_df["Dropout_predicted-Error_gaussian"] = results
                elif dataset == "sim_adult_"+kind+"_missing_cgmerror":
                    summary_df["Dropout_predicted-Error_cgm"] = results
                elif dataset == "sim_adult_"+kind+"_missing_error":
                    summary_df["Dropout_predicted-Error_predicted"] = results
                elif dataset == "sim_adult_"+kind+"_cgmerror_missing":
                    summary_df["Error_cgm-Dropout_predicted"] = results
                elif dataset == "sim_adult_"+kind+"_error_missing":
                    summary_df["Error_predicted-Dropout_predicted"] = results
                elif dataset == "sim_adult_"+kind+"_noise_missing":
                    summary_df["Error_gaussian-Dropout_predicted"] = results
                else:
                    summary_df[dataset] = results
        
        else:
            if flag == "true":
                if dataset == "sim_standard":
                    summary_df["Sim"] = results
                elif dataset == "sim_standard_dropout":
                    summary_df["Dropout_random"] = results
                elif dataset == "sim_standard_"+kind+"_missing":
                    summary_df["Dropout_predicted"] = results
                elif dataset == "sim_standard_noise":
                    summary_df["Error_gaussian"] = results
                elif dataset == "sim_standard_cgmerror":
                    summary_df["Error_cgm"] = results
                elif dataset == "sim_standard_"+kind+"_error":
                    summary_df["Error_predicted"] = results
                elif dataset == "sim_standard_dropout_noise":
                    summary_df["Error_gaussian-Dropout_random"] = results
                elif dataset == "sim_standard_cgmerror_dropout":
                    summary_df["Error_cgm-Dropout_random"] = results
                elif dataset == "sim_standard_"+kind+"_error_dropout":
                    summary_df["Error_predicted-Dropout_random"] = results
                elif dataset == "sim_standard_"+kind+"_missing_noise":
                    summary_df["Dropout_predicted-Error_gaussian"] = results
                elif dataset == "sim_standard_"+kind+"_missing_cgmerror":
                    summary_df["Dropout_predicted-Error_cgm"] = results
                elif dataset == "sim_standard_"+kind+"_missing_error":
                    summary_df["Dropout_predicted-Error_predicted"] = results
                elif dataset == "sim_standard_"+kind+"_cgmerror_missing":
                    summary_df["Error_cgm-Dropout_predicted"] = results
                elif dataset == "sim_standard_"+kind+"_error_missing":
                    summary_df["Error_predicted-Dropout_predicted"] = results
                elif dataset == "sim_standard_"+kind+"_noise_missing":
                    summary_df["Error_gaussian-Dropout_predicted"] = results
                else:
                    summary_df[dataset] = results
        

    print(summary_df.T)
    summary_df.to_csv(path + "summary_results.csv")


dataset_type = "sim"
dataset = "race"
model_names = ["REG","TREE","RNN","LSTM"]#,"SVR"]

first_index = []
for model in model_names:
    first_index.extend([model]*1)

second_index = []
for i in range(0,len(model_names)):
    second_index.extend(["Mean"])

if dataset_type == "real":
    path = "/data/PHI/PHI_OAPS/sandbox/lgomez/Sim_Project/simulation/Real_data/Results_Expr_A/"
    datasets = ["rct","oaps","race","ohio"]
    real_combine(datasets,first_index,second_index)

else:
    path = "/data/PHI/PHI_OAPS/sandbox/lgomez/Sim_Project/simulation/Simulated_data/" + dataset + "_results_Expr_A/"
    
    if dataset in ["rct","oaps","ohio"]:
        datasets = ["sim_adult","sim_adult_dropout","sim_adult_"+dataset+"_missing","sim_adult_noise","sim_adult_cgmerror","sim_adult_"+dataset+"_error",
                    "sim_adult_dropout_noise","sim_adult_cgmerror_dropout","sim_adult_"+dataset+"_error_dropout","sim_adult_"+dataset+"_missing_noise",
                    "sim_adult_"+dataset+"_missing_cgmerror","sim_adult_"+dataset+"_missing_error","sim_adult_"+dataset+"_cgmerror_missing",
                    "sim_adult_"+dataset+"_error_missing","sim_adult_"+dataset+"_noise_missing"]
    else:
        datasets = ["sim_standard","sim_standard_dropout","sim_standard_"+dataset+"_missing","sim_standard_noise","sim_standard_cgmerror","sim_standard_"+dataset+"_error",
                    "sim_standard_dropout_noise","sim_standard_cgmerror_dropout","sim_standard_"+dataset+"_error_dropout","sim_standard_"+dataset+"_missing_noise",
                    "sim_standard_"+dataset+"_missing_cgmerror","sim_standard_"+dataset+"_missing_error","sim_standard_"+dataset+"_cgmerror_missing",
                    "sim_standard_"+dataset+"_error_missing","sim_standard_"+dataset+"_noise_missing"]

    
    sim_combine(datasets,first_index,second_index,dataset)





