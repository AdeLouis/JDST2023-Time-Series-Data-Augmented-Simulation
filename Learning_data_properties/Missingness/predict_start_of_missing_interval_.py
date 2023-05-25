
import sys
import glob
import pandas as pd
import pickle
import random
import math
from collections import Counter

import numpy as np
from joblib import Parallel, delayed

from missing_prediction_utils import *

import os
import tensorflow as tf
import tensorflow.keras as keras
os.environ["CUDA_VISIBLE_DEVICES"] = '5'

from numpy.random import seed
seed(1)
tf.random.set_seed(2)

def extract_all(file_paths):

    def func(hist,all_subj,num_features,model_type,dataset):
        print("History window: {}".format(hist))

        all_train_subj,validation_data,scaler = run_for_all(all_subj,hist,num_features,model_type,dataset)

        return (hist,all_train_subj,validation_data,scaler)

    all_subj = {}
    subj_names = []

    for file in file_paths:        
        subj = file.split("/")[-1]
        subj = subj.split(".")[0]

        print("-------------EXTRACT DATA FOR PATIENT: " + str(subj) + " -------------")
        indices, temp_df, df = load_csv(file,sampling_rate)
        subsets = get_valid_subsets(indices,temp_df,df,dataset)
        
        all_subj[subj.split(".")[0]] = (subsets)
        subj_names.append(subj.split(".")[0])

    hists = create_window_sizes(sampling_rate,max_size=history_window*sampling_rate) #different lengths of history windows
    res = []
    stored_train = []
    res = Parallel(n_jobs = 22)(delayed(func)(hist,all_subj,num_features,dataset) for hist in hists)

    for group in res:
        hist,all_train_subj,validation_data,scalers = group
        stored_train.append((hist,all_train_subj,validation_data,scalers))

    return stored_train
        
def extract_LOPO(file_paths):

    all_subj = {}
    subj_names = []

    for file in file_paths:        
        subj = file.split("/")[-1]
        subj = subj.split(".")[0]

        print("-------------EXTRACT DATA FOR PATIENT: " + str(subj) + " -------------")
        indices, temp_df, df = load_csv(file,sampling_rate)
        subsets = get_valid_subsets(indices,temp_df,df,dataset)
        
        all_subj[subj.split(".")[0]] = (subsets)
        subj_names.append(subj.split(".")[0])

    hists = create_window_sizes(sampling_rate,max_size=int(history_window*sampling_rate)) 
    stored_train, stored_test = [], []
    
    for hist in hists:
        print("-------------history: {}".format(hist))
        test_all_subj = {}
        train_all_subj = {}

        res = []
        res = Parallel(n_jobs=30)(delayed(run_for_lopo)(subj,all_subj,subj_names,hist,num_features,dataset) for subj in subj_names)

        
        for all_res in res:
            subj,lopo_train_subj,validation_data,test_windows,test_labels,scaler = all_res
            train_all_subj[subj] = (lopo_train_subj,validation_data)
            test_all_subj[subj] = (test_windows,test_labels)
        
        stored_train.append((hist,train_all_subj))
        stored_test.append((hist,test_all_subj))
        
    return stored_train, stored_test

def run_anomaly_LOPO(unpickled_train_data, unpickled_test_data,test_subjs):
    
    result_data = pd.DataFrame()
    windowed_result_data = pd.DataFrame()
  
    for test_subj in test_subjs:
        
        print("-------------LOPO PATIENT: " + test_subj + " -------------")
        xtrain_data, xval_data = [], []
        
        for hist,fixed_sized_data in unpickled_train_data:
            subjs_data = fixed_sized_data[test_subj]
            train_subjs_data, validation_data = subjs_data
            train_subjs = list(train_subjs_data.keys())
            random.shuffle(train_subjs)

            val_X,val_xvec,val_y = validation_data
            #print("validation distribution: {}".format(Counter(val_y)))
            train_X,train_xvec,train_y = setup_data(train_subjs,train_subjs_data,hist,num_features)  #unpack data

            print("group {} with distribution: {}".format(hist,Counter(train_y)))
            print("group {} with distribution: {}".format(hist,Counter(val_y)))

            xtrain_data.append((train_X,train_y))
            xval_data.append((val_X,val_y))

        xtrain_data,ycounts = generate_batches(xtrain_data)
        xval_data,valycounts = generate_batches(xval_data)

        model,th,_ = train_model(xtrain_data,xval_data,ycounts,num_features,valycounts)
        
        print("-------------TESTING-------------")
        xtest_data = []
        
        for hist,fixed_sized_data in unpickled_test_data:       #cycle through each input sequence length
            test_subj_data = fixed_sized_data[test_subj]
            test_X,test_y = test_subj_data
            xtest_data.append((test_X,test_y))
        
        xtest_data,_ = generate_batches(xtest_data)
        test_subj_df,window_df,base_auprc = test_model(xtest_data,model,test_subj,th)

        test_subj_df["pred_threshold"] = th
        test_subj_df["Base_auprc"] = base_auprc
        result_data = pd.concat([result_data,test_subj_df])
        windowed_result_data = pd.concat([windowed_result_data,window_df])
        
    result_data = result_data.sort_values(by = ["LOPO_Subj"])
    windowed_result_data = windowed_result_data.sort_values(by=["LOPO_Subj"])
    result_data.set_index("LOPO_Subj", inplace = True)
    windowed_result_data.set_index("LOPO_Subj",inplace = True)

    return result_data,windowed_result_data

def run_anomaly_all(unpickled_train_data):

    xtrain_data, xval_data = [], []

    for group in unpickled_train_data:
        hist,train_subjs_data,validation_data,scalers = group
        train_subjs = list(train_subjs_data.keys())
        random.shuffle(train_subjs)

        val_X,val_y = validation_data
        train_X,train_y = setup_data(train_subjs,train_subjs_data,hist,num_features)  #unpack data

        #print("group {} with distribution: {}".format(hist,Counter(val_y)))
        #print("group {} with distribution: {}".format(hist,Counter(train_y)))

        xtrain_data.append((train_X,train_y))
        xval_data.append((val_X,val_y))
    
    xtrain_data,ycounts = generate_batches(xtrain_data)
    xval_data,valycounts = generate_batches(xval_data)

    model,th,p_05 = train_model(xtrain_data,xval_data,ycounts,num_features,valycounts)
    #we save the model, threshold for finding anomaly and the scalers
    model.save(output + "prediction_model.h5")

    with open(output + 'prediction_model_scalers.pickle', 'wb') as f:
        pickle.dump(scalers, f, pickle.HIGHEST_PROTOCOL)

    with open(output + 'prediction_model_params.pickle', 'wb') as f:
        pickle.dump((th,p_05), f, pickle.HIGHEST_PROTOCOL)
    

if __name__ == "__main__":

    if len(sys.argv) > 1:
        path = str(sys.argv[1])                 #path for direcotry to where data is stored
        num_features = int(sys.argv[2])
        dataset = sys.argv[3]
        data_to_use = sys.argv[4]               #csv file for data to use for missing data prediction
        ind = int(sys.argv[5])                  #1 for LOPO, 2 for ALL
        output = sys.argv[6]                    #output path
        history_window = int(sys.argv[7])
        sampling_rate = int(sys.argv[8])        #sampling rate of variable of interest

        output = output + dataset + "/"

        df = pd.read_csv(data_to_use)           #get ids for subjects to use
        ids = list(df["PID"])
        file_paths = []
        for id in ids:
            file_paths.append(path + str(id) + ".csv")
        ids = [str(id) for id in ids]
            
        if ind == 1:
            #for LOPO
            unpickled_train_data,unpickled_test_data = extract_LOPO(file_paths)
            results,window_results = run_anomaly_LOPO(unpickled_train_data,unpickled_test_data,ids)
            results.to_csv(output + "anomaly-results.csv")
            window_results.to_csv(output +"anomaly_window_results.csv")

        else:
            unpickled_train_data = extract_all(file_paths)
            run_anomaly_all(unpickled_train_data)
