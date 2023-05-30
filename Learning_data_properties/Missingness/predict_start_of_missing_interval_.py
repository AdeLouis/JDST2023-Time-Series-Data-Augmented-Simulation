
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

def extract_windowed_data(file_paths):

    def func(hist,all_subj,num_features,dataset):
        print("History window: {}".format(hist))

        all_train_subj,validation_data,min_val,max_val = run_for_all(all_subj,hist,num_features,dataset)
        return (hist,all_train_subj,validation_data,min_val,max_val )

    def standardize_func(group,scaler,num_features):
        
        hist,all_train_subj,validation_data,_,_ = group
        print("Starting standardization for hist size: {}".format(hist))
        all_train_subj = standardize(all_train_subj,scaler,"train",num_features)
        validation_windows, validation_windows_vec, validation_labels = validation_data
        validation_windows = standardize(validation_windows,scaler,"test",num_features)
        validation_data = (validation_windows,validation_windows_vec,validation_labels)

        print("Ending standardization for hist size: {}".format(hist))
        return (hist,all_train_subj,validation_data,scaler)

    all_subj = {}
    subj_names = []

    for file in file_paths: 
        if dataset == "ohio":
            subj = file.split("/")[-2]    
        else:            
            subj = file.split("/")[-1]
            if dataset in ["rct","race"]:
                subj = subj.split("_")[0]

        print("-------------EXTRACT DATA FOR PATIENT: " + str(subj) + " -------------")
        try:
            indices, temp_df, df = load_csv(file,dataset)
            subsets = get_valid_subsets(indices,temp_df,df,dataset)

        except:
            continue
        
        all_subj[subj.split(".")[0]] = (subsets)
        subj_names.append(subj.split(".")[0])

    hists = create_window_sizes(sampling_rate,max_size=history_window*sampling_rate) #different lengths of history windows
    results = []
    stored_train = []
    results = Parallel(n_jobs = 20)(delayed(func)(hist,all_subj,num_features,dataset) for hist in hists)

    min_values = np.zeros((1,num_features))
    max_values = np.zeros((1,num_features))

    for group in results:
        _,_,_,hist_min,hist_max = group
        min_values = np.concatenate((min_values,hist_min))
        max_values = np.concatenate((max_values,hist_max))

    min_values = min_values[1:]
    max_values = max_values[1:]
    min_values = np.nanmin(min_values,axis = 0)
    max_values = np.nanmax(max_values,axis = 0)

    #create fake array to form scaler
    curated_array = np.concatenate(([min_values],[max_values])) #shape is 2 by num_features
    
    scaler = MinMaxScaler()
    scaler.fit(curated_array)

    print("Starting standardization")

    final_results = Parallel(n_jobs = 20)(delayed(standardize_func)(group,scaler,num_features) for group in results)
    for group in final_results:
        stored_train.append(group)

    print("Completing standardization")

    return stored_train


if __name__ == "__main__":

    if len(sys.argv) > 1:
        path = str(sys.argv[1])                 #path for direcotry to where data is stored
        num_features = int(sys.argv[2])
        dataset = sys.argv[3]
        data_to_use = sys.argv[4]               #csv file for data to use for missing data prediction
        output = sys.argv[5]                    #output path
        history_window = int(sys.argv[6])
        sampling_rate = int(sys.argv[7])        #sampling rate of variable of interest

        df = pd.read_csv(data_to_use)           #get ids for subjects to use
        ids = list(df["PID"])
        file_paths = []
        for id in ids:
            file_paths.append(path + str(id) + ".csv")
        ids = [str(id) for id in ids]

        extracted_train_data = extract_windowed_data(file_paths)

            
    
