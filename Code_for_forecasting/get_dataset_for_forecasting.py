'''This script is a one stop file to process all the datasets we have and prep them for time series forecasting'''

import numpy as np
import pandas as pd
import pickle
import os
import random
from datetime import timedelta
import sys
import glob as glob
from joblib import Parallel, delayed

from forecasting_utils import *

random.seed(10)
np.random.seed(10)


def series_to_supervised(all_subjects,dataset_type):
    '''Function to convert to time windows for forecasting'''

    def func(subj,all_subjects,prediction_window):
        print('-----------------Windowing data for Subject: '+subj+'-------------------')
        all_windowed_data[subj] = {}
        keys = list(all_subjects[subj].keys())
        cgm_store = np.zeros((1,samples_per_hour,1))
        target_store = np.zeros(1)

        for key in keys:
            df = all_subjects[subj][key]
            
            cgm = df["glucose_level"].to_numpy()

            cgm_seq = rolling_window(cgm,samples_per_hour,1)
            target = rolling_window(cgm,samples_per_hour+prediction_window,1)[:,-1]

            diff = len(target)-len(cgm_seq)
            cgm_seq = cgm_seq[0:diff,:]
            

            assert len(cgm_seq) == len(target)

            cgm_seq = cgm_seq.reshape((cgm_seq.shape[0],samples_per_hour,1))
            cgm_store = np.concatenate((cgm_store,cgm_seq))
            target_store = np.concatenate((target_store,target))

        print(cgm_store.shape)
        return cgm_store,target_store,subj

    subjs = list(all_subjects.keys())
    all_windowed_data = {}
    prediction_window = int(30/sampling_rate)

    res = []
    res = Parallel(n_jobs=1)(delayed(func)(subj_name,all_subjects,prediction_window) for subj_name in subjs)

    for item in res:
        cgm_store,target_store,subj = item
        all_windowed_data[subj] = (cgm_store,target_store)

    with open(output + dataset+ "_" + str(samples_per_hour) +"_" + str(no_days) +"_windowed_" +dataset_type +".pickle", 'wb') as f:
            pickle.dump(all_windowed_data, f, pickle.HIGHEST_PROTOCOL)

    
    if experiment == 1:    
        with open(output + dataset+ "_" + str(samples_per_hour) +"_" + str(no_days) +"_windowed_" +dataset_type +".pickle", 'wb') as f:
            pickle.dump(all_windowed_data, f, pickle.HIGHEST_PROTOCOL)

    else:
        if "sim" in dataset:
            with open(output + dataset+ "_" + str(samples_per_hour) +"_" + str(no_days) +"_"+ "windowed_all.pickle", 'wb') as f:
                pickle.dump(all_windowed_data, f, pickle.HIGHEST_PROTOCOL)

        else:
            with open(output + dataset+ "_" + str(samples_per_hour) +"_" +  str(no_days) +"_windowed_" +dataset_type +".pickle", 'wb') as f:
                pickle.dump(all_windowed_data, f, pickle.HIGHEST_PROTOCOL)
    
def real_data_processing(subjs,dir,no_days):

    '''Function to process the data, divide into train and test sets'''

    all_subj_train, all_subj_test = {}, {}

    if experiment == 2:
        poss_train_subjs = check_length(subjs,dataset)
        subjs_to_add_to_test = [id for id in subjs if id not in poss_train_subjs]

        train_subjs = np.random.choice(poss_train_subjs,size = no_subjs, replace = False)
        subjs_not_selected = [id for id in poss_train_subjs if id not in train_subjs]
        test_subjs = subjs_not_selected + subjs_to_add_to_test

        for subj in train_subjs:
            train = get_data_splits(subj,dir,no_days,"train")
            res = process_data(train,"train",encoding_dataset)

            if res is None:
                pass
            else:
                all_subj_train[subj] = {}
                k = 0
                for item in res:
                    all_subj_train[subj][k] = item
                    k = k + 1

        for subj in test_subjs:
            test = get_data_splits(subj,dir,no_days,"test")
            res = process_data(test,"test",encoding_dataset)
            if res is None:
                pass
            else:
                all_subj_test[subj] = {}
                k = 0
                for item in res:
                    all_subj_test[subj][k] = item
                    k = k + 1

    else:
        #exclude patients with not enough days of data
        subjs = check_length(subjs,dataset)
        
        #sample x number of subjects based on user requirement
        subjs = np.random.choice(subjs,size = no_subjs, replace = False)

        for subj in subjs:
            train,test = get_data_splits(subj,dir,no_days)
            res = process_data(train,"train",encoding_dataset)

            all_subj_train[subj] = {}
            k = 0
            for item in res:
                all_subj_train[subj][k] = item
                k = k + 1

            res = process_data(test,"test",encoding_dataset)
            all_subj_test[subj] = {}
            k = 0
            for item in res:
                all_subj_test[subj][k] = item
                k = k + 1

    return all_subj_train,all_subj_test

def simulated_data_processing(dir,no_days):

    subjs = glob.glob(dir + "*.csv")
    subjs = [subj.split("/")[-1] for subj in subjs]
    subjs = [subj.split(".")[0] for subj in subjs]
    print(subjs)

    all_subj_train, all_subj_test = {}, {}
    if experiment == 2:
        subjs = np.random.choice(subjs,size = no_subjs, replace = False)

        for subj in subjs:
            train = get_data_splits(subj,dir,no_days,"train")
            all_subj_train[subj] = {}
            train["date"] = train["dates"].dt.date
            days = list(train["date"].unique())

            res = []
            for day in days:
                subset = train[train["date"] == day].copy()
                subset = subset.drop(columns="date")
                res.append(process_data(subset,"train",encoding_dataset))

            k = 0
            for day in res:
                for item in day:
                    all_subj_train[subj][k] = item
                    k = k + 1

    else:
        for subj in subjs:
            train,test = get_data_splits(subj,dir,no_days)

            all_subj_train[subj] = {}
            all_subj_test[subj] = {}
            print(train)
            
            train["date"] = train["dates"].dt.date
            days = list(train["date"].unique())
            
            res = []
            for day in days:
                subset = train[train["date"] == day].copy()
                subset = subset.drop(columns="date")
                res.append(process_data(subset,"train",encoding_dataset))

            k = 0
            for day in res:
                for item in day:
                    all_subj_train[subj][k] = item
                    k = k + 1

            #testing

            test["date"] = test["dates"].dt.date
            days = list(test["date"].unique())
            res = []
            for day in days:
                subset = test[test["date"] == day].copy()
                subset = subset.drop(columns="date")
                res.append(process_data(subset,"test",encoding_dataset))

            k = 0
            for day in res:
                for item in day:
                    all_subj_test[subj][k] = item
                    k = k + 1

    return all_subj_train,all_subj_test


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_directory = sys.argv[1]    #input folder for the csv files
        dataset = sys.argv[2]           #name of dataset
        data_learn = sys.argv[3]        #set of PIDs to be used for forecasting. In a csv file
        no_days = int(sys.argv[4])      #no. of days to use for forecasting
        no_subjs = int(sys.argv[5])     #number of subjects to use for forecasting
        output = sys.argv[6]            #output folder to store pickle files
        encoding_dataset = sys.argv[7]  #often the same dataset name
        
        experiment = int(sys.argv[8])
    
        sampling_rate = 5               #dafulat sampling rate of the dataset. For most cgm is 5 minutes
        samples_per_hour = 12

        #Processing for real data
        df = pd.read_csv(data_learn)         
        subjs_to_use = df.PID.to_list() 
        subjs_to_use = [str(subj) for subj in subjs_to_use]

        all_subj_train,all_subj_test = real_data_processing(subjs_to_use,data_directory,no_days)

        #convert the train and test sets into time windows for forecasting
        series_to_supervised(all_subj_test,"test")
        series_to_supervised(all_subj_train,"train")


        #Processing for Simulated data
        all_subj_train,all_subj_test = simulated_data_processing(data_directory,no_days)
        if experiment == 1:
            series_to_supervised(all_subj_test,"test")
        series_to_supervised(all_subj_train,"train")