

import sys
import numpy as np
import pandas as pd
import glob as glob
import math
import os
import random
import pickle
import time

#import tsfresh
#from tsfresh import extract_features
import tsfel

'''Parallelization'''
from joblib import Parallel, delayed, load
#clf = load('filename.joblib') 
from datetime import timedelta, datetime

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

#from tensorflow import keras
#model = keras.models.load_model('path/to/location')

#random.seed(10)
#np.random.seed(10)

'''Function related to Missing data prediction'''

def miss_standardize(data,scaler):

    fill = -1
    x,xvec = data
    x_scaler,xvec_scaler = scaler
    vec_test = []

    #operations for test data (test data can be the held out subject or validation dataset)
    test = x.reshape((x.shape[0]*x.shape[1],x.shape[2]))
    #test_data = test[:,x:].copy()
    test = x_scaler.transform(test)
    test = np.nan_to_num(test,nan = fill)
    #test[:,x:] = new_test
    test = test.reshape((x.shape[0],x.shape[1],x.shape[2]))

    #vec_test = xvec_scaler.transform(xvec)

    return test,vec_test

def miss_helper(df):
    df["Mask"] = 1
    df['hour'] = df['dateString'].dt.hour
    df["dow"] = df["dateString"].dt.dayofweek
    if dataset != "race":
        df["time_interval"] = 0.5
    else:
        df["time_interval"] = 1.5

    feature = "meal"
    value = list(df[feature])
    value = [1 if x > 0 else 0 for x in value] 
    df[feature + "_mask"] = value
    df.drop(columns = [feature], inplace = True)
    

    return df

def miss_loadmodels(model_path):
    '''Function to lead models needed for missing data prediction'''
    gap_scaler = None
    th = None
    p_05 = None

    #load prediction model
    model1 = load_model(model_path + dataset + "/prediction_model.h5",compile=False)

    #load scaling
    with open(model_path + dataset + "/prediction_model_scalers.pickle", 'rb') as f:
        pred_scalers = pickle.load(f)

    #load prediction threshold
    with open(model_path + dataset + "/prediction_model_params.pickle", 'rb') as f:
        th,p_05 = pickle.load(f)

    #load the gap length model
    model2 = load(model_path + dataset + "/length_tsfel_model.joblib")

    #load scaling
    with open(model_path + "Results_length/" + dataset + "/gap_model_tsfel_scaler.pickle", 'rb') as f:
        gap_scaler = pickle.load(f)

    return (model1,pred_scalers,th,p_05),(model2,gap_scaler)

def miss_model1(ts,missing_prediction_params):
    'Function to aid in predicting when missing data occurs'

    model,scaler,th,p_05 = missing_prediction_params             #unpack the tuple
    if dataset in ["oaps","ohio"]:
        features = ["glucose_level","time_interval","hour","dow","meal_mask"]
    elif dataset == "rct":
        features = ["glucose_level","time_interval","hour","dow"]
    elif dataset == "race":
        features = ["glucose_level","time_interval","hour","dow"]

    #format data for model input
    mini_w_size = len(ts)
    ts_input = ts[features].to_numpy()
    ts_input = ts_input.reshape(1,mini_w_size,len(features))

    temp = ["dow","hour"] #"meal_mask"

    ts_input_vec = ts[temp].to_numpy()
    ts_input_vec = ts_input_vec[-1]
    ts_input_vec = ts_input_vec.reshape(1,len(ts_input_vec))

    #standardize input data
    ts_input,ts_input_vec = miss_standardize([ts_input,ts_input_vec],scaler)

    #model prediction
    result = model.predict(ts_input)[0][0]
    rresult = (result > th).astype("int") #0.50 oaps 0.49 rct

    if rresult == 1:
        result = np.random.binomial(1,p_05,1)[0]
    else:
        result = 0

    return result
    
def miss_model2(ts,length_pred_params):
    '''Function to aid in predicitng the size of missing windows'''

    def feature_extraction(ts):
        test_data = None
        id = 1
        stats_file = tsfel.get_features_by_domain("statistical")
        temporal_file = tsfel.get_features_by_domain("temporal")
        try:
            stats_features = tsfel.time_series_features_extractor(stats_file,ts, fs=None, window_size=len(ts),n_jobs = 1,verbose=0,overlap = 0)
            temporal_features = tsfel.time_series_features_extractor(temporal_file,ts, fs=None, window_size=len(ts),n_jobs = 1,verbose = 0,overlap = 0)

            stats_features["ID"] = id
            temporal_features["ID"] = id
            test_data = stats_features.set_index('ID').join(temporal_features.set_index('ID'))
            test_data.reset_index(drop = True, inplace = True)
        except:
            print("failed for some reason")

        return test_data
    
    if dataset != "race":
        val = 5 * 60
    else:
        val = 15 *60
    
    model, scaler = length_pred_params
   
    #get variables that are used
    ts = ts.reset_index(drop = True)
    ts["ID"] = 1
    firsttime = ts.at[0,"dateString"]
    ts["time"] = ts["dateString"] - firsttime
    ts["time"] = ts["time"].dt.total_seconds()/val

    ts = ts[["ID","time","glucose_level"]].copy()
    ts = list(ts["glucose_level"])

    #feature extraction for tsfel
    transformed_ts = feature_extraction(ts)
    if transformed_ts is None:
        result = None
    else:
        result = model.predict(transformed_ts)

    return result

def predict_missing_data(data,miss_data_pred_params,length_pred_params):

    #setup dataset
    data = miss_helper(data)
    df_m = data[["dateString"]].copy(deep = True)
    df_m["Ind"] = 1
    ts_len = len(data)
    i = hist_max
    time_ranges = []

    if dataset != "race":
        min_val = 5
    else:
        min_val = 15

    while i < ts_len:                               
        ts_window = data[i:i+hist_max].copy()                               #we first get a time slice that is the length of the history window
        missing_data_pred = 0
        j = 1

        while (j < len(ts_window)+1) and (missing_data_pred == 0):          #The loop keep running until we are at the end of the history window or we predict a missing data
            test_window = ts_window[0:j].copy()                             #we take smaller time slice starting with size 1 up to the history window                         
            missing_data_pred = miss_model1(test_window,miss_data_pred_params)   #predict using missing data model, output is binary
            j = j + 1

        if missing_data_pred == 0:
            i = i + hist_max
        else: 
            print("here")   

            #get the gap prediction window
            gap_end_window = test_window.iloc[-1]["dateString"]
            if dataset == "race":
                gap_start_window = gap_end_window - timedelta(minutes = 180) 
            else:
                gap_start_window = gap_end_window - timedelta(minutes = 60)

            pred_window = data[(data["dateString"] > gap_start_window) & (data["dateString"] <= gap_end_window)].copy(deep = True)
            len_of_missing_window = miss_model2(pred_window,length_pred_params)

            if len_of_missing_window is None:
                i = i + hist_max
            else:
                missing_start_time = test_window.iloc[-1]["dateString"] + timedelta(minutes = min_val) #where the missing entries begin from
                missing_end_time = missing_start_time + timedelta(minutes = int(len_of_missing_window-min_val))
                time_ranges.append((missing_start_time,missing_end_time))
                i = i + (j-1) + int(len_of_missing_window/min_val)

    #mask indicator values as 0 where time was predicted missing
    for time_range in time_ranges:
        start, end = time_range
        mask = df_m['dateString'].between(start, end)
        df_m.loc[mask, 'Ind'] = np.nan

    data["Ind"] = df_m["Ind"]
    return data

def dataset_specific_operations_missing(csv_data,miss_data_pred_params, length_pred_params):

    for data in csv_data:
        print(data)
        subj = data.split("/")[-1]
        df = pd.read_csv(data,parse_dates = [0])
        try:
            df.rename(columns = {"dates":"dateString"}, inplace = True)
        except:
            pass
        df["Date"] = df["dateString"].dt.date
        days = df["Date"].unique()

        df_pred = pd.DataFrame()

        #Prediction is performed per day since simulation is for a day (i.e its not continuous)
        for day in days:
            print(day)
            df_subset = df[df["Date"] == day].copy()
            df_subset.reset_index(drop=True, inplace = True)

            df_subset = predict_missing_data(df_subset,miss_data_pred_params,length_pred_params)
            df_pred = pd.concat([df_pred,df_subset])
            
        #mask using indicator values
        df_pred["glucose_level"] = df_pred["glucose_level"] * df_pred["Ind"]

        #load cgmerror and preictederror datasets and mask them using the indicator variable
        #mask_error_datasets(subj,df_pred)

        #save file
        #output_dir = data_directory + simulation_type + "_" + dataset + "_" + mod + "/"
        output_dir = data_directory + simulation_type + "_" + dataset + "_" + mod +  "/"
        check_folder = os.path.exists(output_dir)
        if not check_folder:
            os.makedirs(output_dir)
        
        df_pred.to_csv(output_dir + subj,index = False)

def mask_error_datasets(subj,df):

    #load the datasets
    df_cgmerror = pd.read_csv(data_directory + "sim_adult_cgmerror/" + subj,parse_dates = [0])
    df_prederror = pd.read_csv(data_directory + "sim_adult_rct_prederror/" + subj,parse_dates = [0])

    #mask glucose values using missingness indicator
    df_cgmerror["glucose_level"] = df_cgmerror["glucose_level"] * df["Ind"]
    df_prederror["glucose_level"] = df_prederror["glucose_level"] * df["Ind"]

    #save for cgmerror
    output_dir_cgmerror = data_directory + "sim_adult_cgmerror_" + dataset + "_missing/" 
    check_folder = os.path.exists(output_dir_cgmerror)
    if not check_folder:
        os.makedirs(output_dir_cgmerror)
    
    df_cgmerror.to_csv(output_dir_cgmerror + subj, index = False)

    #save for predicted error
    
    output_dir_prederror = data_directory + "sim_adult_prederror_" + dataset + "_missing/"
    check_folder = os.path.exists(output_dir_prederror)
    if not check_folder:
        os.makedirs(output_dir_prederror)
    
    df_prederror.to_csv(output_dir_prederror + subj, index = False)
    
'''Functions related to Error Prediction'''
def group_into_daily(patient):
    '''
    Function to group a patient's readings into days

    patient: the patient's csv file
    '''
    #extract it into df per day
    patient.dateString = pd.to_datetime(patient.dateString)
    patient_daily = []
    for i in patient.groupby(patient.dateString.dt.floor('d')):
        patient_daily.append(i[1])
    return patient_daily

def error_extract_features(window_size, patient_daily, no_of_rows,cfg_file):
    '''
    Function to extract the tsfel features for each observation/row
    
    window_size: 60
    
    patient_daily_list: list containing array for each day's observations for the patient
    
    no_of_rows: total number of observation for the patient. 
    '''
    #cfg_file = tsfel.get_features_by_domain("statistical")

    X_train_total = pd.DataFrame()
    for i in range(len(patient_daily)): 
        print("Day: {}".format(i))
        per_day = patient_daily[i]
        per_day = per_day.reset_index()
        per_day = per_day.set_index('dateString')
        for j in range(len(patient_daily[i])):
            X_train = pd.Series([])
            #print("Index: {} || Day: {}".format(per_day['index'][j], i))
            t1 = per_day.index[j]
            t2 = t1 - pd.to_timedelta(window_size, unit='m')

            window = per_day[t2:t1]['glucose_level']
            #try:
            X_train = tsfel.time_series_features_extractor(cfg_file, window, n_jobs = 40, verbose = 0)
            #except:
               # print("Issues with: Index: {} || Day: {}".format(per_day['index'][j], i))
            X_train_total = X_train_total.append(X_train)
    X_train_total.index = range(0, no_of_rows)
    return X_train_total 

def error_predict_errors(X_all_test,model):
    '''
    Function to predict the CGM error
    '''
    if dataset == "race":
        scaler = pickle.load(open(model_path + "Results_error/race/scaler_racial.pkl",'rb'))
        X_all_test = pd.DataFrame(scaler.transform(X_all_test))
        X_all_test = X_all_test.fillna(0)
    #correlated_features = {'0_ECDF_9', '0_ECDF_6', '0_ECDF_8', '0_ECDF_2', '0_ECDF_7', '0_ECDF_1', '0_ECDF_3', '0_ECDF_5'}
    
    #X_all_test.drop(labels=correlated_features, axis=1, inplace=True)
    y_pred_2 = model.predict(X_all_test)
    print("Done")
    return y_pred_2

def retrieve_subset(patient, number_of_days):
    patient['dateString'] = pd.to_datetime(patient['dateString'])
    patient = patient[patient['dateString'] < patient['dateString'][0] + timedelta(days=number_of_days)]
    return patient

def dataset_specific_operations_error(csv_data):

    if dataset == "race":
        window_size = 180
    else:
        window_size = 60
    features = tsfel.load_json(model_path + dataset + "/features.json")
    load_model = pickle.load(open(model_path + dataset + '/error_model.pkl', 'rb'))

    for patient in csv_data:
        subj = patient.split("/")[-1]
        patient = pd.read_csv(patient,parse_dates = [0])
        n = 42 #number of days of data to predict error for
        patient = retrieve_subset(patient, n)
        patient_daily_list = group_into_daily(patient) #divide time series to per day
        X_features = error_extract_features(window_size,patient_daily_list,len(patient),features)

        patient['error'] = error_predict_errors(X_features,load_model)
        patient['glucose_level'] = patient['glucose_level'] + patient['error']

        #save file
        #output_dir = data_directory + simulation_type + "_" + dataset + "_" + mod + "/"
        output_dir = data_directory + simulation_type + "_" + dataset + "_" + mod + "B/"
        check_folder = os.path.exists(output_dir)
        if not check_folder:
            os.makedirs(output_dir)
        
        patient.to_csv(output_dir + subj,index = False)


if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        dataset = str(sys.argv[1])              #name of the dataset - OHIO, OAPS
        simulation_type = str(sys.argv[2])      #name of simulated data file e.g sim_adult_cgmerror
        data_directory = str(sys.argv[3])       #input dir of file - should contain one csv file per ID
        mod = sys.argv[4]                       #property to be predicted - missing, error
        model_path = str(sys.argv[5])           #path to where prediction models are stored

        if dataset != "race":
            hist_max = 12
        else:
            hist_max = 12  #4
        input_dir = data_directory  + simulation_type
        csv_data = glob.glob(input_dir + "/*.csv")

        n = 0
        if mod == "missing":
            miss_data_pred_params, length_pred_params = miss_loadmodels(model_path)
            dataset_specific_operations_missing(csv_data,miss_data_pred_params,length_pred_params)                              #perform dataset specific operations
        elif mod == "error":
            dataset_specific_operations_error(csv_data)






























        '''
        for day in days:
            df_subset = df[df["Date"] == day].copy()
            df_subset.reset_index(drop=True, inplace = True)

            if "noise" in mod:
                if n == 0:
                    print("---------------Adding Gaussian Noise----------------")
                df_subset = gauss_noise(df_subset)      #noise model

            elif mod == "missing":
                if n == 0:
                    print("---------------Adding Predicted Missigness----------------")
                df_subset = predict_missing_data(df_subset)             #louis missing data module

            elif mod == "error":
                if n == 0:
                    print("---------------Adding Predicted Error----------------")
                df_subset = predict_error(df_subset)

            elif mod == "error_missing":
                if n == 0:
                    print("---------------Adding Predicted Missingness and Error----------------")
                df_err = predict_error(df_subset)
                df_miss = predict_missing_data(df_subset)

                df_err["Ind"] = df_miss["Ind"]
                df_err["glucose_level"] = df_err["Ind"] * df_err["glucose_level"]

                df_subset = df_err.copy(deep = True)

            
            df_mod = pd.concat([df_mod,df_subset])
            n = n + 1   
        
        df_mod.reset_index(drop=True, inplace = True)
        
        #save the data file
        output_dir = data_directory + dataset + "_results/" + simulation_type + "_" + mod + "/"
        check_folder = os.path.exists(output_dir)
        if not check_folder:
            os.makedirs(output_dir)
        df_mod.to_csv(output_dir + subj,index = False)
        
        '''

            
            

        
