#Utils for forecasting

import numpy as np
import pandas as pd
import pickle
import os
import random
from datetime import timedelta
import sys
import glob as glob
from joblib import Parallel, delayed


def check_length(subjs,dataset):
    '''Function to check the length of each ID file. Excludes IDs with not enough data'''
    subj_to_keep = []

    for subj in subjs:
        df = pd.read_csv(data_directory + "/" + subj + ".csv",infer_datetime_format=True)
        df['datetime']= pd.to_datetime(df['datetime'])
        
        first_date = df.at[0,"datetime"]
        last_date = df["datetime"].iloc[-1]
        duration = (last_date-first_date).days

        if duration > no_days:
            subj_to_keep.append(subj)
        else:
            print("Excluding subject: ",subj)

    return subj_to_keep

def rolling_window(ts, window, stride):
    shape = ts.shape[:-1] + (int((ts.shape[-1] - window)/stride + 1), window)
    strides = (stride*ts.strides[-1],) + (ts.strides[-1],)
    return np.lib.stride_tricks.as_strided(ts,shape=shape,strides=strides)

def expand_dates(df):
    dates = list(df["dates"])
    i = 0
    j = len(dates) - 2

    if dataset == "race":
        min_val = 15
    else:
        min_val = 5

    new_dates = []
    while i < j:
        curr = dates[i]
        next = dates[i+1]
        val = next-curr

        if val >= timedelta(minutes = min_val):
            val = val.total_seconds()
            val = val/(min_val*60) #minutes
            val = int(val - 1) #number of new times to add

            new_dates.append(curr)
            temp = curr
            for n in range(0,val):
                temp = temp + timedelta(minutes = min_val)
                new_dates.append(temp)
            new_dates.append(next)
        else:
            new_dates.append(curr)
            new_dates.append(next)  
        i = i + 1

    new_dates.append(dates[-1])
    new_dates = list(np.sort(list(set(new_dates))))
    new_df = pd.DataFrame(data = {"dates": new_dates})
    df.set_index("dates",inplace = True)
    new_df.set_index("dates",inplace = True)
    df = df.loc[~df.index.duplicated(keep='first')]
    df = pd.concat([new_df,df],axis = 1)

    df.reset_index(drop = False, inplace = True)
    return df

def quick_ops(df,feature,kind,dataset_type,dataset):

    if dataset == "race":
        sample_rate = "15T"
        interp =  1 #1 making it the same as below
    else:
        sample_rate = "5T"
        interp = 5

    if kind == "mask":
        #Remove rows for which cgm value is missing (nan)
        mask = df[feature].notna()
        a = mask.ne(mask.shift()).cumsum()
        df = df[(a.groupby(a).transform('size') < 1) | mask]
        df.reset_index(inplace=True,drop = True)

    elif kind == "resample":
        #df.set_index("dates", inplace = True)
        df = df.resample(sample_rate, origin = "start").mean()
        df.reset_index(drop = False, inplace = True)

    elif kind == "interpolate":
        mask = df.copy()
        grp = ((mask.notnull() != mask.shift().notnull()).cumsum())
        grp["ones"] = 1
        mask[feature] = (grp.groupby(feature)['ones'].transform('count') <= interp) | df[feature].notnull()
        df[feature] = df[feature].astype('float32')

        if dataset_type == "train":
            df[feature] = df[feature].interpolate()[mask[feature]]
        else:
            df[feature] = df[feature].ffill()[mask[feature]] #forward filling if test set

    return df

def process_data(df,dataset_type,encoding_dataset):

    if encoding_dataset == "race":
        num_samples_per_hour = 4 #4 should be three hour for race again here
        filter_window = 1 #1 making it the same for the bottom
    else:
        num_samples_per_hour = 12
        filter_window = 5
    
    df['glucose_level'].loc[df['glucose_level'] <= 15] = np.nan
    try:
        df = df[np.where(~df['glucose_level'].isnull())[0][0]:] 
    except:
        all_subj_keys = None
        return all_subj_keys

    #step 1: perform masking
    tdf = df.copy(deep = True)
    df.set_index("dates", inplace = True)
    feature = "glucose_level"
    temp_df = quick_ops(tdf,feature,"mask",dataset_type,encoding_dataset)

    #step 2: Get the time difference of success rows
    temp_df["Time_diff"] = temp_df['dates'].diff()
    gaps = temp_df[temp_df["Time_diff"] > '00:30:00']
    indexes = list(gaps.index)

    start_index = 0
    key = 0
    all_subj_keys = []
    
    #step 3: window data into subsets based on gap length
    for i in range(len(indexes)+1):
        if i < len(indexes):
            temp = temp_df.iloc[start_index:indexes[i]]
            firsttime_point = temp.iloc[0].dates
            lasttime_point = temp.iloc[-1].dates
            subset_df = df.loc[firsttime_point:lasttime_point]
        else:
            temp = temp_df.iloc[start_index:]
            firsttime_point = temp.iloc[0].dates
            subset_df = df.loc[firsttime_point:]
            #remove the continuous nans at the end of the time series if present
            last_index = subset_df["glucose_level"].last_valid_index()
            subset_df = subset_df[firsttime_point:last_index]
            
        if len(np.where(~subset_df['glucose_level'].isnull())[0]) > 0:
                subset_df = subset_df[np.where(~subset_df['glucose_level'].isnull())[0][0]:]
            
        if subset_df.shape[0] < 2*num_samples_per_hour:
            if i < len(indexes):
                start_index = indexes[i] #move on to the next index
            continue

        #Step 3A: resample the subset to 5mins
        #This is valid because irregular entries are averaged over
        
        subset_df = quick_ops(subset_df,feature,"resample",dataset_type,encoding_dataset)
        
        #Step 3B: Perform interpolation or forward filling
        subset_df = quick_ops(subset_df,feature,"interpolate",dataset_type,encoding_dataset)

        #Step 3C: Perform Filtering if indicated
        if dataset_type == 'train':
            subset_df['glucose_level'] = sp.signal.medfilt(subset_df['glucose_level'].values,filter_window) #median filtering

        if i < len(indexes):
            start_index = indexes[i] #move on to the next index

        if subset_df.shape[0] > 2 * num_samples_per_hour:
            all_subj_keys.append(subset_df)
            key = key + 1

    return all_subj_keys

def get_data_splits(subj,data_directory,num_of_days,kind):
    '''Splits data into train and test sets'''

    if "sim" not in dataset:
        df = pd.read_csv(data_directory + "/" + subj + ".csv",infer_datetime_format=True)
        df['datetime']= pd.to_datetime(df['datetime'])
        df.rename(columns = {"datetime":"dates"},inplace = True)
        df.reset_index(drop = True, inplace = True)
        df = expand_dates(df)

       
    elif "sim" in dataset:
        df = pd.read_csv(data_directory + subj + ".csv",infer_datetime_format=True,parse_dates=[0])

        if (encoding_dataset == "ohio") and (experiment == 1): #take the first two weeks for forecasting
            first_date = df.at[0,"dateString"]
            two_week = first_date + timedelta(weeks = 2)
            mask = df["dateString"] < two_week
            df = df.loc[mask].copy(deep = True)
            
            last_idx = df["glucose_level"].last_valid_index()
            #print(transfromed_data.index)
            first = 0
            #print(first)
            df = df[first:last_idx]

    try:
        df.rename(columns = {"dateString":"dates"}, inplace = True)
    except:
        pass

    df = df[["dates","glucose_level"]].copy()
    df.reset_index(drop = True, inplace = True)

    if experiment == 1:
        if dataset in ["oaps","rct","race"]:
            first_date = df.at[0,"dates"]
            end_date = df["dates"].iloc[-1]
            duration = (end_date - first_date).days
            mid_point = duration //2
            subset_begin = df.at[0,"dates"] + timedelta(days = mid_point)
            subset_end = subset_begin + timedelta(days = num_of_days)

            #greater than the start date and smaller than the end date
            mask = (df['dates'] > subset_begin) & (df['dates'] <= subset_end)             
            subset_df = df.loc[mask].copy()
            subset_df.reset_index(drop = True, inplace = True)
            df = subset_df.copy(deep = True)

    #added today april 11 to get the same size dataset on sim data for experiment 2
    elif experiment == 2: 
        if kind == "train":
            #experiment = 2, take a subset of data that macthes the number of days of data
            first_date = df.at[0,"dates"]
            subset_end = first_date + timedelta(days = num_of_days)
            #greater than the start date and smaller than the end date
            mask = (df['dates'] >= first_date) & (df['dates'] <= subset_end)             
            subset_df = df.loc[mask].copy()
            subset_df.reset_index(drop = True, inplace = True)
            df = subset_df.copy(deep = True)

        #if (dataset in ["oaps","rct","ohio"]) and (encoding_dataset == "race"): 
        if encoding_dataset == "race":
            #convert real datsets to 15 minute intervals
            freq = '15T'
            df.set_index("dates", inplace = True)
            df = df.resample(freq).agg(dict(glucose_level='first'))#,basal="first",bolus="sum",meal='sum'))
            df.reset_index(drop=False, inplace = True)
            print(df)

    if experiment == 1:
        if encoding_dataset == "ohio":
            n_train = 0.65
            train_data = df.iloc[0:int(n_train*len(df))]
            test_data = df.iloc[int(n_train*len(df)):]
            test_data.reset_index(drop = True, inplace = True)
        else:
            n_train = 0.85   #70/15/15
            train_data = df.iloc[0:int(n_train*len(df))]
            test_data = df.iloc[int(n_train*len(df)):]
            test_data.reset_index(drop = True, inplace = True)

        return train_data, test_data

    elif experiment == 2:
        #train_data = df.copy(deep = True)
        return df
