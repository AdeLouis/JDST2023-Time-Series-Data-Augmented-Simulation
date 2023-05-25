
import pandas as pd
import numpy as np
from datetime import timedelta
import math
import random

import sys
import glob as glob
import os
from os import path
import tsfel
from joblib import Parallel, delayed, dump

from sklearn.ensemble import RandomForestRegressor
from collections import Counter

from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang

random.seed(10)
np.random.seed(10)

def expand_dates(df):
    '''Function to add the rows of missing data entries'''

    dates = list(df["dates"])
    i = 0
    j = len(dates) - 2

    new_dates = []
    while i < j:
        curr = dates[i]
        next = dates[i+1]
        val = next-curr

        if val >= timedelta(minutes = sampling_rate):
            val = val.total_seconds()
            val = val/(60 * sampling_rate) #minutes
            val = int(val - 1) #number of new times to add

            new_dates.append(curr)
            temp = curr
            for n in range(0,val):
                temp = temp + timedelta(minutes = sampling_rate)
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

def reformat_tsfel(b,tdata,test_df):

    tdata["ID"] = b
    tdata = tdata[["ID","glucose_level"]].copy()

    #resample to remove minor irregular timing
    sample = str(sampling_rate) + "T"
    tdata = tdata.resample(sample,origin = "start").mean()

    #linearly interpolate nan point in the middle
    tdata = tdata.interpolate(method = "linear",limit_direction="both")
    b = b + 1
    test_df = pd.concat([test_df,tdata])

    return test_df,b

def helper(df, var,feature = None):
    if var == "mask":
        value = list(df["glucose_level"])
        value = [0 if math.isnan(x) else 1 for x in value]                  #mask variables tells us whether a value is present or not
        df["Mask"] = value

    elif var == "label":
        value = list(df["glucose_level"])
        value = [1 if math.isnan(x) else 0 for x in value]                  #mask variables tells us whether a value is present or not
        df["Label"] = value

    elif var == "var-mask":
        value = list(df[feature])
        value = [1 if x > 0 else 0 for x in value] 
        df[feature + "_mask"] = value
        df.drop(columns = [feature], inplace = True)

    return df

def time_interval(subset):

    '''Function to create time interval values since last missing entry'''
    
    subset = helper(subset,"mask")
    first = subset.at[0,"dates"]
    subset["times"] = subset["dates"] - first
    subset['times'] = subset['times'].map(lambda a: (a.total_seconds()*0.1) / 60)

    m = list(subset["Mask"])
    s = list(subset["times"])

    time_interval = []
    for n in range(0,len(subset)):
        if n == 0:
            time_interval.append(0)
        elif m[n-1] == 1:
            time_interval.append(s[n] - s[n-1])
        elif m[n-1] == 0:
            past = time_interval[-1]
            time_interval.append(s[n] - s[n-1] + past)

    subset["time_interval"] = time_interval
    subset = subset.drop(columns = ["times"])
    return subset

def load_csv(subj):

    '''Function to load the csv file and do some quick processing like making sure all time stamps are consistent'''

    df = pd.read_csv(input_path + "/" + subj + ".csv",infer_datetime_format=True)
    df = expand_dates(df)

    df['glucose_level'].loc[df['glucose_level'] <= 15] = np.nan  #remove all glucose values below 15
    df = df[np.where(~df[['glucose_level']].isnull())[0][0]:]
    df.reset_index(drop = True, inplace = True)

    df = time_interval(df)
    df["hour"] = df["dates"].dt.hour
    df_other = df.copy(deep = True)

    mask = df.glucose_level.notna()
    a = mask.ne(mask.shift()).cumsum()
    df = df[(a.groupby(a).transform('size') < 1) | mask] 
  
    df.reset_index(inplace=True, drop = True)
    df["Times"] = df["dates"].diff()

    #this ensure we only look at gaps greater than the sampling rate as those are the missing data entries we care about
    if sampling_rate == 15:
        gaps = df[df["Times"] > '00:15:00']
    else:
        gaps = df[df["Times"] > '00:05:00']
    gaps.reset_index(drop = True, inplace = True)

    return gaps, df_other

def extract_missing_representation(subj_ids,dataset):

    '''Function to extract the time windows before each missing data interval'''

    test_df = pd.DataFrame()
    testy = []
    b = 1
    missing_data_rep = []

    max_missing = 120 #in minutes; Domian specific value

    for subj in subj_ids:
        print("-------------EXTRACT DATA FOR PATIENT: " + str(subj) + " -------------")
        df,df_def = load_csv(subj)
        df_def.set_index("dates", inplace = True)

        #get the time,length and id
        for n in range(0,len(df)):
            gap_length = (df.at[n,"Times"] - timedelta(seconds = sampling_rate * 60)).total_seconds() #in seconds
            gap_length = gap_length/60 #in minutes
            time_of_occurence = df.at[n,"dates"] - timedelta(seconds = gap_length * 60)
            
            if (gap_length > max_missing) or (gap_length < sampling_rate):                            #max missing is 2 hours, 
                pass                                                                                  #if the missing gap is larger, we ignore it
            else:
                n = history_window * 60 #in minutes
                start_window = time_of_occurence - timedelta(seconds = sampling_rate * 60) 
                end_window = time_of_occurence - timedelta(seconds = n * 60) 
                
                data = df_def.loc[end_window:start_window].copy()                                     #extract time window prior to the missing interval

                '''For tsfresh'''
                tdata = data.copy(deep = True)
                test_df,b = reformat_tsfel(b,tdata,test_df)                                           #reformats data to the format tsfel accepts
                data = data[["glucose_level","hour"]]
                data.reset_index(drop = False, inplace = True)
               
                missing_data_rep.append((subj,np.around(gap_length),time_of_occurence,data))
                testy.append(gap_length)

    test_df.reset_index(drop = False, inplace = True)

    dict = {'y':testy}  
    testy = pd.DataFrame(dict) 

    return test_df, testy

def length_of_interval_model(X,y):
    train_df,test_df,trainy,testy = unpack(X,y)
    df = pd.concat([train_df,test_df],axis = 0)
    y = np.concatenate((trainy,testy))

    #extract feature
    X = feature_extraction(df)
    X = X.to_numpy()

    model = RandomForestRegressor(n_jobs = 20,random_state = 10)
    weights = None
    weights = LDS(y)
    model.fit(X,y,sample_weight = weights)
    dump(model, output_path + 'length_tsfel_model.joblib')

def unpack(X,y):

    '''Unpack csv files and divide into train and test sets'''

    uniq_ids = X["ID"].unique()
    y["ID"] = uniq_ids
    y.reset_index(drop = False,inplace = True)

    #select ids for training
    train_ids = np.random.choice(uniq_ids, size = int(0.7*len(uniq_ids)), replace = False)
    test_ids = []
    for id in uniq_ids:
        if id not in train_ids:
            test_ids.append(id)

    train_df = X[X["ID"].isin(train_ids)].copy()
    test_df = X[X["ID"].isin(test_ids)].copy()
    trainy = y[y["ID"].isin(train_ids)].copy()
    testy = y[y["ID"].isin(test_ids)].copy()

    train_df.sort_values(by=["ID","dates"],ignore_index=True,inplace=True)
    test_df.sort_values(by=["ID","dates"],ignore_index=True,inplace=True)
    trainy.sort_values(by=["ID","index"],ignore_index=True,inplace=True)
    testy.sort_values(by=["ID","index"],ignore_index=True,inplace=True)

    trainy = trainy["y"].to_numpy()
    testy = testy["y"].to_numpy()

    return train_df,test_df,trainy,testy

def feature_extraction(df):

    def setup(id,X):
        temp = X[X["ID"] == id].copy()
        ts = list(temp["glucose_level"])
        print(id)

        try:
            stats_features = tsfel.time_series_features_extractor(stats_file,ts, window_size=len(ts),n_jobs = 1,verbose=0,overlap = 0)
        except:
            stats_features = pd.DataFrame()
        try:
            temporal_features = tsfel.time_series_features_extractor(temporal_file,ts, window_size=len(ts),n_jobs = 1,verbose = 0,overlap = 0)
        except:
            temporal_features = pd.DataFrame()

        return (stats_features,temporal_features,id)
    
    stats_file = tsfel.get_features_by_domain("statistical")
    temporal_file = tsfel.get_features_by_domain("temporal")   

    stats_df = pd.DataFrame()
    temporal_df = pd.DataFrame() 
    X = df.copy(deep = True)

    ids = list(df["ID"].unique())
    res = Parallel(n_jobs=100)(delayed(setup)(id,X) for id in ids)

    for group in res:
        stats_features, temporal_features,id = group
        if stats_features.empty:
            stats_features["ID"] = [id]
        else:
            stats_features["ID"] = id
        
        if temporal_features.empty:
            temporal_features["ID"] = [id]
        else:
            temporal_features["ID"] = id

        stats_df = pd.concat([stats_df,stats_features])
        temporal_df = pd.concat([temporal_df,temporal_features])

    stats_df.sort_values(by=["ID"],ignore_index=True,inplace=True)
    temporal_df.sort_values(by=["ID"],ignore_index=True,inplace=True)
    
    #joing dataframes together
    train = stats_df.set_index('ID').join(temporal_df.set_index('ID'))
    train.reset_index(drop = True, inplace = True)

    return train

def LDS(labels):

    '''Function to re-weight samples from minority class'''

    def get_lds_kernel_window(kernel, ks, sigma):
        assert kernel in ['gaussian', 'triang', 'laplace']
        half_ks = (ks - 1) // 2
        if kernel == 'gaussian':
            base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
            kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
        elif kernel == 'triang':
            kernel_window = triang(ks)
        else:
            laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
            kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

        return kernel_window

    def get_bin_idx(label):
        if label <= 0:
            print("Error")
            exit()
        
        bin_idx = label/5 #5 minutes is the sampling frequency
        bin_idx = int(bin_idx-1)
        
        return bin_idx

    # assign each label to its corresponding bin (start from 0)
    # with your defined get_bin_idx(), return bin_index_per_label: [Ns,] 
    bin_index_per_label = [get_bin_idx(label) for label in labels]

    # calculate empirical (original) label distribution: [Nb,]
    # "Nb" is the number of bins
    Nb = max(bin_index_per_label) + 1
    num_samples_of_bins = dict(Counter(bin_index_per_label))
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]

    # lds_kernel_window: [ks,], here for example, we use gaussian, ks=5, sigma=2
    lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=5, sigma=2)
    # calculate effective label distribution: [Nb,]
    eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')

    # Use re-weighting based on effective label distribution, sample-wise weights: [Ns,]
    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    #take the square root of the effective empircal values
    #eff_num_per_label = [np.minimum(np.sqrt(v),40) for v in eff_num_per_label]
    eff_num_per_label = [np.sqrt(v) for v in eff_num_per_label]

    #print(max(eff_num_per_label))
    #print(min(eff_num_per_label))

    weights = [np.float32(1 / x) for x in eff_num_per_label]

    #from other code base
    scaling = len(weights) / np.sum(weights)
    weights = [scaling * x for x in weights]

    return weights


if __name__ == "__main__":
    input_path = str(sys.argv[1])
    dataset = sys.argv[2]               #name of real dataset
    data_to_learn = sys.argv[3]         #Subject IDs to use for learning dataset properties
    sampling_rate = int(sys.argv[4])    #sampling rate of the dataset in minutes
    output_path = sys.argv[5]           #output folder

    if dataset == "race":
        history_window = 3 #Hours, 12 samples if data is sampled every 15 minutes
    else:
        history_window=1 # hours, 12 samples if data is sampled every 5 minutes

    df = pd.read_csv(data_to_learn)
    subjs_to_include = df.PID.to_list()
    
    X,y = extract_missing_representation(subjs_to_include,dataset)

    length_of_interval_model(X,y)

    
    


    

    
   
    

        



    
