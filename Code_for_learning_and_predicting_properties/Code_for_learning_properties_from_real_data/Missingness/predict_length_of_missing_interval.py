
import pandas as pd
import numpy as np
import random
import sys
import os
from os import path
import tsfel
from joblib import Parallel, delayed

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from collections import Counter
import math

from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang

from joblib import dump
from sklearn.model_selection import LeaveOneOut

random.seed(10)
np.random.seed(10)

def LDS(labels):

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


if __name__ == "__main__":
    input_path = str(sys.argv[1])    #input path to where the tsfel csv for each datasets are stored
    dataset = sys.argv[2]            #datasets name
    lds_ind = int(sys.argv[3])       # indicator to use LDS (a weighting method) 1 or 0
    output_path = sys.argv[4]
    mode = sys.argv[5]

    if dataset == "race":
        majority_label = 15
    else:
        majority_label = 5

    if not path.exists(output_path + dataset + "/"):
        os.mkdir(output_path + dataset + "/")
    output_path = output_path + dataset + "/"

    #if dataset == "ohio" and mode == "extract":
    X = pd.read_csv(input_path + dataset + "_tsfel_x.csv",index_col = 0)
    y = pd.read_csv(input_path+ dataset + "_tsfel_y.csv",index_col = 0)

    train_df,test_df,trainy,testy = unpack(X,y)
    df = pd.concat([train_df,test_df],axis = 0)
    y = np.concatenate((trainy,testy))

    #extract feature
    X = feature_extraction(df)
    X = X.to_numpy()

    '''Model A: train on all the data for simulated data prediction'''
    #run model on all data for simulated data prediction
    model = RandomForestRegressor(n_jobs = 20,random_state = 10)
    weights = None
    weights = LDS(y)
    model.fit(X,y,sample_weight = weights)
    dump(model, output_path + 'length_tsfel_model.joblib')


    '''Model B: LOco train and test'''
    #Perform Leave one sample out here and average rmse performance
    loo = LeaveOneOut()
    rmse_all = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #perform feature selection on traning dataset

        model = RandomForestRegressor(n_jobs = 20,random_state = 10)
        weights = None
        weights = LDS(y_train)
        model.fit(X_train,y_train,sample_weight = weights)

        y_bar_rf = model.predict(X_test)
        rmse = np.round(math.sqrt(mean_squared_error(y_bar_rf, y_test)),3)
        print(rmse)
        rmse_all.append(rmse)

    print("RMSE: {}".format(np.mean(np.asarray(rmse_all))))

