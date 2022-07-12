
import sys
sys.path.insert(0, '/data/PHI/PHI_OAPS/sandbox/lgomez/Sim_Project/simulation/Code/forecasting')

from get_dataset_for_forecasting import process_simulated_datasetv,process_series_to_supervised

import glob
import pandas as pd
import numpy as np
import math
import os
from os import path
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from datetime import timedelta

state_vector_length = 32 #after manual grid search
epochs = 50
batch_size = 248
activation = 'relu' #activation for LSTM and RNN
models = ['REG','TREE','RNN','LSTM']#,"SVR"]
#models = ["LSTM",'RNN']

def generate_dataset_agnostic_operations(operation,csv_data):

    def agnostic_operations(data,op):

        if encoding_dataset == "oaps":
            ratio = 8.0555 #avg missing percentage in the oaps dataset used for leanring missing data
        elif encoding_dataset == "rct":
            ratio = 5.6409
        elif encoding_dataset == "race":
            ratio = 0.786
        elif encoding_dataset == "ohio":
            ratio = 1.86

        if op == "dropout":
            n = len(data)
            to_dropout = int((ratio/100) * n)
            drop_indices = np.random.choice(data.index,to_dropout,replace = False)
            data.loc[drop_indices,'glucose_level'] = np.nan

            return data

        elif op == "noise":
            glu = data["glucose_level"].to_numpy()
            noise = np.random.normal(scale = 2,size = len(glu))  
            data["glucose_level"] = glu + noise

            return data

        elif op == "dropout_noise":
            #dropout
            n = len(data)
            to_dropout = int((ratio/100) * n)
            drop_indices = np.random.choice(data.index,to_dropout,replace = False)
            data.loc[drop_indices,'glucose_level'] = np.nan
            value = list(data["glucose_level"])
            value = [np.nan if math.isnan(x) else 1 for x in value]                  #mask variables tells us whether a value is present or not
            data["Mask"] = value

            #noise
            glu = data["glucose_level"].to_numpy()
            noise = np.random.normal(scale = 2,size = len(glu))  
            data["glucose_level"] = glu + noise

            #dropout_noise
            data["glucose_level"] = data["glucose_level"] * data["Mask"]

            return data

    all_subj_train, all_subj_test = {}, {}

    for data in csv_data:
        subj = data.split("/")[-1]
        print(subj)
        df = pd.read_csv(data,parse_dates = [0])
        df["Date"] = df["dateString"].dt.date

        tdf = df.copy(deep = True)
        transfromed_data = agnostic_operations(tdf,operation)
        transfromed_data.rename(columns = {"dateString":"dates"}, inplace = True)

        if encoding_dataset == "ohio":
            first_date = transfromed_data.at[0,"dates"]
            two_week = first_date + timedelta(weeks = 2)
            mask = transfromed_data["dates"] < two_week
            transfromed_data = transfromed_data.loc[mask].copy(deep = True)

        #save noise 
        #transfromed_data.to_csv(data_directory + "noise/" + subj + ".csv")

        last_idx = transfromed_data["glucose_level"].last_valid_index()
        #print(transfromed_data.index)
        first = 0
        #print(first)
        transfromed_data = transfromed_data[first:last_idx]
        
        #split into train and test
        if encoding_dataset == "ohio":
            n_train = 0.65
        else:
            n_train = 0.85 #70/15/15
        train_data = transfromed_data.iloc[0:int(n_train*len(transfromed_data))]
        test_data = transfromed_data.iloc[int(n_train*len(transfromed_data)):]
        test_data.reset_index(drop = True, inplace = True)

        all_subj_train,all_subj_test = process_simulated_datasetv(train_data,test_data,
                                        all_subj_train,all_subj_test,subj,encoding_dataset)

    
    return all_subj_train,all_subj_test

def CGM_Prediction(model_name,unpickled_train_data,unpickled_test_data,i):

    def train_model(model,train_X,train_y,val_X,val_y):
        
        if model_name in ["RNN","LSTM"]:
            early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
            history = model.fit(train_X, train_y, validation_data=(val_X, val_y), shuffle=False, epochs=epochs,batch_size=batch_size,verbose=1,callbacks=[early_stop])
            y_bar = model.predict(val_X)
        
        else:
            model.fit(train_X, train_y) #training the algorithm 
            
            y_bar = model.predict(val_X)
            history = {}  

        y_bar = [int(element) for element in y_bar]
        valScore = math.sqrt(mean_squared_error(y_bar, val_y))
        print('Validation RMSE: %.3f' % valScore) 
        return model, history, valScore

    def test_model(model,test_X,test_y):

        if model_name in ["RNN","LSTM"]:
            #[no. of samples, timestamps, no. of features]
            test_X = test_X.reshape((test_X.shape[0], history_window , n_features))
        else:
            test_X = test_X.reshape((test_X.shape[0],history_window))
        
        y_bar = model.predict(test_X)
        y_bar = [int(element) for element in y_bar]

        predicted_values = pd.DataFrame(list(zip(test_y,y_bar)),columns=['True','Estimated'])
        testScore = math.sqrt(mean_squared_error(y_bar, test_y))
        return testScore, predicted_values

    train_subjs = list(unpickled_train_data.keys())
    test_subjs = list(unpickled_test_data.keys())
    random.shuffle(train_subjs)

    testScores = list()
    valScores = list() #validation error
    subjects = list()
    model_train_history = list()
    model_val_history = list()
    
    train_X,train_y,Xval,yval = combine_data_acorss_subjects(train_subjs,unpickled_train_data)

    #[no. of samples, timestamps, no. of features]
    if model_name in ["RNN","LSTM"]:
        train_X = train_X.reshape((train_X.shape[0], history_window , n_features))
        Xval = Xval.reshape((Xval.shape[0],history_window,n_features))
        model = deepLearningModels(model_name,train_X, train_y)
    else:
        train_X = train_X.reshape((train_X.shape[0],history_window))
        Xval = Xval.reshape((Xval.shape[0],history_window))
        model = baselineModels(model_name)
    
    model,history,valScore = train_model(model,train_X,train_y,Xval,yval)
    
    all_subjs_predicted_values = {}
    for subj in test_subjs:
        print('----------Testing on subject: ',subj,'----------')
        df = unpickled_test_data[subj]
        test_X,test_y = df
        testScore, predicted_values = test_model(model,test_X,test_y)
        all_subjs_predicted_values[subj] = predicted_values 
        #print('Test RMSE: %.3f' % testScore) 
        testScores.append(testScore)              

    #results_df = pd.DataFrame(list(zip(subjects,valScores,testScores)),columns=['Subject','valRMSE','testRMSE'])
    results_df = pd.DataFrame(list(zip(test_subjs,testScores)),columns=['Subject','testRMSE'])
    results_df.sort_values(by=['Subject'], inplace = True)      
    return results_df,all_subjs_predicted_values

def make_directories(dataset,ops):

    if not path.exists(output_directory):
        os.mkdir(output_directory)

    if not path.exists(output_directory + dataset + "_" + ops + "/"):
        os.mkdir(output_directory + dataset + "_" + ops + "/")

    return output_directory + dataset + "_" + ops + "/"

def combine_data_acorss_subjects(train_subjs,unpickled_train_data):

    Xtrain = np.zeros((1,history_window,1))
    Xval = np.zeros((1,history_window,1))
    ytrain = np.zeros((1))
    yval = np.zeros((1))

    #if experiment == 1:
    for subj in train_subjs:
        print('----------Unpack subject: ',subj,'----------')
        df = unpickled_train_data[subj]
        trainx, trainy = df

        n_train = int(0.595*trainx.shape[0])
        trainx,valx = trainx[:n_train,:,:], trainx[n_train:,:,:]
        trainy,valy = trainy[:n_train], trainy[n_train:]

        Xtrain = np.concatenate((Xtrain,trainx))
        Xval = np.concatenate((Xval,valx))
        ytrain = np.concatenate((ytrain,trainy))
        yval = np.concatenate((yval,valy))

    Xtrain = Xtrain[1:]
    Xval = Xval[1:]
    ytrain = ytrain[1:]
    yval = yval[1:]

    return Xtrain,ytrain,Xval,yval

def deepLearningModels(model_name,X,y):
    model = Sequential()
    if model_name == 'LSTM':
        model.add(LSTM(state_vector_length, activation='relu', input_shape=(X.shape[1], X.shape[2])))
        
    elif model_name == 'RNN':
        model.add(SimpleRNN(state_vector_length, activation=activation, input_shape=(X.shape[1], X.shape[2])))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

    return model

def baselineModels(model_name):
    if model_name == 'REG':
        model = LinearRegression()
    elif model_name == 'SVR':
        model = SVR(cache_size=1000)
    elif model_name == 'TREE':
        model = RandomForestRegressor(n_jobs = 15)
    
    return model

def run_extraction(i,ops,csv_data):
    random.seed(i)
    np.random.seed(i)

    all_subj_train,all_subj_test = generate_dataset_agnostic_operations(ops,csv_data)
    windowed_train = process_series_to_supervised(all_subj_train,sampling_rate,samples_per_hour)
    windowed_test = process_series_to_supervised(all_subj_test,sampling_rate,samples_per_hour)

    return windowed_train,windowed_test

if __name__ == "__main__":
    if len(sys.argv) > 4:
        root_directory = sys.argv[1]
        data_directory = sys.argv[2]
        output_directory = sys.argv[3]
        dataset = sys.argv[4]
        ops = sys.argv[5]
        no_days = sys.argv[6]
        encoding_dataset = sys.argv[7]

        no_iterations = range(1)
        n_features = 1

        if encoding_dataset == "race":
            history_window= 4 #4 
            prediction_window=30
            sampling_rate = 15
            samples_per_hour = 4 #4
        else:
            history_window=12 
            prediction_window=30  
            sampling_rate = 5  
            samples_per_hour = 12    
        
        csv_data = glob.glob(data_directory + "/*.csv")
        store = []

        #newly added: 03/26
        #store, res = [], []
        #res = Parallel(n_jobs=3)(delayed(run_extraction)(iteration,ops,csv_data) for iteration in no_iterations)

        #for item in res:
        #    windowed_train,windowed_test = item
        #    store.append((windowed_train,windowed_test))

        
        for i in no_iterations:
            random.seed(i)
            np.random.seed(i)
            print("Iteration: {}".format(i))
            all_subj_train,all_subj_test = generate_dataset_agnostic_operations(ops,csv_data)
            windowed_train = process_series_to_supervised(all_subj_train,sampling_rate,samples_per_hour)
            windowed_test = process_series_to_supervised(all_subj_test,sampling_rate,samples_per_hour)

            store.append((windowed_train,windowed_test))
        
        np.random.seed()
        for model_name in models:
            overall_results = pd.DataFrame()
            results_directory = make_directories(dataset,ops)

            for i in no_iterations:
                print("Iteration: {}".format(i))
                windowed_train,windowed_test = store[i]
                results_df, all_subjs_predicted_values = CGM_Prediction(model_name,windowed_train,windowed_test,i)

                if i == no_iterations[0]:
                    overall_results = pd.concat([results_df, overall_results], axis=1)
                else:
                    overall_results = results_df.merge(overall_results, on='Subject', how='inner', suffixes=('_1', '_2'))

            overall_results['Mean Test RMSE']= overall_results.mean(axis=1) 
            overall_results['STD Test RMSE']= overall_results.std(axis=1)

            print(overall_results) 
            
            filename = results_directory + model_name + "_results.csv"
            overall_results.to_csv(filename)



