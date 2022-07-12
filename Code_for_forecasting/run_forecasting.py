
#datastructure packages
import numpy as np
import pandas as pd
from numpy import array

#file system packages
import sys
import os
from os import path
import glob
import warnings
warnings.filterwarnings('ignore')
import gzip

#datetime packages
import datetime
import time
import dateutil.parser
import pytz

#machine learning packages
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

#miscellaneous
import math
from joblib import dump, load
import json
import random
import pickle

#set which GPU to use
#Can be any number through 0 to 7
os.environ["CUDA_VISIBLE_DEVICES"] = '7'


#(default configuration for model)
state_vector_length = 32 #after manual grid search
epochs = 50
batch_size = 248
activation = 'relu' #activation for LSTM and RNN
poly_degree = 2 #degree of polynomial

models = ["REG","TREE",'RNN','LSTM']#,"SVR"]

def make_directories(model_name):
    #datetime_now = datetime.datetime.now()
    #datetime_now = datetime_now.strftime("%d-%m-%Y_%I-%M-%S_%p")
    if not path.exists(output_directory):
        os.mkdir(output_directory)

    if not path.exists(output_directory + dataset + "/"):
        os.mkdir(output_directory + dataset + "/")

    return output_directory + dataset + "/"

def get_data(dataset,no_days,mode="train"): 

    if experiment == 1:
        if dataset in ["oaps","race","ohio","rct"]:
            print('Getting data from ', data_directory + '\n')
            unpickled_train_data = pd.read_pickle(data_directory + dataset + "_" + str(history_window) + "_" + str(no_days) + "_windowed_train.pickle") #e.g. oaps_12_60_windowed_train.pickle
            unpickled_test_data = pd.read_pickle(data_directory + dataset + "_" + str(history_window) + "_" + str(no_days) + "_windowed_test.pickle")

        elif "sim" in dataset:
            print('Getting data from ', data_directory + '\n')
            unpickled_train_data = pd.read_pickle(data_directory + dataset + "_" + str(history_window) + "_" + str(no_days) + "_windowed_train.pickle") #e.g. oaps_12_60_windowed_train.pickle
            unpickled_test_data = pd.read_pickle(data_directory + dataset + "_" + str(history_window) + "_" + str(no_days) + "_windowed_test.pickle")

        return unpickled_train_data,unpickled_test_data

    else:
    
        if mode == "test":
            if encoding_dataset != "race":
                unpickled_train_data = pd.read_pickle(root_directory + "Real_data/" + dataset + "_" + str(history_window) + "_" + str(no_days) + "_windowed_test.pickle")
            else:
                unpickled_train_data = pd.read_pickle(root_directory + "Real_data/" + dataset + "_race_" + str(history_window) + "_" + str(no_days)+ "_windowed_test.pickle")
        else:
            if "sim" in dataset:
                unpickled_train_data = pd.read_pickle(data_directory + dataset + "_" + str(history_window) + "_" + str(no_days) + "_windowed_all.pickle")
            elif (dataset == "race") and (encoding_dataset == "race"):
                unpickled_train_data = pd.read_pickle(data_directory + dataset + "_race_" + str(history_window) + "_" + str(no_days) + "_windowed_train.pickle")
            else:
                unpickled_train_data = pd.read_pickle(data_directory + dataset + "_" + str(history_window) + "_" + str(no_days) + "_windowed_train.pickle")
        #test set is a list of unpickled_test-data
        #test_data = []
        #for data in ["oaps","rct","race"]:
        #    unpickled_test_data = pd.read_pickle(data_directory + data + "_" + str(history_window) + "_" + str(no_days) + "_windowed_all.pickle")
        #    test_data.append(unpickled_test_data)

        return unpickled_train_data

#initializes a deep learning model (LSTM or RNN)
def deepLearningModels(model_name,X,y):
    model = Sequential()
    if model_name == 'LSTM':
        model.add(LSTM(state_vector_length, activation='relu', input_shape=(X.shape[1], X.shape[2])))
        
    elif model_name == 'RNN':
        model.add(SimpleRNN(state_vector_length, activation=activation, input_shape=(X.shape[1], X.shape[2])))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

    return model

#initializes a baseline model (Linear regression, 2nd order polynomial regression, Random forest regressor or ensemble of all three)
def baselineModels(model_name):
    if model_name == 'REG':
        model = LinearRegression()
    elif model_name == 'SVR':
        model = SVR(cache_size=1000)
    elif model_name == 'TREE':
        model = RandomForestRegressor(n_jobs = 20)
    
    return model

def combine_data_acorss_subjects(train_subjs,unpickled_train_data,ind="null"):

    Xtrain = np.zeros((1,history_window,1))
    Xval = np.zeros((1,history_window,1))
    ytrain = np.zeros((1))
    yval = np.zeros((1))

    if experiment == 1:
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
    
    else:
        if ind == "test":
            for subj in train_subjs:
                print('----------Unpack subject: ',subj,'----------')
                df = unpickled_train_data[subj]
                trainx, trainy = df

                if len(trainx) == 0:
                    pass
                else:
                    Xtrain = np.concatenate((Xtrain,trainx))
                    ytrain = np.concatenate((ytrain,trainy))

            Xtrain = Xtrain[2:]
            ytrain = ytrain[2:]

            return Xtrain,ytrain

        else:
            for subj in train_subjs:
                print('----------Unpack subject: ',subj,'----------')
                df = unpickled_train_data[subj]
                trainx, trainy = df

                n_train = int(0.80*trainx.shape[0])
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

def CGM_Prediction(model_name,unpickled_train_data,unpickled_test_data,i):

    def train_model(model,train_X,train_y,val_X,val_y):
        
        if model_name in ["RNN","LSTM"]:
            early_stop = EarlyStopping(monitor='val_loss', patience=7, verbose=1)
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

def main_1():

    for model_name in models:

        overall_results = pd.DataFrame() #saving subject ID, test and validation RMSE for each iteration, overall mean RMSE and MAE 
        results_directory = make_directories(model_name)
        no_iterations = range(10)
        unpickled_train_data, unpickled_test_data = get_data(dataset,no_days)

        for i in no_iterations:
            print('Iteration #: ',i)
            results_df, all_subjs_predicted_values = CGM_Prediction(model_name,unpickled_train_data,unpickled_test_data,i)
            
            if i == no_iterations[0]:
                overall_results = pd.concat([results_df, overall_results], axis=1)
            else:
                overall_results = results_df.merge(overall_results, on='Subject', how='inner', suffixes=('_1', '_2'))
            

        overall_results['Mean Test RMSE']= overall_results.mean(axis=1) 
        overall_results['STD Test RMSE']= overall_results.std(axis=1)

        print(overall_results) 

        filename = results_directory + model_name +"_results.csv"
        overall_results.to_csv(filename)

def CGM_prediction_2(model_name,unpickled_train_data):

    def train_model(model,train_X,train_y,val_X,val_y):
        
        if model_name in ["RNN","LSTM"]:
            early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=1) #15
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
    random.shuffle(train_subjs)
    testScores = list()

    train_X,train_y,Xval,yval= combine_data_acorss_subjects(train_subjs,unpickled_train_data)

    if model_name in ["RNN","LSTM"]:
        train_X = train_X.reshape((train_X.shape[0], history_window , n_features))
        Xval = Xval.reshape((Xval.shape[0],history_window,n_features))
        model = deepLearningModels(model_name,train_X, train_y)
    else:
        train_X = train_X.reshape((train_X.shape[0],history_window))
        Xval = Xval.reshape((Xval.shape[0],history_window))
        model = baselineModels(model_name)
    
    model,history,valScore = train_model(model,train_X,train_y,Xval,yval)

    #for testing, we test on the real world datasets
    #RWD = ["oaps","rct","ohio"]#,'race']
    RWD = ["race"]
  
    for rwd in RWD:
        rwd_test = get_data(rwd,no_days,"test")
        rwd_X,rwd_y = combine_data_acorss_subjects(list(rwd_test.keys()),rwd_test,"test")

        testscore,_ = test_model(model,rwd_X,rwd_y)
        testScores.append(testscore)

    results_df = pd.DataFrame(list(zip(RWD,testScores)),columns=['RW-Dataset','testRMSE'])
    results_df.sort_values(by=['RW-Dataset'], inplace = True)
    return results_df

def main_2():

    for model_name in models:
        overall_results = pd.DataFrame() #saving subject ID, test and validation RMSE for each iteration, overall mean RMSE and MAE 
        results_directory = make_directories(model_name)
        no_iterations = range(10)
        print(no_days)
        unpickled_train_data= get_data(dataset,no_days)
        
        for i in no_iterations:
            print('Iteration #: ',i)
            results_df = CGM_prediction_2(model_name,unpickled_train_data)

            if i == no_iterations[0]:
                overall_results = pd.concat([results_df, overall_results], axis=1)
            else:
                overall_results = results_df.merge(overall_results, on='RW-Dataset', how='inner', suffixes=('_1', '_2'))

        overall_results['Mean Test RMSE']= overall_results.mean(axis=1) 
        overall_results['STD Test RMSE']= overall_results.std(axis=1)

        print(overall_results) 

        filename = results_directory + model_name +"_results.csv"
        overall_results.to_csv(filename)


if __name__ == "__main__":
    if len(sys.argv) > 4:
        root_directory = sys.argv[1]
        data_directory = sys.argv[2]
        output_directory = sys.argv[3]
        dataset = sys.argv[4]
        no_days = sys.argv[5]
        experiment = int(sys.argv[6])
        encoding_dataset = sys.argv[7]

        n_features = 1
        if encoding_dataset == "race":
            sampling_rate = 15
            history_window = 4 #4
            prediction_window = 30
        else:
            sampling_rate = 5
            history_window = 12
            prediction_window = 30

        PH = str(prediction_window) #prediction horizon
        if prediction_window == 30 or prediction_window == 60:
            prediction_window = prediction_window//sampling_rate

        if experiment == 1:
            main_1()
        else:
            print(no_days)
            #Experiment B where we train on simulated dataset and test on real data
            if "sim" in dataset:
                output_directory = root_directory + "Simulated_data/"+ encoding_dataset + "_results_Expr_race/"
            else:
                output_directory = root_directory + "Real_data/Results_Expr_B/"
            main_2()
        