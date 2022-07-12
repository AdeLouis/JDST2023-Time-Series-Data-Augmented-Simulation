
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from datetime import timedelta
import random

state_vector_length = 32
epochs = 30
batch_size = 128

import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN,Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix, fbeta_score, recall_score, precision_score,average_precision_score, auc, roc_curve


'''Code For Processing data for predicting when missing data occurs'''


def load_csv(file,sampling_rate):
    '''Function used to lead the data and dividie it into subsets for extracting time windows'''

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

    def create_subsets(df):
        mask = df.glucose_level.notna()
        a = mask.ne(mask.shift()).cumsum()
        df = df[(a.groupby(a).transform('size') < 1) | mask] 
        
        df.reset_index(inplace=True, drop = True)
        df["Times"] = df["dateString"].diff()

        gaps = df[df["Times"] > '02:05:00']     #Currently the max missing window is 2 hours
        return gaps,df

    def expand_dates(df):
        
        dates = list(df["dateString"])
        i = 0
        j = len(dates) - 2

        new_dates = []
        while i < j:
            curr = dates[i]
            next = dates[i+1]
            val = next-curr

            if val >= timedelta(minutes = sampling_rate):
                val = val.total_seconds()
                val = val/(sampling_rate*60) #minutes
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
        new_df = pd.DataFrame(data = {"dateString": new_dates})
        df.set_index("dateString",inplace = True)
        new_df.set_index("dateString",inplace = True)
        df = df.loc[~df.index.duplicated(keep='first')]
        df = pd.concat([new_df,df],axis = 1)

        df.reset_index(drop = False, inplace = True)
        return df

    df = pd.read_csv(file,parse_dates = [0],infer_datetime_format = True)
    df['datetime']= pd.to_datetime(df['datetime'])
    
    #load the data and some processing
    try:
        df = df[["dateString","glucose_level","meal"]]
    except:
        df = df[["dateString","glucose_level"]]

    df = expand_dates(df)

    df['hour'] = df['dateString'].dt.hour
    df["dow"] = df["dateString"].dt.dayofweek
    df['glucose_level'].loc[df['glucose_level'] <= 15] = np.nan
    df = df[np.where(~df[['glucose_level']].isnull())[0][0]:]
    df.reset_index(drop = True, inplace = True)

    df = helper(df,"mask")                                  #create mask varibales to know when data is missing (from lipton and che paper)
    df = helper(df,"label")                                 #state labels, 1 when data is missing (minority class)
    try:
        df = helper(df,"var-mask","meal")                       #mask for meal parameter
    except:
        pass

    #split data into window chunks if missing interval is greater than two hours (24 samples)
    tdf = df.copy(deep = True)
    df.set_index("dateString", inplace = True)
    gaps,temp_df = create_subsets(tdf)
    indices = gaps.index.to_list()

    return indices,temp_df,df

def get_valid_subsets(indices,temp_df,df,dataset):
    #for each indices, extract the subset given
    #initial start is the time index at the top of the file
    start = 0
    all_subset = []

    def time_interval(subset):
        subset.reset_index(drop = False, inplace = True)
        first = subset.at[0,"dateString"]
        subset["times"] = subset["dateString"] - first
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

    for n in range(len(indices)+1):
        
        if n < len(indices):
            #first get subset data from dataframe thats been modified to exclude all nan values
            temp = temp_df.iloc[start:indices[n]]
            #get the first and last time
            first = temp.iloc[0].dateString
            last = temp.iloc[-1].dateString
            #get correct subset from original dataframe with nan values
            subset = df.loc[first:last]             
        else:
            temp = temp_df.iloc[start:]
            first = temp.iloc[0].dateString
            subset = df.loc[first:]
            #get the last valid value for cgm. This is needed because the ends are usually filled with nans
            #with no new cgm values
            last_idx = subset["glucose_level"].last_valid_index()    
            subset = subset[first:last_idx]
            
        if len(np.where(~subset['glucose_level'].isnull())[0]) > 0:
                subset = subset[np.where(~subset['glucose_level'].isnull())[0][0]:]

        #check if subset is large enough (thia assumes all data is sampled every five minutes)
        if dataset == "race":
            val = 4
        else:
            val = 12
        
        if subset.shape[0] < (val * 2): #+ operation_window:
            if n < len(indices):
                start = indices[n] #move on to the next index
            continue

        #Add time interval column
        subset = time_interval(subset)

        #subset = subset.drop(columns = ["Times"])
        all_subset.append(subset)

        if n < len(indices):
            start = indices[n]

    print("Number of subsets created is: ",len(all_subset))

    return all_subset

def create_window_sizes(sampling_rate,max_size):
    '''Create window size from min to max window size possible'''
    list_of_sizes_in_mins = []
    list_of_sizes_in_mins.append(sampling_rate)

    val = sampling_rate
    while val < max_size:
        val = val + sampling_rate
        list_of_sizes_in_mins.append(val)
        print(list_of_sizes_in_mins)

    list_of_sizes_in_mins = [int(x/sampling_rate) for x in list_of_sizes_in_mins]

    return list_of_sizes_in_mins

def run_for_all(all_subj,hist,num_features,dataset):

    #print("-------------Extracting PATIENT: " + str(subj) + " -------------")
    all_train_subj = all_subj.copy()

    all_train_subj = windows_data(all_train_subj,hist,"train",num_features,dataset)
    all_train_subj, validation_data = get_validation_data(all_train_subj,hist,num_features)
    validation_windows, validation_labels = validation_data

    all_train_subj,scaler = standardize(all_train_subj,"train",num_features)
    validation_windows= standardize(validation_windows,"test",num_features,scaler)
    validation_data = (validation_windows,validation_labels)

    return all_train_subj,validation_data,scaler

def run_for_lopo(subj,all_subj,subj_names,hist,num_features,dataset):

    '''This function is used to create the training and testong LOPO sets acorss all subjects we use'''

    print("-------------HELD OUTPATIENT: " + str(subj) + " -------------")
    test_subsets = all_subj[subj]                                                  #test data is the current subject

    train_names = subj_names.copy()                                                #exclude test data and extract the rest of the training data from subjects
    train_names.remove(subj)
    lopo_train_subj = all_subj.copy()
    del lopo_train_subj[subj]                                                       #delete the held out subject data
    lopo_test_subj = {}
    lopo_test_subj[subj] = test_subsets

    lopo_train_subj = windows_data(lopo_train_subj,hist,"train",num_features,dataset)
    test_windows,test_labels, = windows_data(lopo_test_subj,hist,"test",num_features,dataset)
    
    lopo_train_subj, validation_data = get_validation_data(lopo_train_subj,hist,num_features)
    validation_windows, validation_labels = validation_data
    
    lopo_train_subj,scaler = standardize(lopo_train_subj,"train",num_features)
    validation_windows = standardize(validation_windows,"test",num_features,scaler)
    validation_data = (validation_windows,validation_labels)
    test_windows = standardize(test_windows,"test",num_features,scaler)

    return subj,lopo_train_subj,validation_data,test_windows,test_labels,None
 
def get_validation_data(lopo_train,hist,num_features):
    '''Divide train data into train and validation'''

    train = {}
    #save validation in combined form already
    x_val = np.zeros((1,hist,num_features))         #define shape
    y_val = []

    for name,data in lopo_train.items():
        x_train,y_train = data

        #take 70% for train, rest for test
        num_of_train = int(len(x_train) * 0.7)

        val = x_train[num_of_train:,:,:]
        valy = y_train[num_of_train:]

        x_train = x_train[:num_of_train,:,:]
        y_train = y_train[:num_of_train]

        train[name] = (x_train,y_train)
        x_val = np.concatenate((x_val,val))
        y_val.extend(valy)

    x_val = x_val[1:]
    assert len(x_val) == len(y_val)
    y_val = np.asarray(y_val).reshape(len(y_val))  #convert to numpy array

    return train, (x_val,y_val)

def standardize(data,dataset,num_features,scaler = None,label_data = None):

    '''This function performs standardization/normalization of our datasets'''
    '''Learns metrics from train and applying to validation or test'''

    x = 0
    fill = -1
    
    xtrain_scale = np.zeros((1,num_features))
    
    lopo = {}
    if dataset == "train":
        for _,val in data.items():
            xtrain = val[0].copy()
            xtrain = xtrain.reshape((xtrain.shape[0]*xtrain.shape[1],xtrain.shape[2]))    #reformat the data for scaling
            xtrain_scale = np.concatenate((xtrain_scale,xtrain))                          #combine all from subjects in LOPO round
 
        xtrain_scale = xtrain_scale[1:]
        x_scaler = MinMaxScaler()
        x_scaler.fit(xtrain_scale)
        
        for name,val in data.items():  #apply scaling
            xtrain = val[0].copy()
            xtrain = xtrain.reshape((xtrain.shape[0]*xtrain.shape[1],xtrain.shape[2]))
            scaled_data = x_scaler.transform(xtrain)
            scaled_data = np.nan_to_num(scaled_data, nan = fill)
            xtrain = scaled_data.reshape((val[0].shape[0],val[0].shape[1],val[0].shape[2]))
            lopo[name] = (xtrain,val[1])

        return lopo,x_scaler

    else:
        x = data
        x_scaler = scaler

        #operations for test data (test data can be the held out subject or validation dataset)
        test = x.reshape((x.shape[0]*x.shape[1],x.shape[2]))
        test = x_scaler.transform(test)
        test = np.nan_to_num(test,nan = fill)
        test = test.reshape((x.shape[0],x.shape[1],x.shape[2]))

        return test

def windows_data(lopo_subsets,hist,dataset_type,num_features,dataset):

    def extract_windows(window,subset,type,stride):

        missing_data_ind = subset["Label"].to_numpy()

        if dataset in ["oaps","ohio"]:
            features = ["glucose_level","time_interval","meal_mask","hour","dow"]
            num_features = len(features)
        elif dataset in ["rct","race"]:
            features = ["glucose_level","time_interval","hour","dow"]
            num_features = len(features)

        subset = subset[features].to_numpy()
        data_store = np.zeros((1,window,num_features))
        label_store = np.zeros((1))
        n = len(subset) - window 
        i = 0

        while i < n:
            try:
                seq = subset[i:i+window]                    #[150,151,152,nan,154,160]
                label = missing_data_ind[i+window]          #[0] or [161] ; [1] or [nan]
                if len(seq) != window:
                    break
            except:
                pass

            i = i + stride
            if np.isnan(seq[:,0]).all():                    #if all input data for glucose is nan exclude time sequence
                continue
            
            if type == "missing":
                if ((label == 1) or (np.isnan(label))) and (np.isnan(seq[-1,0]) == False): #accept a sequence as a label for missing if the last entry is not nan
                    seq = seq.reshape(1,window,num_features)
                    data_store = np.concatenate((data_store,seq))
                    label = label.reshape(1)
                    label_store = np.concatenate((label_store,label))
            
            else:
                if ((label == 0) or (np.isnan(label) == False)) and (np.isnan(seq[-1,0]) == False):
                    seq = seq.reshape(1,window,num_features)
                    data_store = np.concatenate((data_store,seq))
                    label = label.reshape(1)
                    label_store = np.concatenate((label_store,label))

        data_store = data_store[1:]
        label_store = label_store[1:]

        return data_store, label_store

    stride = 12

    lopo = {}                                   #dictionary with key as test subj and values and train data
    for name, content in lopo_subsets.items():
        #print("Working on: {}".format(name))
        seq_windows = np.zeros((1,hist,num_features))     #create numpy array in shape of final sequence data
        labels = []

        for n in range(0,len(content)):                     
            subset = content[n]
            'missing'
            X_seq,y = extract_windows(hist,subset,"missing",stride=1) 
            seq_windows = np.concatenate((seq_windows,X_seq))
            labels.extend(y)

            'non-missing'
            X_seq,y = extract_windows(hist,subset,"non-misssing",stride) 
            seq_windows = np.concatenate((seq_windows,X_seq))
            labels.extend(y)

        seq_windows = seq_windows[1:]
        print(Counter(labels))

        assert len(seq_windows) == len(labels)
       
        if dataset_type == "train":
            #save result in dictionary with name as train subject id and value as tuple as x and y
            lopo[name] = (seq_windows,labels)
        else:
            #return (seq_windows,labels,subset_group)
            return (seq_windows,labels)

    return lopo   


'''Utils Code for Learning model to predict when missing data occurs'''

def setup_data(train_subjs, train_subjs_data,hist, num_features):
    print(hist)

    train_X = np.zeros((1,hist,num_features))
    train_y = np.zeros((1))
    
    for subj in train_subjs:
            
        df = train_subjs_data[subj]
        train_X_temp, train_y_temp = df 
        ytrain = np.asarray(train_y_temp).reshape(len(train_y_temp))
        train_X = np.concatenate((train_X,train_X_temp))
        train_y = np.concatenate((train_y,ytrain))

    #remove zeros at first index
    train_X = train_X[1:]
    #train_xvec = train_xvec[1:]
    train_y = train_y[1:]

    assert len(train_X) == len(train_y) #Make sure the dimension are valid
    return train_X,train_y

def generate_batches(varying_length_data):
    #This function is used to section the dataset into batches of specificed sizes for training
    new_data = []
    y_counts = []
    for fixed_length_data in varying_length_data:

        if len(fixed_length_data) == 2:
            x,y = fixed_length_data                                   #x is the features, y is the label
            y_counts.extend(y)                                        #get number of individuals samples per label for class reweighting
            i = 0
            while i < len(x):
                x_temp = x[i:i+batch_size]                            #section data into batches based on specified batch size for model training
                y_temp = y[i:i+batch_size]

                i = i + batch_size                                    #Move onto the next batch of data       
                new_data.append((x_temp,y_temp))

        else:
            x= fixed_length_data                                   #x is the features       
            i = 0
            while i < len(x):
                x_temp = x[i:i+batch_size]                         #section data into batches based on specified batch size for model training
                i = i + batch_size                                 #Move onto the next batch of data       
                new_data.append((x_temp,None))
    
    return new_data,y_counts

def helper_fn(trainy,valy):

    def get_weight(no_classes, samples_per_class, power):
        weights_for_samples = 1.0 / np.array(np.power(samples_per_class,power))
        print(weights_for_samples)
        weights_for_samples = weights_for_samples / np.sum(weights_for_samples) * no_classes

        weights = {0:weights_for_samples[0],1:weights_for_samples[1]}
        print(weights)
        return weights
    
    w = list(Counter(trainy).values())
    print(Counter(trainy))
    
    pos = min(w)
    neg = max(w)
    #base_auprc = pos/(pos+neg)
    
    output_bias = np.log([pos/neg])
    output_bias = tf.keras.initializers.Constant(output_bias)
    class_weight = get_weight(2,[max(w),min(w)],1) #1 is inverse, 0.5 is squared inverse
    
    val_weight = Counter(valy)
    neg = val_weight[0]
    pos = val_weight[1]
    ratio = 0.5+(neg/(neg+pos))

    return output_bias,class_weight,ratio

def get_threshold(xval,model):

    per_batch_predict, per_batch_true = [], []
    for i in range(0,len(xval)):
        xbatch = xval[i]
        results = model.predict(xbatch[0])
        results = results.reshape((len(results),))
        per_batch_predict.extend(results.tolist())
        per_batch_true.extend(xbatch[1])

    recal_res, precision_res,res  = [],[], []
    
    th_list = [0.5]
    ypred = [1 if pred > 0.5 else 0 for pred in per_batch_predict]
    res.append(fbeta_score(per_batch_true,ypred,beta = 1)) #1.5 for oaps
    recal_res = recall_score(per_batch_true,ypred)
    precision_res = precision_score(per_batch_true,ypred)

    auprc = average_precision_score(per_batch_true,per_batch_predict)
    fpr,tpr,_ = roc_curve(per_batch_true,per_batch_predict,pos_label=1)
    aucv = auc(fpr,tpr)
    
    max_value = max(res)                #max fbeta score
    max_index = res.index(max_value)    #index of max beta score

    data = {"th":th_list, "f1-score":res,"recall":[recal_res], "precision":[precision_res]}
    df = pd.DataFrame(data)
    df["AUPRC"] = auprc
    df["AUROC"] = aucv
    #df.to_csv("rctit.csv")
    #print("ratio: ",ratio)
    th = 0.5
    return 0.5,precision_res

def train_model(train_data,val_data,ycounts,num_features,valycounts):

    def data_generator(input):
        random.shuffle(input)
        #input is a list of tuples. Each tuple is a batch is data. Each batch can vary in size.
        n = 0
        while n < len(input):
            batch = input[n]
            n = n + 1
            yield batch[0],batch[1]

            if n >= len(input):
                n = 0

    #early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
    output_bias,class_weights,ratio = helper_fn(ycounts,valycounts)
    
    METRICS = [keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='prc', curve='PR'),
            keras.metrics.AUC(name="auc")]

    #define model
    model = Sequential()
    model.add(SimpleRNN(state_vector_length, activation='relu', input_shape=(None,num_features)))
    model.add(Dense(1,activation = "sigmoid",bias_initializer=output_bias))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=METRICS)

    model.summary()

    history = model.fit(data_generator(train_data),steps_per_epoch=len(train_data),epochs = epochs,verbose=1,
            validation_data = data_generator(val_data),validation_steps = len(val_data),
            callbacks=[early_stop],class_weight = class_weights)

    #plot training and validation performance
    #plot_perf(history)

    #get optimal threshold value that gives max average precision score
    th,p_05 = get_threshold(val_data,model)

    return model,th,p_05

def test_model(seq_data,model,subj,fbeta_threshold):

    def window_results(results,fbeta_threshold,true_labels,subj):

        metrics = ["precision","recall","tp","fn","tn","fp"] 
        df = pd.DataFrame()

        for window_size,predictions in results.items():
            metrics_dictionary = {}
            #apply threshold ot get final ypred
            ypred = [1 if pred > fbeta_threshold else 0 for pred in predictions]

            #compute confusion matrix
            tn, fp, fn, tp = confusion_matrix(true_labels[window_size],ypred).ravel()
            recall = tp/(tp + fn)
            precision = tp/(tp + fp)
            values = [precision,recall]

            #store results into dictionary
            for name,value in zip(metrics,values):
                metrics_dictionary[name+"_"+window_size] = [value]

            new_df = pd.DataFrame.from_dict(metrics_dictionary)

            df = pd.concat([df,new_df],axis = 1)
        df["LOPO_Subj"] = [subj]

        return df

    nmetrics = ["auc","prc","precision","recall"]
    per_batch_predict, per_batch_true = [], []
    metrics_dictionary = {}

    results_for_specific_windows = {}
    true_for_specific_windows = {}

    for i in range(0,len(seq_data)):
        xbatch = seq_data[i]
        results = model.predict(xbatch[0])

        results = results.reshape((len(results),))
        per_batch_predict.extend(results.tolist())
        per_batch_true.extend(xbatch[1])

        wsize = str(xbatch[0].shape[1])

        if wsize in results_for_specific_windows:
            curr = results_for_specific_windows[wsize]
            curr.extend(results.tolist())
            results_for_specific_windows[wsize] = curr

            bcurr = true_for_specific_windows[wsize]
            bcurr.extend(xbatch[1])
            true_for_specific_windows[wsize] = bcurr
        else:
            results_for_specific_windows[wsize] = results.tolist()
            curr = []
            curr.extend(xbatch[1])
            true_for_specific_windows[wsize] = curr

    "get results for each history size"
    window_df = window_results(results_for_specific_windows,fbeta_threshold,true_for_specific_windows,subj)

    #apply threshold ot get final ypred
    ypred = [1 if pred > fbeta_threshold else 0 for pred in per_batch_predict]

    #calculate base auprc
    w = list(Counter(per_batch_true).values())
    base_auprc = min(w)/(min(w)+max(w))

    #compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(per_batch_true,ypred).ravel()
    recall = tp/(tp + fn)
    precision = tp/(tp + fp)
    fpr,tpr,_ = roc_curve(per_batch_true,per_batch_predict,pos_label=1)
    auc = auc(fpr,tpr)
    prc = average_precision_score(per_batch_true,per_batch_predict)
    values = [auc,prc,precision,recall]

    #store results into dictionary
    for name,value in zip(nmetrics,values):
        metrics_dictionary[name] = [value]

    new_df = pd.DataFrame.from_dict(metrics_dictionary)
    new_df["LOPO_Subj"] = [subj]

    
    return new_df ,window_df ,base_auprc   