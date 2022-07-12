
import pandas as pd
import numpy as np
from datetime import timedelta
import math

import sys
import glob as glob

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

    '''Function to load the csc file and do some quick processing like making sure all time stamps are consistent'''

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

    '''Function to extract the window before each missing data interval'''

    test_df = pd.DataFrame()
    testy = []
    b = 1
    missing_data_rep = []

    max_missing = 120 #in minutes. Domian specific value

    #extract the missing representations for them
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
                data = data[["glucose_level","hour"]]#,"Mask","time_interval",,"meal_mask"]]
                data.reset_index(drop = False, inplace = True)
               
                missing_data_rep.append((subj,np.around(gap_length),time_of_occurence,data))
                testy.append(gap_length)

    test_df.reset_index(drop = False, inplace = True)
    test_df.to_csv(output_path + dataset + "_tsfel_x.csv")

    dict = {'y':testy}  
    testy = pd.DataFrame(dict) 
    testy.to_csv(output_path + dataset + "_tsfel_y.csv")


if __name__ == "__main__":
    input_path = str(sys.argv[1])
    dataset = sys.argv[2]               #name of real dataset
    data_to_learn = sys.argv[3]         #IDs to use for. Should be a csv file
    sampling_rate = int(sys.argv[4])    #sampling rate of the dataset in minutes
    output_path = sys.argv[5]           #output folder

    history_window=1 #1 hours, 12 samples if data is sampled every 5 minutes

    df = pd.read_csv(data_to_learn)
    subjs_to_include = df.PID.to_list()
    
    extract_missing_representation(subjs_to_include,dataset)

    
    


    

    
   
    

        



    
