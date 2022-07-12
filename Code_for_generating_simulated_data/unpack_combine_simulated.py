import sys
import numpy as np
import pandas as pd
import glob as glob
import os
from datetime import timedelta

def fix_basal_bolus(df):
    '''This function is used to seperate basal and bolus'''
    df["bolus"] = 0.0
    init_basal = df.at[0,"insulin"]
    bolus_times = df[df["insulin"] > init_basal].copy()

    for index,value in bolus_times.iterrows():
        df.at[index,"bolus"] = value["insulin"]
        df.at[index,"insulin"] = 0
    
    return df

def unpack(file, subject_path):
    df = pd.read_csv(subject_path + file, parse_dates=[0], infer_datetime_format=True)
    #extract individual basal and bolus columns
    df = fix_basal_bolus(df)
    df.rename(columns = {"Time": "dateString","CHO":"meal","BG":"glucose_level","insulin":"basal"}, inplace = True)
    #TO DO: add basal and bolus

    if type == "raw":
        df = df[["dateString","glucose_level","basal","bolus","meal","HR"]]
    else:
        df.drop(["glucose_level"],axis = 1,inplace = True)
        df.rename(columns = {"CGM":"glucose_level"},inplace = True)
        df = df[["dateString","glucose_level","basal","bolus","meal","HR"]]


    glu_level = df["glucose_level"].to_numpy()
    glu_level = [x for x in glu_level if x <= 15]

    if len(glu_level) > 0:
        print(file)

    #resample data to every x minutes
    df.set_index("dateString", inplace = True)
    if origin == "race":
        df = df.resample(freq).agg(dict(glucose_level='first',basal="first",bolus="sum",meal='sum',HR="mean"))
    else:
        df = df.resample(freq).agg(dict(glucose_level='mean',basal="first",bolus="sum",meal='sum',HR="mean"))

    df.reset_index(drop = False, inplace = True)
    df = df.iloc[:-1]               # i remove the last time because i want 00:00 - 23:55

    return df
    
def unpack_combine(sim_subj):
    print("------------Unpacking "+sim_subj + "------------")
    subject_path = input_folder + "/" + sim_subj + "/"
    #get all csv files assciated with subject
    data_paths = glob.glob(subject_path + "*.csv")
    data_files = [x.split(subject_path)[1] for x in data_paths]
    data_files = sorted(data_files)

    save_raw_simulated_data(data_files,subject_path)
    
def save_raw_simulated_data(data_files, subject_path):

    agg_data_per_subject = pd.DataFrame()
    i = 0
    for file in data_files:
        orig_df = unpack(file, subject_path)

        #perform the same set of operations for both missing and the raw simulated versions
        if i == 0:
            agg_data_per_subject = pd.concat([agg_data_per_subject,orig_df])
        else:
            #get the time of last time point and genrate new date range for a day
            last_time = agg_data_per_subject.dateString.iloc[-1] + timedelta(minutes=val)
            end_time = last_time + timedelta(days = 1) 
            last_time = last_time.date() #get only the date for date range
            end_time = end_time.date()
            new_date_range = pd.date_range(start = last_time,end = end_time, freq=freq)[:-1]

            #replace datestring with new date range
            orig_df["dateString"] = new_date_range
            agg_data_per_subject = pd.concat([agg_data_per_subject,orig_df])

        i = i + 1

    agg_data_per_subject.reset_index(drop = True, inplace = True)
    agg_data_per_subject.to_csv(output_folder + sim_subj + ".csv", index = 0)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        simulation_type = str(sys.argv[1])      #sim_standard, sim_adult
        data_directory = str(sys.argv[2])       #input folder where simulated data is stored
        type = str(sys.argv[3])                 #type: either raw (which is just BG) or cgmerror (which is BG with cgm error)
        origin = str(sys.argv[4])               #dataset of origin: ohio, race, rct, or oaps

        if origin == "race":
            freq = "15T" #15 minutes
            val = 15
        else:
            freq = "5T"
            val = 5
        
        #simulation_type = simulation_type.lower()

        input_folder = data_directory + "Sim_results/" + simulation_type
        if type == "raw":
            output_folder = data_directory + origin + "_sim_results/sim_adult/"
        else:
            output_folder = data_directory + origin + "_sim_results/sim_adult_cgmerror/"

        check_folder = os.path.exists(output_folder)
        if not check_folder:
            os.makedirs(output_folder)

        #gets the names of folder in a list -> ['adult#009-3', 'adult#010-3', 'adult#002-3'...]
        subject_list = sorted(os.listdir(input_folder))

        try:
            subject_list.remove('.DS_Store')
            subject_list.remove('._.DS_Store')
        except:
            pass

        print(subject_list)
        
        for sim_subj in subject_list:
            #for each subject, we want to access all the days of data
            unpack_combine(sim_subj)



            
            


        


        
        
