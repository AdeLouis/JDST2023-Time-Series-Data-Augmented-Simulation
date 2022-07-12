'''This file is used to generate meal information
    in terms of meal sizes and meal times from
    diabetes datasets to inform distribution for 
    simulation'''


import sys

import numpy as np
import pandas as pd
import glob as glob
import pickle
from collections import Counter,OrderedDict

def get_meal_information(file):
    '''This function is used to gather meal size and time information'''
    df = pd.read_csv(file, index_col = 0,parse_dates = [0],infer_datetime_format = True)
    df.reset_index(drop = False, inplace = True)
                                                            
    date_column = df.columns[0]                             #the first columns should be datetimeindex
    df['hour'] = df['dateString'].dt.hour
    df["day"] = df["dateString"].dt.date
    
    df = df[[str(date_column)] + ["meal","hour","day"]]
    df_meal = df[df["meal"] != 0].copy()              #get only rows where meals are recorded
    
    #divide and process per day here

    days = list(df["day"].unique())
    num_of_meals, meal_sizes, meal_times = [],[],[]
    for day in days:
        df_day = df_meal[df_meal["day"] == day].copy()

        num = len(df_day)
        if num == 0:
            pass
        else:
            num_of_meals.append(num)
            meal_sizes.append(df_day["meal"].to_numpy())
            meal_times.append(list(df_day["hour"]))

    
    print(len(num_of_meals))

    return (meal_sizes,meal_times,num_of_meals)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_folder = str(sys.argv[1])

        #get the data files
        file_paths = glob.glob(data_folder + "*.csv")

        all_meal_size, all_meal_times, all_meals_per_day = [], [], []
        for file in file_paths:
            subj_meal_size, subj_meal_time, subj_meal_count = get_meal_information(file)
            all_meal_size.extend(subj_meal_size)                            #add all infomation to the same list
            all_meal_times.extend(subj_meal_time)
            all_meals_per_day.extend(subj_meal_count)

        
        #save results as a pickle file
        #this information will be used later to generate different meal sizes and meal times
        #meal sizes will be fitted to a guassian distribution and meal times a discrete distribution
        with open('OAPS_meal_information.pickle', 'wb') as f:
            pickle.dump((all_meal_size,all_meal_times,all_meals_per_day), f, pickle.HIGHEST_PROTOCOL)
        
        print("here")

        


