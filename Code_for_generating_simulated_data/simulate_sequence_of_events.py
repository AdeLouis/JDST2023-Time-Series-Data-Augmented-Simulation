
'''Code used to generate a sequence of events for simulating data'''

import pickle
import numpy as np
import pandas as pd
import sys
from collections import Counter
import random

def convert_to_probability(meal_info,tag):
    if tag == "meals":
        meal = Counter(meal_info)
        meal = {k:v for k,v in meal.items() if k < 6}
        temp = list(meal.values())
        prob = temp / np.sum(temp)
        keys = list(meal.keys())

        meal = dict(zip(keys, prob))
        return meal

    else:
        meal = Counter(meal_info)
        meal = {k:v for k,v in meal.items()}
        temp = list(meal.values())
        prob = temp / np.sum(temp)
        keys = list(meal.keys())

        meal = dict(zip(keys, prob))
        return meal

def cselect_hours(num_of_meals,hours_list):
    
    def mini_fn(meal_hours,meal_def):
        count = 0
        meal_hour = meal_def -1                                           #set initial value same last meal value
        while (meal_def >= meal_hour) or (count > 20):                 #exit while loop when next meal is greater than previous meal
            meal_hour = np.random.choice(list(meal_hours),size = 1)[0]  #select meal time
            count = count + 1
            
            if count > 20:
                meal_hour = meal_def

        return meal_hour
    
    first_meal_hours = hours_list[0]
    first_meal_hour = np.random.choice(list(first_meal_hours),size=1)[0]  #get first meal
    hours_of_meals = [first_meal_hour]
    
    meal_def = first_meal_hour
    
    #get hours for other meals
    for n in range(num_of_meals-1):
        meal_hours = hours_list[n+1]
        meal_hour = mini_fn(meal_hours,meal_def)
        
        meal_def = meal_hour                                            #update meal times
        hours_of_meals.append(meal_hour)
        
    return hours_of_meals

def process_meals(v,num_of_daily_meals,meal_sizes):
    #Distribution of meal sizes for num of meals per day

    meal_sizes_dict = {}

    for n in v.keys():
        ind_match_meal_num = [i for i, j in enumerate(num_of_daily_meals) if j == n]      #index for number of meals is n
        n_meal_sizes = [meal_sizes[i] for i in ind_match_meal_num]                        #list of list of meal sizes for each day
        all_meals_sizes = np.concatenate(n_meal_sizes).ravel()                            #array of all meals concatenated
        all_meals_sizes = [100 if x > 100 else x for x in all_meals_sizes]
        
        n_meal_sums = [np.sum(x) for x in n_meal_sizes]                             #list of daily carbs consumed
        mean_sum = np.mean(np.asarray(n_meal_sums))                                 #avg daily carbs
        std_sum = np.std(np.asarray(n_meal_sums))                                   #sd of daily carbs consumed
        
        meal_sizes_dict[n] = (mean_sum,std_sum,all_meals_sizes)

    return meal_sizes_dict

def process_hours(v,num_of_daily_meals,meal_times):
    #Distribution of hours for each meal per daily count of meals
    hours_dict = {}
    meal_hour_category = {}

    #def categorize():

    for n in v.keys():
        ind_match_meal_num = [i for i, j in enumerate(num_of_daily_meals) if j == n]      #index for number of meals is n
        n_meal_hours = [meal_times[i] for i in ind_match_meal_num] 
        all_meals_hours = np.concatenate(n_meal_hours).ravel()   

        hours_dict[n] = (all_meals_hours)

    return hours_dict  

def select_hours(all_meal_hours, n_meals):

    def mini_select_hours(all_meal_hours):
        get_times = convert_to_probability(all_meal_hours,"hour")
        meal_hour = np.random.choice(list(get_times.keys()),p = list(get_times.values()),size = 1)[0]
        mins = np.random.choice(vals, size = 1)[0]
        meal_time = meal_hour + mins    
        
        #sample duration
        dura = np.random.normal(loc = 45,scale = 15,size = 1)[0] #assume duration is a normal guassian , this should create a min of 1 and max of 90
        return meal_time, dura

    early_morn = [0,1,2,3,4,5]
    breakfast = [6,7,8,9,10]
    lunch = [11,12,13,14,15]
    dinner = [16,17,18,19,20]
    late_night = [21,22,23]

    vals = np.linspace(0,0.9) 
    meal_hours = []

    if n_meals == 1: #randomly sample a meal time and duration
        meal_time, dura = mini_select_hours(all_meal_hours)
        meal_hours.append((meal_time,dura))

    elif n_meals == 2:
        #get the first meal hours information
        meal_time, dura = mini_select_hours(all_meal_hours)
        meal_hours.append((meal_time,dura))

        end_window = meal_time + (dura/60) + 2 #meal time + duration in hours + 1 hour interval between meals
        hours_to_exclude = np.arange(int(meal_time)-2,int(end_window)+1,1) #if meal_time is 14.20, ref is 17.30, hours to exclude is 14,15,16,17
        subset_hours = [x for x in all_meal_hours if x not in hours_to_exclude]
        meal_time, dura = mini_select_hours(subset_hours)
        meal_hours.append((meal_time,dura))

    elif n_meals >= 3:
        end_times = []
        #first get the three main meals of brekfast, lunch and dinner
        for period in [breakfast,lunch,dinner]:
            checker = 1
            subset_hours = [x for x in all_meal_hours if x in period]

            if len(meal_hours) == 0:
                meal_time, dura = mini_select_hours(subset_hours)
                meal_hours.append((meal_time,dura))
                end_times.append((int(meal_time),int(meal_time + (dura/60) + 0.5)))     #records the time ranges of meal in hours

            else:
                end_window = meal_hours[-1][0] + (meal_hours[-1][1]/60) + 1.5    #get the end time of the last meal + no meal window
                while checker == 1:
                    meal_time, dura = mini_select_hours(subset_hours)
                    if end_window > meal_time:                                   #we make sure that the meals dont overlap
                        checker = 1
                    else:
                        meal_hours.append((meal_time,dura))
                        checker = 0
                        end_times.append((int(meal_time),int(meal_time + (dura/60) + 0.5)))

        n_meals = n_meals - 3   #for the rest of the meals, we reduce the duration
        
        #create a list of hours to exclude from further meal selection
        exclude_times = [np.arange(x,y+1,1) for x,y in end_times]
        exclude_times = np.concatenate(end_times).ravel().tolist()
        subset_hours = [x for x in all_meal_hours if x not in exclude_times]

        for n in range(0,n_meals):
            #exclude hours where meals already are present
            meal_time,_ = mini_select_hours(subset_hours)
            dura = np.random.normal(loc = 15,scale = 5)
            meal_hours.append((meal_time,dura))
            exclude_times = np.arange(int(meal_time),int(meal_time + dura)+1,1)
            subset_hours = [x for x in subset_hours if x not in exclude_times]

    
    return meal_hours
    
def generate_sequence(dataset_pickle):

    data = pd.read_pickle(dataset_pickle)
    meal_sizes, meal_times, num_of_daily_meals = data
    
    '''Part A: Select the number of meals per each simulated day'''
    v = convert_to_probability(num_of_daily_meals,"meals")
    meal_num = np.random.choice(list(v.keys()),p = list(v.values()),size = 1)[0] #part a result here

    #print(meal_num)

    '''Part B: Select meal sizes'''
    meal_sizes_dict = process_meals(v,num_of_daily_meals,meal_sizes)                    #process meal information
    mean_sum,std_sum,all_meals_sizes = meal_sizes_dict[meal_num]
    carbs_consumed_in_day = np.random.normal(mean_sum,std_sum)                          #assume meal size is gaussian and sample carbs consumed in a day                        
    rand_meal_sizes = np.random.choice(all_meals_sizes,size = meal_num)                 #array of n meals
    norm_rand_meal_sizes = (rand_meal_sizes/np.sum(rand_meal_sizes))  #array of n meals normalized

    if meal_num <= 4:
        rand_meal_sizes = norm_rand_meal_sizes * carbs_consumed_in_day
    else:
        temp = 0.5
        rand_meal_sizes = norm_rand_meal_sizes ** (1/temp)
        rand_meal_sizes = rand_meal_sizes / np.sum(rand_meal_sizes)
        rand_meal_sizes = rand_meal_sizes * carbs_consumed_in_day

    rand_meal_sizes = [min(100,x) for x in rand_meal_sizes]         #max meal size is 100g of carbs
    rand_meal_sizes = [max(1,x) for x in rand_meal_sizes]           #min meal size is 1g of carbs

    #print(rand_meal_sizes)

    '''Part C: Select the hour of meals per day'''
    hours_dict = process_hours(v,num_of_daily_meals,meal_times)
    all_meal_hours = hours_dict[meal_num]  
    meal_hours = select_hours(all_meal_hours,meal_num)                                                             
   
    meal_tag = ["meal"] * meal_num
    event_seq = list(zip(meal_tag,meal_hours,rand_meal_sizes))

    #print(event_seq)

    '''Part D: Add discrete exercise activity'''
    #the model cannot handle continuous changes in heart rate, so we have to keep this discrete time intervals
    #assume three activity periods
    prob_activity = [0.5,0.5,0.5]
    for p in prob_activity:
        res = np.random.binomial(n = 1, p = p)
        if res == 1:  # add an activity
            hr_increase = np.random.normal(loc = 45,scale = 10)
            d = np.linspace(1,45)
            duration = np.random.choice(d,size = 1)[0]
            h = np.linspace(0,24)
            hour = np.random.choice(h,size = 1)[0]

            event_seq.append(("activity",(hour,duration),hr_increase))

    #check that all activities are sorted in time sequentially, sort by hour
    event_seq = sorted(event_seq, key = lambda x: x[1][0])
    return event_seq



if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_pickle = str(sys.argv[1])

        #meal_info = pd.read_pickle(dataset_pickle)
        #meal_info = pickle.load(open(dataset_pickle, "rb" ))
        seq = generate_sequence(dataset_pickle)