import sys
sys.path.append('/data/code/lgomez/T1D_opensource/simglucose/') #the code in OAPS is the same

#from Code.simulate_sequence_of_events import *
from simulate_sequence_of_events import generate_sequence
'''Simulator'''
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.controller.pid_ctrller import PIDController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient_v2 import T1DPatient
#from opensource_t1d.simglucose.simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim

'''Others'''
from datetime import timedelta
from datetime import datetime
import os
from collections import Counter
import random
import time

'''Parallelization'''
from joblib import Parallel, delayed

def setup_eachday(init_subj_name,start_time,dataset,n):

    if simulation_type == "sim_adult":
        subj_name = init_subj_name.split("-")[0]
    else:
        subj_name = init_subj_name

    # --------- Create Custom Scenario --------------
    # Create a simulation environment
    patient = T1DPatient.withName(subj_name)
    sensor = CGMSensor.withName('Dexcom', seed=1)
    pump = InsulinPump.withName('Insulet')
    dayid = str("_day") + str(n)

    error_ind = True

    while error_ind == True:
        # custom scenario is a list of tuples (time, meal_size)
        did_genmean_fail = True

        while did_genmean_fail == True:
            try:
                seq_of_events = generate_sequence(file_path + dataset + "_meal_information.pickle")
                did_genmean_fail = False
                #seq_of_events = [("meal",(7.5, 30), 45), ("meal",(12, 45), 70), ("meal",(18, 20), 80),("activity",(20,50),20)] #start time, duration,cho amount
                #seq_of_events = [("meal",7.5, 30, 45), ("meal",12, 45, 70), ("meal",18, 20, 80),("activity",20,50,120)] #start time, duration,cho amount
                #seq_of_events = [('meal', 1.0, 11, 51.0), ('meal', 1.2, 14, 35.0), ('meal', 1.5, 28, 30.0), ('meal', 9.3, 36, 20.0), ('meal', 14.2, 69, 20.0), ('meal', 20.1, 40, 3.0)]
            except:
                did_genmean_fail = True
                

        scenario = CustomScenario(start_time=start_time, scenario=seq_of_events)
        env = T1DSimEnv(patient, sensor, pump, scenario)

        # Create a controller
        controller = BBController()
        #controller = PIDController()

        # Put them together to create a simulation object
        s2 = SimObj(env, controller, timedelta(days=1), animate=False, path=save_folder)

        _,error_ind = sim(s2,subj_name,dayid,init_subj_name)


if __name__ == "__main__":
    t0 = time.time()
    if len(sys.argv) > 1:
        dataset = str(sys.argv[1])              #name of the dataset - OHIO, OAPS
        simulation_type = str(sys.argv[2])      #sim_standard, sim_adult
        simulation_days = int(sys.argv[3])      #number of days to simulate in days

        
        file_path = '/data/PHI/PHI_OAPS/sandbox/lgomez/Sim_Project/simulation/'

        if simulation_type == "sim_standard":
            type_of_subj = ["child","adolescent","adult"]
            all_subjs = []
            for type in type_of_subj:
                for x in range(1,10):
                    all_subjs.extend([type + "#00" + str(x)])
            all_subjs.extend(["child#010","adolescent#010","adult#010"])

        elif simulation_type == "sim_adult":
            type = "adult"
            all_subjs = []
            n = 1
            for x in range(1,10):
                all_subjs.extend([type + "#00" + str(x) + "-" + str(n)])
            all_subjs.extend([type + "#010" + "-" + str(n)])
            n = 2
            #all_subjs.extend(["adult#001" + "-" + str(n)])
            #all_subjs.extend(["adult#002" + "-" + str(n)])
            

        #for each subject, simulate a select number of days of data
        for subj_name in all_subjs:
            print(subj_name)
            save_folder = "/data/PHI/PHI_OAPS/sandbox/lgomez/Sim_Project/simulation/Simulated_data/Sim_results/" + dataset + "_" + simulation_type + "/" + subj_name
            #save_folder = "data/PHI/PHI_OAPS/sandbox/lgomez/Sim_Project/Simulated_data/" + dataset + '_results/' + simulation_type + "/" + subj_name 
            #save_folder = os.path.join(os.path.dirname(__file__), dataset + '_results/' + simulation_type + "/" + subj_name)
            
            # specify start_time as the beginning of today
            now = datetime.now()
            start_time = datetime.combine(now.date(), datetime.min.time())
            Parallel(n_jobs=60)(delayed(setup_eachday)(subj_name,start_time,dataset,n) for n in range(0,simulation_days))

     
    t1 = time.time()

    print(t1-t0)

 