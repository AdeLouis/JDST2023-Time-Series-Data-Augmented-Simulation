Author: Louis Gomez
Date: July 12, 2022

Code in this folder is used to generate simulated data from the diabetes simulators

generate_meal_info_distribution.py: script used to generate what the distribution of meals looks like from
datasets that contain meal time and size information. This is a template on OAPS and can be modified for other datasets

simulate_sequence_of_events.py: script used for generating a sequence of events. Events are meals and activity

simulate_dataset.py: script used to generate simulated data. This calls the simulator stored on the server

unpack_combine_simulated.py: combine each simulated subjects data into one csv file and performs some pre-processing on them