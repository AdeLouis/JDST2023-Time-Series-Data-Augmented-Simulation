Author: Louis Gomez
Date: July 12, 2022

*Missing Data*

preprocess_for_length_of_missing_data.py: script used to extract data used for predicting the length of missing intervals.
Output is fed into the next script

predict_length_of_missing_interval.py: script used to learn a model for predicting the length of missing data. Output is the 
model file (trained on all data), and RMSE results for leave one out cross validation

missing_prediction_utils.py: script used to store function used in the process of predicting the start of missing intervals

predict_start_of_missing_intervals.py: script to predict the start of missing intervals. Depending on type (LOPO or ALL), the output is the
results file or model files

