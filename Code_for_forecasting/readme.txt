Author: Louis Gomez
Date: July 12, 2022

The code in this folder is related to the task of glucose forecasting and includes both files for preparing the data 
(like creating time windows) and also performing forecasting.

get_dataset_for_forecasting.py: script used for prep files for forecasting, outputs a pickle for train and test

run_forecasting.py: script used to run forecasting for both experiments

loop_run_forecasting.py: used to run forecasting on datasets that need to be re-started every iteration like random-dropout
and gaussian noise

combine_results.py: used to combine forecasting all forecasting results into one main csv file (for expr 1 )