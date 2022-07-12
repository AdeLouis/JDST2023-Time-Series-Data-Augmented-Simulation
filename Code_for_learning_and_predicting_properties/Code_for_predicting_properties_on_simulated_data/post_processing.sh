
#!/bin/sh

encoding_dataset="ohio" #oaps
modification="missing" #missing,error,error_missing
simulation_type="sim_adult_ohio_errorB" #sim_adult, sim_adult_cgmerror
data_dir="/data/PHI/PHI_OAPS/sandbox/lgomez/Sim_Project/simulation/Simulated_data/"$encoding_dataset"_sim_results/"
root_dir="/data/PHI/PHI_OAPS/sandbox/lgomez/Sim_Project/simulation/"


if [ "$encoding_dataset" = "Sim" ]; then
    model_path="/"
fi

if [ "$encoding_dataset" = "oaps" ]; then
    model_path="/data/PHI/PHI_OAPS/sandbox/lgomez/Sim_Project/simulation/Real_data/"
fi

if [ "$encoding_dataset" = "rct" ]; then
    model_path="/data/PHI/PHI_OAPS/sandbox/lgomez/Sim_Project/simulation/Real_data/"
fi

if [ "$encoding_dataset" = "race" ]; then
    model_path="/data/PHI/PHI_OAPS/sandbox/lgomez/Sim_Project/simulation/Real_data/"
fi

if [ "$encoding_dataset" = "ohio" ]; then
    model_path="/data/PHI/PHI_OAPS/sandbox/lgomez/Sim_Project/simulation/Real_data/"
fi


python $root_dir"Code/simulation/post_processing_steps.py" $encoding_dataset $simulation_type $data_dir $modification $model_path