#!/bin/bash
#
# Usage
# -----
# $ bash launch_experiments.sh ACTION_NAME
#
# where ACTION_NAME is either 'list' or 'submit' or 'run_here'

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi

file_pattern=*.csv

for filename in $file_pattern
do
    export input_file=$filename
    export gar_model="gar_models/${input_file::${#input_file}-4}.json"
    export ggp_model="ggp_models/${input_file::${#input_file}-4}.json"
    export perf_plot="performance/${input_file::${#input_file}-4}.png"
    export pred_plot="forecasts/${input_file::${#input_file}-4}.png"
    ## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file
    
    if [[ $ACTION_NAME == 'list' ]]; then
        echo $input_file
        echo $gar_model
        echo $ggp_model
        echo $perf_plot
        echo $pred_plot
        echo

    elif [[ $ACTION_NAME == 'submit' ]]; then
        ## Use this line to submit the experiment to the batch scheduler
        sbatch do_experiment.slurm
    
    elif [[ $ACTION_NAME == 'run_here' ]]; then
        ## Use this line to just run interactively
        bash do_experiment.slurm
    fi
done