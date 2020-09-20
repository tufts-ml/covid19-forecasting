#!/bin/bash
#SBATCH -n 1                # Number of cores
#SBATCH -t 3-00:00          # Runtime in D-HH:MM
#SBATCH -p preempt          # Partition to submit to
#SBATCH --mem-per-cpu 2000  # Memory (in MB) per cpu
#SBATCH -o output/log_%j.out       # Write stdout to file named log_JOBIDNUM.out in current dir
#SBATCH -e output/log_%j.err       # Write stderr to file named log_JOBIDNUM.err in current dir
#SBATCH --export=ALL        # Pass any exported env vars to this script and its children

# Load the right conda environment
source /cluster/home/alee30/miniconda3/bin/activate spr_2020s_env

# Pass along all ENV variables as arguments to my Python script
python grid_search.py \
    $input_file \
    -m $gar_model \
    -f $ggp_model \
    -p $perf_plot

python run_simple_forecast.py \
    $input_file \
    -m $gar_model \
    -f $ggp_model \
    -p $pred_plot

conda deactivate