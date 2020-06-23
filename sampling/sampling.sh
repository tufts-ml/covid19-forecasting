#!/bin/bash
#SBATCH --account=normal
#SBATCH --job-name=s250
#SBATCH --output=logs/s250.log
#SBATCH --error=logs/s250.err
#SBATCH --time=3-00:00:00
#SBATCH -p batch
#SBATCH -N 2
#SBATCH -c 2
#SBATCH --mem=16g
#SBATCH --mail-type=END
#SBATCH --mail-user=gian_marco.visani@tufts.edu
source activate /cluster/tufts/hassounlab/gvisan01/enzyme_promiscuity/gian
python sampling.py --algorithm abc --num_iterations 250 --output_file example_output/s250.txt --start_epsilon 150 --annealing_constant 0.997 --lowest_epsilon 25