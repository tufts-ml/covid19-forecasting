'''
Usage
------
To run a simple example simulation
$ snakemake --cores 1 --use-conda run_simple_example_simulation

To run a specific random seed
$ snakemake --config random_seed=101 --profile profiles/mhughes_laptop/ single_simulation_following_template

To run with multiple random seeds (prespecified in a config file)
$ snakemake --profile profiles/mhughes_laptop/ many_simulations_following_template

Prerequisites
-------------
1) Need local install of anaconda
2) Need local install of Snakemake
See README.md for install instructions
'''

import os
import numpy as np

configfile: "config_simple_example.json"

rule run_simple_example_simulation:
    input:
        config_json="params_simple_example.json"

    output:
        output_csv="results.csv"

    conda:
        "semimarkov_forecaster.yml"

    shell:
        """
        python run_forecast.py \
            --config_file "{input.config_json}" \
            --random_seed 8675309 \
            --output_file "{output.output_csv}" \
        """


# Target rule to run multiple simulations saved to different CSV files
#
# Uses the 'template' defined below
rule many_simulations_following_template:
    input:
        ["results-random_seed={random_seed}.csv".format(random_seed=seed)
            for seed in range(config['min_random_seed'], config['max_random_seed'])]


# Target rule to run one specific simulation (with seed provided by manual command line input)
#
# Uses the 'template' defined below
rule single_simulation_following_template:
    input:
        "results-random_seed={random_seed}.csv".format(random_seed=config['random_seed'])


rule template:
    input:
        config_json="params_simple_example.json"

    output:
        output_csv="results-random_seed={random_seed}.csv"

    conda:
        "semimarkov_forecaster.yml"

    shell:
        """
        python run_forecast.py \
            --config_file "{input.config_json}" \
            --random_seed "{wildcards.random_seed}" \
            --output_file "{output.output_csv}" \
        """
