# covid19-forecasting

Simulation software for forecasting demand at various stages of hospitalization

PI: Michael C. Hughes

## Usage

### Getting Started

```console
$ conda activate semimarkov_forecaster
$ python run_forecast.py \
    --config_file workflows/example_simple/params.json \
    --output_file /tmp/results-101.csv \
    --random_seed 101
```

**Expected output:**
```console
----------------------------------------
Loaded SemiMarkovModel from config_file:
----------------------------------------
State #0 Presenting
    prob. 0.100 recover
    prob. 0.900 advance to state InGeneralWard
State #1 InGeneralWard
    prob. 0.100 recover
    prob. 0.900 advance to state OffVentInICU
State #2 OffVentInICU
    prob. 0.100 recover
    prob. 0.900 advance to state OnVentInICU
State #3 OnVentInICU
    prob. 0.100 recover
    prob. 0.900 advance to state TERMINAL
random_seed=101 <<<
----------------------------------------
Simulating for 120 timesteps with seed 101
----------------------------------------
100%|██████████████████████████████████████████████████████████████████████████████| 120/120 [00:00<00:00, 148.03it/s]
----------------------------------------
Writing results to /tmp/results-101.csv
----------------------------------------
```


### With Snakemake for reproducibility

Run the following, which will install all necessary python packages in a separate environment, and then run a single simple simulation with results saved to file `results.csv`

```
$ snakemake --use-conda --cores 1 run_simple_example_simulation
```

## Summary

`params.json` specifies the parameters for a "semi-Markov" model of patient progression through the hospital system.

At each timestep, a patient can be described by:
* a binary health state ('Recovering' or 'Declining')
* an ordinal location state (e.g. 'Presenting', 'InGeneralWard', 'InICU', 'OnVentInICU')

We take an initial population, and run the model forward for a desired number of days.

## Install

#### 1. Install Anaconda

Follow the instructions here: <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>

#### 2. Install Snakemake

```
$ conda install -c bioconda -c conda-forge snakemake-minimal
```
Having trouble? See the full install instructions: <https://snakemake.readthedocs.io/en/stable/getting_started/installation.html>
