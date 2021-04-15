# ACED-HMM - A model for COVID-19 hospitalized patient trajectories

Our proposed mechanistic model can be used to forecast how a hospitalized COVID-19 patient can move through various stages of care (in general ward, in ICU, in ICU on the ventilator)

Using this model, we can:

* **fit parameters** to aggregated daily count time-series from a specific hospital site or region
* **forecast** future daily counts to help administrative officials understand future demand for resources
* **assess the societal value** of possible interventions (e.g. would decreasing admissions by X% help California avoid a lockdown in late 2020?)

See our preprint manuscript:

    Approximate Bayesian Computation for an Explicit-Duration Hidden Markov Model of COVID-19 Hospital Trajectories.
    Gian Marco Visani, Alexandra Hope Lee, Cuong Nguyen, David M. Kent, John B. Wong, Joshua T. Cohen, and Michael C. Hughes
    [TODO arXiv link here]

Jump to: [Usage](#usage) - [Modeling](#modeling) - [Installation](#installation)

For questions or concerns about the code, please [report an Issue](https://github.com/tufts-ml/aced-hmm-hospitalized-patient-trajectory-model/issues)

### Contact:

For questions or concerns about the code, please [report an Issue](https://github.com/tufts-ml/aced-hmm-hospitalized-patient-trajectory-model/issues)

For general questions, please email [Prof. Michael C. Hughes](https://www.michaelchughes.com) - mike (AT) michaelchughes.com


# Usage

### Getting Started

Here's a very simple example, that will run our probabilistic progression model (with dummy initial conditions and dummy parameters) to forecast ahead for 120 days. (Requires you have already [installed this project's conda environment](#installation)

```console
$ conda activate aced_hmm
$ python -m aced_hmm.run_forecast --func_name python --config_path workflows/example_simple/config.json --output_dir workflows/example_output --output_file results-{{random_seed}}.csv --approximate None --random_seed 1001 --num_seeds 10
```

**Expected output:**
```console
Forecasting with fixed parameters ...
Using 10 random seeds to generate simulations using the following parameters:
State #0 InGeneralWard
    prob. 0.100 recover
    prob. 0.900 advance to state OffVentInICU
    Recovering Duration: mean 1.9 | 1st% 1.0   10th% 1.0   90th% 4.0   99th% 5.0
    Declining Duration:  mean 1.9 | 1st% 1.0   10th% 1.0   90th% 4.0   99th% 5.0
State #1 OffVentInICU
    prob. 0.100 recover
    prob. 0.900 advance to state OnVentInICU
    Recovering Duration: mean 1.9 | 1st% 1.0   10th% 1.0   90th% 4.0   99th% 5.0
    Declining Duration:  mean 1.9 | 1st% 1.0   10th% 1.0   90th% 4.0   99th% 5.0
State #2 OnVentInICU
    prob. 0.100 recover
    prob. 0.900 advance to state TERMINAL
    Recovering Duration: mean 1.9 | 1st% 1.0   10th% 1.0   90th% 4.0   99th% 5.0
    Declining Duration:  mean 1.9 | 1st% 1.0   10th% 1.0   90th% 4.0   99th% 5.0
--------------------------------------------
           Running 10 simulations
--------------------------------------------
100%|██████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 15.41it/s]

```

This will write 10 CSV files to workflows/example_output, each identified by a specific random seed. Each file will have columns for each census count and a row for each day.

See an example output in (example_output/](.example_output/)

<!-- ### Using Snakemake workflows for reproducibility

Run the following, which will install all necessary python packages in a separate environment, and then run a single simple simulation with results saved to file `results.csv`

```
$ cd /path/to/covid19-forecasting/workflows/simple_example
$ snakemake --use-conda --cores 1 run_simple_example_simulation
```

### Using Snakemake workflows on the Tufts HPC cluster

If you are in the hugheslab group and have access to the HPC cluster, you can 

PREREQUISITE bashrc settings:
```
export PATH="/cluster/tufts/hugheslab/miniconda2/bin:$PATH"
```

Then login to the HPC system and do:
```
$ conda activate semimarkov_forecaster
$ pushd /cluster/tufts/hugheslab/code/covid19-forecasting/workflows/simple_example/
$ snakemake --cores 1 run_simple_example_simulation # Do NOT use '--use-conda' here, you already have the environment
``` -->


# Installation

#### 1. Install Anaconda

Follow the instructions here: <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>

#### 2. Install Snakemake

```
$ conda install -c bioconda -c conda-forge snakemake-minimal
```
Having trouble? See the full install instructions: <https://snakemake.readthedocs.io/en/stable/getting_started/installation.html>

#### 3. Install `aced_hmm` conda environment

Use the project's included YAML file to specify all packages needed: [aced_hmm.yml](./aced_hmm.yml)
```
conda env create -f aced_hmm.yml
```

#### 4. Setup the Cython extensions (optional)

```
$ cd aced-hmm-hospitalized-patient-trajectory-model
$ python setup.py # builds the Cython extensions!
```

# Modeling

We have developed a probabilistic "semi-Markov" model to simulate individual patient trajectories through the major stages or levels of care within the hospital (present with symptoms, general ward, ICU, ICU with mechanical ventilation). When entering a stage, the patient first draws a new health status (recovering or declining), and then based on this status samples a “dwell time” duration (number of days to remain at current care stage) from a status-specific distribution. After the dwell time expires, recovering patients improve and leave the model, while declining patients progress to the next stage.

At each timestep, a patient can be described by:
* a binary health state ('Recovering' or 'Declining')
* an ordinal location state (e.g. 'InGeneralWard', 'OffVentInICU', 'OnVentInICU')
* the time left before transition to the next location state

Every parameter governing these distributions can be specified by the user, and all are readily estimated from local data or the literature (e.g. by counting the fraction of ventilator patients who recover).

We take an initial population, and run the model forward for a desired number of days.

By reading parameters in from a plain text file [example](./workflows/example_simple/params.json), the model transparently facilitates communication of assumptions and invites modifications.
