# ACED-HMM - A model for COVID-19 hospitalized patient trajectories

Our proposed mechanistic model can be used to forecast how a hospitalized COVID-19 patient will move through various stages of care (in general ward, in ICU, in ICU on the ventilator). 

Using this model, we can:

* **fit parameters** to aggregated daily count time-series from a specific hospital site or region
* **forecast** future daily counts to help administrative officials understand future demand for resources
* **assess the societal value** of possible interventions (e.g. would decreasing admissions by X% help California avoid a lockdown in late 2020?)

The model and its fitting and forecasting procedures satisfy two desired properties:

* We focus on **probabilistic** modeling using a Bayesian approach, in order to capture and communicate uncertainty about the varied possible outcomes. We estimate distributions over all quantities of interest.
* To be **portable** to health systems around the world, we assume access only to aggregated daily counts of resource usage (number of occupied beds in general ward, in ICU, on ventilator). No patient-specific data (demographics, comorbidities, length-of-stay, etc.) are used.

See our preprint manuscript:

Gian Marco Visani, Alexandra Hope Lee, Cuong Nguyen, David M. Kent, John B. Wong, Joshua T. Cohen, and Michael C. Hughes. <i>Approximate Bayesian Computation for an Explicit-Duration Hidden Markov Model of COVID-19 Hospital Trajectories.</i>. Technical Report, 2021. <a href="https://www.michaelchughes.com/papers/VisaniEtAl_arXiv_2021.pdf">https://www.michaelchughes.com/papers/VisaniEtAl_arXiv_2021.pdf</a>

Now also available on arXiv preprint server: <https://arxiv.org/abs/2105.00773>

See also this slide deck overview: <https://docs.google.com/presentation/d/1MLkQLV8a5w1o6hBdfa8-b1J0DKh13JHWyVhEo8ddyB8>

Jump to: [Usage](#usage) - [Modeling](#modeling) - [Installation](#installation) - [ABC](#ABC)

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

# Installation

#### 1. Install Anaconda

Follow the instructions here: <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>

#### 2. Install `aced_hmm` conda environment

Use the project's included YAML file to specify all packages needed: [aced_hmm.yml](./aced_hmm.yml)
```
conda env create -f aced_hmm.yml
```

#### 3. Setup the Cython extensions (optional)

```
$ cd aced-hmm-hospitalized-patient-trajectory-model
$ python setup.py build_ext --inplace # builds the Cython extensions!
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

# ABC

We provide here an example workflow, together with an overview of the necessary commands, to fit the posterior of our model parameters to true counts via ABC, and to run forecasts from such posterior. To get started, we recommend running each command with the default parameters, which will "fit" the parameters to the MA data we used in our experiments (only for 25 timesteps, NOT until convergence), and then generate forecasts from the "fitted" posterior.  
In the results folder, we also provide posteriors already fitted to 4 US states and 2 UK hospitals.

## Setup

Requirements:  
- A json file containing the prior over each set of parameters  
- A json file containing the config information necessary for the hospital model to run a simulation. This includes:  
    - Initial counts at each state of the hospital  
    - Number of timesteps to simulate (training + testing)  
    - Number of *training* timesteps only  
    - A source to draw admissions from, usually a pointer to a file containing admissions per timestep  
    - Parameters of the model. Values are just placeholders, with the exception of *proba_Die_after_Declining_OnVentInICU* which must be set to 1.0 and never changed  
- A csv file containing admissions per timestep. These admissions an be at any state of the hospital  
- A csv file containing the true hospital census counts for training and testing

## Data collection

We provide the datasets we used in our experiments, as well as code to automatically collect and format US state level data from HHS and from the Covid Tracking Project.  
Code and README are provided in the folder datasets/US.

## Step 0: Cython

We provide a cython implementation of our model, which grants a 2-fold speedup over our optimized python implementation. Follow the [installation](#installation) instructions to enable the use of cython for fitting the posterior using ABC, and for running forecasts.
To use the cython implementation, set the argument *func_name* to *cython* in any script that has the argument.

## Step 1: Run ABC

Run the following command:  
```console
$ python -m aced_hmm.fit_posterior_with_abc
```

See the source code at aced_hmm/fit_posterior_with_abc.py for an explanation of the arguments.

Output:  
- Samples file (json), containing the last 2000 samples of parameters as a list of dictionaries.  
- Config file (json), a copy of the input config file, with also a pointer to the samples file
- Stats file (csv) containing a history of all relevant statistics at each step of the sampling procedure (epsilon trace, accepted distances, all distances, accepted alphas, all alphas). Useful to check convergence of the algorithm (code to automatically analyze this coming soon).

## Step 2: run forecasts (training + testing)

Run the following command:  
```console
$ python -m aced_hmm.run_forecast --func_name [python] \
                                  --config_path [results/US/MA-20201111-20210111-20210211/config_after_abc.json] \
                                  --output_dir [results/US/MA-20201111-20210111-20210211/individual_forecasts] \
                                  --output_file [results_after_abc-{{random_seed}}.csv] \
                                  --approximate [5] \
                                  --random_seed [1001] \
                                  --num_seeds [None]
```

Arguments:  
- *config_path* is the config file for the model. It must contain a pointer to a JSON file containing model parameters with which to forecast. These parameters can be provided either as single float values, in which case forecasts are made with fixed parameters and changing random seeds, or as parallel lists of float values (i.e. samples), in which case forecasts are made with unique pairs of samples and random seeds.
- *approximate* number of patients modeled jointly. Higher approximation means less granularity in forecast, but much higher speedup. We recommend the use of higher approximations for higher admissions regimes, where the speedup is much needed and the granularity can be lower without reducing the quality of the forecasts.  
- *output_file*: must include '-{{random_seed}}' at the end of your desired output file to allow the scripts to generate and then identify the individual forecasts. The script will generate *num_seeds* csv files, each containing one forecast.  
- *num_seeds*: if None, it defaults to 1 if forecasting with fixed parameters, and to number of samples if forecasting from samples.

We already provide a set of samples from the posterior distribution of 4 US states, trained from Nov 11th to Jan 11th, and of 2 UK hospitals, trained from Nov 3rd to Jan 3rd. To make forecasts with the MA posterior, simply set *config_path* to 'results/US/MA-20201111-20210111-20210211/PRETRAINED_config_after_abc.json'.

We **strongly** recommend making forecasts using samples from multiple runs of ABC, as opposed to just one.

## Step 3: compute summaries of forecasts

Run the following command:  
```console
$ python -m aced_hmm.summarize_forecasts --input_dir [results/US/MA-20201111-20210111-20210211/individual_forecasts] \
                                         --output_dir [results/US/MA-20201111-20210111-20210211] \
                                         --output_template [summary_after_abc_] \
                                         --input_csv_file_pattern [results_after_abc*.csv] \
                                         --comma_sep_percentiles [1,2.5,5,10,25,50,75,90,95,97.5,99]
```

Arguments:  
- *input_csv_file_pattern* identifies all the files containing the forecasts.

The summaries include (one file for each one): mean, stddev, and all percentiles specified by *comma_sep_percentiles*.

## Step 4: compute relevant metrics

Run the following command:  
```console
$ python -m aced_hmm.abc_test_metrics --input_dir [results/US/MA-20201111-20210111-20210211] \
                                      --output_dir [results/US/MA-20201111-20210111-20210211] \
                                      --output_template [metrics_after_abc] \
                                      --config_file [results/US/MA-20201111-20210111-20210211/config_after_abc.json] \
                                      --true_stats [datasets/US/MA-20201111-20210111-20210211/daily_counts.csv] \
                                      --input_summaries_template [summary_after_abc_] \
                                      --coverages [2.5_97.5,10_90,25_75] \
                                      --comma_sep_expected_columns [n_InGeneralWard,n_OffVentInICU,n_OnVentInICU,n_InICU,n_occupied_beds,n_TERMINAL,n_TERMINAL_5daysSmoothed]
```

Arguments:  
- *input_summaries_template* identifies the files containing summaries of the forecasts, i.e. the files generated by the script summarize_forecasts.py.
- *comma_sep_expected_columns* identifies the columns for which metrics want to be computed. It defaults to all possible columns. We note, however, that the 'n_TERMINAL_5daysSmoothed' column is only available for US states).

The script currently computes:  
- MAE for each expected_column, averaged over timesteps.  
- Coverage for each expected_column. Coverages are user-specified in the following format: low1_high1,low2_high2,...,lowN_highN. The requested coverages must be computable from the percentiles computed in step 3 (e.g. can compute 10\_90 coverage only if 10 and 90 percentiles were computed in step 3).

## Step 5: visualize posterior and forecasts

Run the following command:  
```console
$ python -m aced_hmm.visalize_forecasts --samples_path [results/US/MA-20201111-20210111-20210211/posterior_samples.json] \
                                        --config_path [results/US/MA-20201111-20210111-20210211/config_after_abc.json] \
                                        --input_summaries_template_path [results/US/MA-20201111-20210111-20210211/summary_after_abc] \
                                        --true_stats [datasets/US/MA-20201111-20210111-20210211/daily_counts.csv]
```
