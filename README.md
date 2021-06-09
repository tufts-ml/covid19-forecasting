
See our model derviation manuscript:

Overleaf link <a href="https://www.michaelchughes.com/papers/VisaniEtAl_arXiv_2021.pdf">https://www.michaelchughes.com/papers/VisaniEtAl_arXiv_2021.pdf</a>

Jump to: [Usage](#usage) - [Modeling](#modeling) - [Installation](#installation) - [ABC](#ABC)

For questions or concerns about the code, please [report an Issue](https://github.com/tufts-ml/covid19-forecasting/tree/pop_cv19)

### Contact:

For questions or concerns about the code, please [report an Issue](https://github.com/tufts-ml/covid19-forecasting/tree/pop_cv19/issues)

For general questions, please email [Prof. Michael C. Hughes](https://www.michaelchughes.com) - mike (AT) michaelchughes.com

# Population Model for COVID-19 susceptible people

Our proposed mechanistic model can be used to forecast how populations of susceptible people will become hospital admissable after declining in health through stages of infected, symptomatic, ailing, then hospital admissible. 

# Modeling

![Image of Model](/supplement/PopulationModelDiagram.png)

Figure 1: Diagram of the proposed Markovian model for a COVID-19 patient’s trajectory through the hospital. Rectangles indicate possible states, defined by the patient’s current pre-hospitalization stages, Defenseless D, Infected I, Symptomatic S, Ailing A, Hospital-admissible H, and recovering R. Defenseless population are individuals who are susceptible COVID-19. Rt is the viral reproductive constant which determines how fast the virus spreads. After a person is infected and joins I-stage population, a fraction of the population (denoted by pS) progresses forward to the Symptomatic S stage, while the remaining 1 − pS fraction exits the progression model and recovers from COVID-19 (denoted by the R stage). This concept of a 2 group split (forward-progressing group, and recovering group) also applies for the remaining transitions S → A and A → H, with the corresponding parameters pA and pH . When an individual is in stage I, they are likely to stay in stage I for some ’sojourn duration (days)’ specified by a sojourn probability mass function (PMF). This sojourn concept also holds true for the symptomatic population (stage S), with a separate sojourn PMF. All state transitions signified with blue arrows can be considered as a 1 time-step (daily) transitions. On the contrary, red arrows indicate transitions take X time-step(s)/day(s), where X is specified by the sojourn PMFs.

At each timestep, a patient can be described by:
* a binary health state ('Recovering' or 'Declining')
* an ordinal location state (e.g. 'InGeneralWard', 'OffVentInICU', 'OnVentInICU')
* the time left before transition to the next location state

These sojourn parameters and transition parameters are readily estimated from local data or the literature (e.g. our prior distributions for these parameters are inspired by previously published works).

We take an initial population, and run the model forward for a desired number of days.

TODO TODO

By reading parameters in from a plain text file [example](./workflows/example_simple/params.json), the model transparently facilitates communication of assumptions and invites modifications.



Using this model, we can:

* **fit parameters** to aggregated daily count time-series from a specific hospital site or region
* **forecast** future daily counts to help administrative officials understand future demand for resources
* **assess the societal value** of possible interventions (e.g. would decreasing admissions by X% help California avoid a lockdown in late 2020?)

The model and its fitting and forecasting procedures satisfy two desired properties:

* We focus on **point-estimate** modeling using a gradient descent approach. The framework of our object loss is a Maximum-a-priori problem. After defining a set of prior distributions on the parameters:
  * param1
  * param2


# Gradient Descent

We provide here an example python notebook, together with an overview of the necessary commands, to optimize the model parameters such that true hospital-admissions counts are well fitted. To get started, we recommend running the example notebook as is. The notebook will guide you through:

* optimize the parameters such that the retrospectively forecast hospital-admission numbers matches hospital-admission numbers observed by the state of MA (this is done via gradient descent) 
* plotting the prior distributions of the parameters
* plotting the results of forecasting with optimized point-estimates of the learnable parameters

TODO 

In the results folder, we also provide a set of parameters (.pickle) that has already been fitted to 4 US states.





we fit these parameters until our model yields retrospective hospital-admission numbers that matches ground-truth data sources of HHS.gov.

* To be **portable** to health systems around the world, we assume access only to aggregated daily counts of hospital-admissions data. So a country or state may collect either regional or wholistic datasets, our population model will just use the daily sums of those numbers.

## Data collection

We provide the datasets we used in our experiments, as well as code to automatically collect and format US state level data from HHS and from the Covid Tracking Project.  
Code and README are provided in the folder datasets/US.




# Usage

### Getting Started

Here's a very simple example, that will run our probabilistic progression model (with dummy initial conditions and dummy parameters) to forecast ahead for 120 days. (Requires you have already [installed this project's conda environment](#installation)

```console
$ source env/bin/activate
$ python TODO run forecast into future 30 days
```

**Expected output:**
```console
Forecasting into future by 30 days

```

TODO This will write a file daily_admissions_forefcast.csv, The CSV should have rows of hospital admissions up to today's date. In addition to all the past dates of hospital-admissions, the last 30 rows of the CSV should contain 30 days of forecasted admissions.

TODO See an example output in (example_output/](.example_output/)

# Installation

#### 1. Install python virtual env

Follow the instructions here: https://docs.python.org/3/library/venv.html

#### 2. Install requirements.txt into environment via pip

Use the project's included requirements.txt file to specify all packages needed
```
pip install -r requirements.txt
```




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





## Step 0: Ensure that the HHS data source of hospital-admission counts is still an active API to date.



## Step 1: Train and Fit the model to hospital-admissions counts observed up to today

Run the following command:  
```console
$ python -m aced_hmm.fit_posterior_with_abc
```

See the source code at aced_hmm/fit_posterior_with_abc.py for an explanation of the arguments.

Output:  TODO
- a pickle file that has the most optimized parameters
- CSV file that predicts for hospital admissions from the start_date to end_date

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
