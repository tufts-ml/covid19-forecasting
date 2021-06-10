
See our model derviation manuscript:

Overleaf link <a href="https://www.michaelchughes.com/papers/VisaniEtAl_arXiv_2021.pdf">https://www.michaelchughes.com/papers/VisaniEtAl_arXiv_2021.pdf</a>

Jump to: [Usage](#usage) - [Modeling](#modeling) - [Installation](#installation) - [ABC](#ABC)


### Contact:

For questions or concerns about the code, please [report an Issue](https://github.com/tufts-ml/covid19-forecasting/tree/pop_cv19/issues)

For general questions, please email [Prof. Michael C. Hughes](https://www.michaelchughes.com) - mike (AT) michaelchughes.com

# Population Model for COVID-19 susceptible people

Our proposed mechanistic model can be used to forecast how populations of susceptible/defenseless people will become hospital admissable after declining in health through stages of infected, symptomatic, ailing, then hospital admissible ("hospitalized"). 

# Modeling

![Image of Model](/supplement/PopulationModelDiagram.png)

Figure 1: Diagram of the proposed Markovian model for a COVID-19 patient’s trajectory through the hospital. Rectangles indicate possible states, defined by the patient’s current pre-hospitalization stages, Defenseless D, Infected I, Symptomatic S, Ailing A, Hospital-admissible H, and recovering R. Defenseless population are individuals who are susceptible COVID-19. Rt is the viral reproductive constant which determines how fast the virus spreads. After a person is infected and joins I-stage population, a fraction of the population (denoted by pS) progresses forward to the Symptomatic S stage, while the remaining 1 − pS fraction exits the progression model and recovers from COVID-19 (denoted by the R stage). This concept of a 2 group split (forward-progressing group, and recovering group) also applies for the remaining transitions S → A and A → H, with the corresponding parameters pA and pH . When an individual is in stage I, they are likely to stay in stage I for some ’sojourn duration (days)’ specified by a sojourn probability mass function (PMF). This sojourn concept also holds true for the symptomatic population (stage S), with a separate sojourn PMF. All state transitions signified with blue arrows can be considered as a 1 time-step (daily) transitions. On the contrary, red arrows indicate transitions take X time-step(s), where X is specified by the sojourn PMFs.

These sojourn PMF and transition parameters are readily estimated from local data or the literature (e.g. our prior distributions for these parameters are inspired by previously published works).

Using this model in conjunction with gradient descent, we can:

* **fit sojourn parameters and transition parameters** such that the predicted hospital-admission numbers matches ground-truth hospital-admission numbers
* **forecast** future daily hospital-admission counts to help administrative officials understand future demand for resources
* **assess the societal value** of possible interventions (e.g. would decreasing admissions by X% help California avoid a lockdown in late 2020?)


# Gradient Descent

We focus on **point-estimate** optimization using a gradient descent approach. We frame the objective loss within a Maximum-a-priori framework.  Our objective is to best fit predicted hospital-admissible-numbers to ground-truth hospital-admissible-numbers while keeping the learnable parameters regularized via pre-defined prior distributions for the sojourn parameters and the transition parameters.

![Python Notebook Demonstrating Gradient Descent](/PopulationModel-train-MA.ipynb)

We provide here an example python notebook, together with an overview of the necessary commands, to optimize the model parameters such that true hospital-admissions counts are well fitted. To get started, we recommend running the example notebook as is. The notebook will guide you through:

* plotting the prior distributions of the learnable parameters.
* optimization of the parameters such that the retrospective forecast hospital-admission numbers  matches ground-truth hospital-admission numbers observed by the state of Massachusetts (MA). In this ![Example notebook](/PopulationModel-train-MA.ipynb), our model sees training data (of MA) for the training period of 01-01-2021 to 04-01-2021, and tries to tweak the learnable parameters to best fit that training period. 
* Forecasting hospital-admissible numbers for the testing period of 04-01-2021 to 06-01-2021, given the optimized learnable parameters.
* plotting the results of forecasting with optimized point-estimates of the learnable parameters, which are saved into ![Results folder](/results)

In the results folder, we also provide a set of parameters (.pickle) that has already been fitted to 4 US states. TODO





## Data collection


To have our model **transferable** to health systems around the world, we assume access only to aggregated daily counts of hospital-admissions data. So a country or state may collect regional datasets, our population model will just use the counts from daily sums-across-regions.

In the ![Data folder](/data), We provide the datasets we used in our experiments, which our code automatically collects and formats from these data sources:
* [Health and Human Services (HHS)](https://healthdata.gov/api/views/g62h-syeh/rows.csv?accessType=DOWNLOAD) - provides us with ground-truth hospital-admissions (H)
* [CovidEstim.org](https://covidestim.s3.us-east-2.amazonaws.com/latest/state/estimates.csv) - provides us with:
  * reproductive constant (Rt)
  * Initialization values for Infected (I)
  * Initialization values for Symptomatic (S)
  * Initialization values for Ailing (A)

Please note that these data sources may become deprecated/outdated. In the case of data-source-deprecation, our code base will require new external data sources, so please notify [Prof. Michael C. Hughes](https://www.michaelchughes.com) - mike (AT) michaelchughes.com


# Usage

### Getting Started

Here's a very simple [example notebook](/PopulationModel-train-MA.ipynb), that uses a model pre-trained (on the training MA dataset of 01-01-2021 to 04-01-2021) to retrospectively forecast the next 2 months of MA hospital-admissions (04-01-2021 to 06-01-2021). (Running notebook requires [Installation](#Installation) and [Running Jupyter Notebook](#Jupyter-Notebook))

Running this example notebook will write a file [daily_admissions_forecast.csv](/results/daily_admissions_forecast.csv). The CSV should have rows of hospital admissions up to today's date. In addition to all the past dates of hospital-admissions, the last 60 rows of the CSV should contain 60 days of forecasted admissions.


# Installation

#### 1. Creating a python virtual environment folder

Follow the instructions here: https://docs.python.org/3/library/venv.html
```
python -m venv c:\path\to\myenv
```

#### 2. Activate the virtual environment
```
cd c:\path\to\myenv
source myenv\bin\activate
```

#### 3. Install requirements.txt into virtual environment via pip

Use the project's included requirements.txt file to specify all packages needed
```
cd c:\path\to\myenv
pip install -r requirements.txt
```


# Jupyter-Notebook

#### 1. Open jupyter notebook interface via the following command lines
```
cd c:\path\to\myenv
jupyter notebook
```

#### 2. Run through each jupyter notebook cell block


# Modifying the Notebook to forecast into the future

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

