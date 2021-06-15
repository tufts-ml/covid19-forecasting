
See our model derviation manuscript:

Overleaf link <a href="https://www.michaelchughes.com/papers/VisaniEtAl_arXiv_2021.pdf">https://TODO</a>

Jump to: [Model Details](#modeling) - [Usage](#usage) - [Installation](#installation) - [Connecting to ACED-HMM](#Connecting-to-ACED-HMM)


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
* optimization of the parameters such that the retrospective forecast hospital-admission numbers  matches ground-truth hospital-admission numbers observed by the state of Massachusetts (MA). In this notebook, our model sees training data (of MA) for the training period of 03-01-2021 to 04-01-2021, and tries to optimize the learnable parameters to best fit that training period. 
* Forecasting hospital-admissible numbers for the testing period of 04-01-2021 to 06-01-2021, given the optimized learnable parameters.
* plotting the results of forecasting with optimized point-estimates of the learnable parameters, which are saved (as .pickle) into ![Results folder](/results)


## Data-collection


To have our model **transferable** to health systems around the world, we assume access only to aggregated daily counts of hospital-admissions data. So a country or state may collect regional datasets, our population model will just use the counts from daily sums-across-regions.

In the ![Data folder](/data), We provide the datasets we used in our experiments, which our code automatically collects and formats from these data sources:
* [Health and Human Services (HHS)](https://healthdata.gov/api/views/g62h-syeh/rows.csv?accessType=DOWNLOAD) - provides us with ground-truth hospital-admissions (H)
* [CovidEstim.org](https://covidestim.s3.us-east-2.amazonaws.com/latest/state/estimates.csv) - provides us with:
  * reproductive constant (Rt)
  * Initialization values for Infected (I)
  * Initialization values for Symptomatic (S)
  * Initialization values for Ailing/Severe (A)

Please note that these data sources may become deprecated/outdated. In the case of data-source-deprecation, our code base will require new external data sources, so please notify [Prof. Michael C. Hughes](https://www.michaelchughes.com) - mike (AT) michaelchughes.com


# Usage

### Getting Started

Here's a very simple [example notebook](/PopulationModel-train-MA.ipynb), that uses a model (trained on the training MA dataset of 03-01-2021 to 04-01-2021) to retrospectively forecast the next 2 months of MA hospital-admissions (04-01-2021 to 06-01-2021). (Running notebook requires [Installations](#Installation) and [Jupyter Notebook](#Jupyter-Notebook))

Running this example notebook will write a file [daily_admissions_forecast.csv](/results/daily_admissions_forecast.csv). The CSV should have rows of ground-truth hospital-admissions up to 04-01-2021. In addition to all the training dataset's dates of hospital-admissions, the last 60 rows of the CSV should contain 60 days of forecasted admissions.


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

#### 3. Install packages from requirements.txt into virtual environment via pip

Use the project's included requirements.txt file to specify all packages needed
```
cd c:\path\to\myenv
pip install -r requirements.txt
```


# Jupyter-Notebook

#### 1. Open jupyter notebook interface
```
cd c:\path\to\myenv
jupyter notebook
```

#### 2. Within the interface, Run each jupyter notebook code block


# Modifying the ![Example notebook](/PopulationModel-train-MA.ipynb) to forecast into the future

## PRE-REQUISITE: 
* Ensure that you understand the ![example notebook](/PopulationModel-train-MA.ipynb) "AS IS" before any modifications.
* Ensure that the HHS data source of hospital-admission counts is still an active API to date. 
* Ensure that Covidestim.org data source is still active API to recover the field names of Rt, Infections, Symptomatics, Ailing/Severe. 
[More details about our data source](#data-collection)


## Step 0: Define Training period to fit to hospital-admissions counts observed up to today
Set the training period to the past 2 months
```
training_end_date = '20210401' <- fill this with TODAY's date or a recent date observed by HHS and Covidestim.org
training_start_date = '20210301' <- fill this with a date that is about 2 months ago
```

Ensure that the "covidestim.csv" file is being used and not the outdated "covidestim_old_version.csv"
```
pd_warmup_data = PopulationData(data_folder+"covidestim.csv", state_name, 
                                start_date=lookback_date, end_date=training_start_date);
```

Run all cell blocks in Step 0

## Step 1: Tweak the hyper parameters of the .fit() method
n_iters, step_size_txn, step_size_soj, lambda_reg needs to be tweaked depending on the USA state being fitted to. As a rule of thumb for the 2 learning_rates ('step_size'), the step_size_soj should be 1 order of magnitude larger than step_size_txn.  

```
pop_model.fit(training_data_obj, 
              n_iters=32, step_size_txn=5e-5, step_size_soj=9e-4, n_steps_between_print=5, lambda_reg=1e-3, plots=True) <- change these hyper parameters accordingly
```

Run all cell blocks in Step 1

Before moving on to the next step, confirm that the training loss has converged and all the parameters' gradients is approaching 0 in the final iterations of .fit()


## Step 2 (Skip straight to Forecasting Beyond Today): 
Set the warmup_data to include data from lookback_date until today's date
```
pop_model.warmup_data.end_date = '20210610' <- set this to today's date
pop_model.forecast_duration=60 <- set this to the number of days you want to forecast
```

Run all cell blocks following 'Forecasting Beyond Today' in Step 2

![Results folder](/results) should contain a daily_admissions_forecast.csv file of your most recent forecast.


# Connecting-to-ACED-HMM
This section is for those of you interested in using this hospital-admissions forecast with the [ACED HMM model](https://github.com/tufts-ml/aced-hmm-hospitalized-patient-trajectory-model).


The ![Results folder](/results)'s daily_admissions_forecast.csv should contain future/forecast entries of hospital-admission numbers that can be fed into the [ACED HMM model](https://github.com/tufts-ml/aced-hmm-hospitalized-patient-trajectory-model). To do this, you must Git-Clone the [ACED HMM model](https://github.com/tufts-ml/aced-hmm-hospitalized-patient-trajectory-model) repo into a folder that is separate from the Population Model, and then follow these steps:

(Note that all of these steps below are done within the ACED HMM repo folder, and do not require any code modification to the pre-existing Population Model repo. Only the Population Model's ![Results folder](/results)'s daily_admissions_forecast.csv file is required for step 1.5)

## [ACED HMM model](https://github.com/tufts-ml/aced-hmm-hospitalized-patient-trajectory-model) Step -1. Generate the data for the future, 

```
cd datasets/US
python generate_data_US.py --state MA --start_date 20210101 --end_training_date 20210524 --end_date 20210801
```
* --end_date 20210801 # replace this with some date in the near future (1 or 2 months)
* --end_training_date 20210524 # replace this with a date 2 months before end_date

(for the rest of this step-by-step tutorial, we will refer to 20210524 and 20210801, but you should recognize to replace the following dates with your desired dates)


## [ACED HMM model](https://github.com/tufts-ml/aced-hmm-hospitalized-patient-trajectory-model) Step 0. Install Cython, and conda environments, 

## [ACED HMM model](https://github.com/tufts-ml/aced-hmm-hospitalized-patient-trajectory-model) Step 1. Run ABC
```
python -m aced_hmm.fit_posterior_with_abc --input_dir datasets/US/MA-20210101-20210524-20210801 --output_dir results/US/MA-20210101-20210524-20210801
```

add this 'func_name' flag to run cython (for faster results)
```
--func_name cython
```

## [ACED HMM model](https://github.com/tufts-ml/aced-hmm-hospitalized-patient-trajectory-model) Step 1.5. Integration between Population Model and ACED HMM model.

a. open the [ACED HMM model](https://github.com/tufts-ml/aced-hmm-hospitalized-patient-trajectory-model) results/US/MA-20210101-20210524-20210801/config_after_abc.json which directs 
```
{
  
  "pmf_num_per_timestep_InGeneralWard": "results/US/MA-20210101-20210524-20210801/daily_admissions.csv", 
  
  ^^^THIS FILE NEEDS TO BE MODIFIED WITH FUTURE ENTRIES^^^
  
}
```
b. open the [ACED HMM model](https://github.com/tufts-ml/aced-hmm-hospitalized-patient-trajectory-model) results/US/MA-20210101-20210524-20210801/daily_admissions.csv for modification

c. append (copy-paste) the future entries from [Population Model](https://github.com/tufts-ml/covid19-forecasting/tree/pop_cv19)'s ![results/daily_admissions_forecast.csv](/results) into the [ACED HMM model](https://github.com/tufts-ml/aced-hmm-hospitalized-patient-trajectory-model) results/US/MA-20210101-20210524-20210801/daily_admissions.csv file


## [ACED HMM model](https://github.com/tufts-ml/aced-hmm-hospitalized-patient-trajectory-model) Step 2. Run forecasts (training + testing)


```
python -m aced_hmm.run_forecast --func_name python \
                                  --config_path results/US/MA-20210101-20210524-20210801/config_after_abc.json \
                                  --output_dir results/US/MA-20210101-20210524-20210801/individual_forecasts \
                                  --output_file results_after_abc-{{random_seed}}.csv \
                                  --approximate 5 \
                                  --random_seed 1001 \
                                  --num_seeds None
```

modify this 'func_name' flag to run cython (for faster results)
```
--func_name cython
```

## [ACED HMM model](https://github.com/tufts-ml/aced-hmm-hospitalized-patient-trajectory-model) Step 3: compute summaries of forecasts

```
python -m aced_hmm.summarize_forecasts --input_dir results/US/MA-20210101-20210524-20210801/individual_forecasts \
                                         --output_dir results/US/MA-20210101-20210524-20210801 \
                                         --output_template summary_after_abc_ \
                                         --input_csv_file_pattern results_after_abc*.csv \
                                         --comma_sep_percentiles 1,2.5,5,10,25,50,75,90,95,97.5,99
```

## [ACED HMM model](https://github.com/tufts-ml/aced-hmm-hospitalized-patient-trajectory-model) Step 5: visualize posterior and forecasts

```
python -m aced_hmm.visualize_forecasts --samples_path results/US/MA-20210101-20210524-20210801/posterior_samples.json \
                                        --config_path results/US/MA-20210101-20210524-20210801/config_after_abc.json \
                                        --input_summaries_template_path results/US/MA-20210101-20210524-20210801/summary_after_abc \
                                        --true_stats datasets/US/MA-20210101-20210524-20210801/daily_counts.csv
```