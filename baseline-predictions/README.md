Quickstart Guide
================
1. Upload all code to one directory in your cluster account.
2. Feel free to modify the hyperparameter spaces initialized in lines 37-38
of `gar_grid_search.py` and lines 36-37 of `ggp_grid_search.py`. If you want to cut down on GAR grid search, I recommend setting
`prior_sigmas = [(0.1, 0.1)]`. For GGP, `c` is the constant mean function for the Gaussian Process, so `c_mu` should be roughly equal
to the log of the average count in the dataset. We are plotting performance across `window_size` and `l_mu` so those hyperparameter
spaces are meant to encompass a wide and equispaced range of values.
3. Set up your conda environment, and edit `do_experiment.slurm` to reflect where you have the environment set up.
I'm using `spr_2020_env` from Mike's class, and I installed PyMC3 by running the following command: `conda install -c conda-forge pymc3`.
4. Create the following subdirectories:
    - `gar_models` - GAR model files
    - `ggp_models` - GGP model files
    - `gar_samples` - GAR forecast samples
    - `ggp_samples` - GGP forecast samples
        - Note: The way I have it set up, the forecasts are all being dumped into `gar_samples` and `ggp_samples`, so the samples get replaced
every time the program is run. If you want to actually save all the forecasts, you could create separate subdirectories for each
dataset and then update `launch_experiments.sh` to create a variable for the folder names, and update `do_experiment.slurm` to supply
them as command line arguments.
    - `performance` - model performance plots
    - `forecasts` - forecast plots
    - `output` - stdout and stderr files
5. Upload your datasets to the main directory with all the code. Make sure they are CSV files with a column `date` with dates in ISO format.
6. Update `do_experiment.slurm` with the name of the CSV column with your data by adding `-c <target_col_name>` to both commands.
7. Run `./launch_experiments.sh submit`. The whole process could take a day or two depending how large the grid search is.
8. Check stdout to see posterior means and heldout likelihood for each set of parameters. Check stderr for convergence warnings.
It's good to check these periodically to make sure the parameters you've set are appropriate for the data. I've had to play around with the
arguments for the PyMC3 `sample()` method (line 67 in `GenPoissonAutoregression.py` and line 75 in `GenPoissonGaussianProcess.py`) to get
the MCMC chains to converge nicely, so they've been carefully selected for the mass.gov datasets but might not work well with other datasets.
9. Check `performance` and `forecasts` for plots.

Detailed specifications of the models and how they translate to code can be
found in the [Jupyter notebooks](https://github.com/tufts-ml/covid19-forecasting/tree/al-baseline-predictions/baseline-predictions/notebooks).

Files
=====
Python:
- `arg_types.py` - Checks that filenames specified by command-line arguments have proper suffixes.
- `gar_forecast.py` - Makes forecasts using GAR model.
- `gar_grid_search.py` - Performs grid search for GAR model.
- `GenPoisson.py` - Defines a Generalized Poisson distribution as a PyMC3 custom Discrete distribution.
- `GenPoissonAutoregression.py` - Defines a generalized autoregressive model with Generalized Poisson likelihood.
- `GenPoissonGaussianProcess.py` - Defines a generalized Gaussian Process model with Generalized Poisson likelihood.
- `ggp_forecast.py` - Makes forecasts using GGP model.
- `ggp_grid_search.py` - Performs grid search for GGP model.
- `grid_search.py` - Driver for grid search.
- `plot_forecasts.py` - Plots summary statistics of forecasts.
- `run_simple_forecast.py` - Driver for forecasts.

JSON:
Grid search writes best model parameters to these files. Forecast reads model parameters from these files.
- `gar_model.json` - Contains GAR model parameters.
- `ggp_model.json` - Contains GGP model parameters.

Shell Scripts:
- `do_experiment.slurm` - Runs grid search and forecasts.
- `launch_experiments.sh` - Sets up experiments for all CSV files in the current directory.

How to Run
==========
For both (1) and (2)
    use `-a` to only run the GAR,
    or `-g` to only run the GGP.
Else, defaults to running both models and producing side-by-side plots.

(1) `python grid_search.py <input_csv_file>`

**Optional arguments and their defaults**

    -c, --target_col_name       'hospitalized_total_covid_patients_suspected_and_confirmed_including_icu'
    -m, --gar_model_file        'gar_model.json'
    -f, --ggp_model_file        'ggp_model.json'
    -p, --plot_file             'performance.png'

**Output**
* Plot of best score for each window size (GAR)
* Plot of best score for each timescale prior mean (GGP)
* JSON file for each model with best model parameters found
* Optional: Uncomment code at the bottom of `grid_search.py`, `gar_grid_search.py`, and `ggp_grid_search.py`
to make forecasts on heldout set using best model parameters found, and plot against observed counts.

(2) `python run_simple_forecast.py <input_csv_file>`

Note: `<input_csv_file>` must have a column `date` with dates in ISO format.

**Optional arguments and their defaults**

    -c, --target_col_name           'hospitalized_total_covid_patients_suspected_and_confirmed_including_icu'
    -m, --gar_model_file            'gar_model.json'
    -f, --ggp_model_file            'ggp_model.json'
    -o, --gar_csv_file_pattern      'gar_samples/output-*.csv'
    -u, --ggp_csv_file_pattern      'ggp_samples/output-*.csv'
    -n, --n_samples                 5000
    -s, --day_forecasts_start       day after last day of data
    -d, --n_days_ahead              7
    -p, --plot_file                 'forecasts.png'

**Output**
* Plot of summary statistics of forecasts
* CSV file for each sample of forecasts


Program Architecture
====================
(1)                 

                              grid_search.py
                                /        \
                               /          \
                              /            \
              gar_grid_search.py            ggp_grid_search.py
                    |                             |
                    |                             |
                    |                             |
        GenPoissonAutoRegression.py       GenPoissonGaussianProcess.py
                              \             /
                               \           /
                                \         /
                               GenPoisson.py


(2)             

                          run_simple_forecast.py
                                /        \
                               /          \
                              /            \
                gar_forecast.py            ggp_forecast.py
                    |                             |
                    |                             |
                    |                             |
        GenPoissonAutoRegression.py       GenPoissonGaussianProcess.py
                              \             /
                               \           /
                                \         /
                               GenPoisson.py              

Model Summary
=============
Both models are implemented using PyMC3.
More information about model parameters and how they were selected can be
found in `gar_grid_search.py` and `ggp_grid_search.py`.

Goal
----
Given a series of counts (admissions, census, etc.)
    
    y(t) for t = 1, ..., T
where t is an integer representing a date in time.
Assuming T is today, want to predict counts
    
    y(T+1), y(T+2), ..., y(T+7)
for the week ahead.

Generalized Autoregressive Model (GAR)
--------------------------------------
We suppose that y is GenPoisson distributed over the exponential of a latent
time series modeled by an autoregressive process with W lags, i.e.,

    y(t) ~ GenPoisson( theta = exp(f(t)), lambda )
where for each t, f(t) is a linear combination of the past W timesteps, i.e.,
 
    f(t) ~ N( mu = b_0 + b_1 * f(t-1) + ... + b_W * f(t-W), sigma = 0.01)

Generalized Gaussian Process (GGP)
----------------------------------
We suppose that y is GenPoisson distributed over the exponential of a latent
Gaussian Process, i.e.,
    
    y(t) ~ GenPoisson( theta = exp(f(t)), lambda )
where f is modeled by a GP
    
    f(t) ~ N( m(t), k(t, t') )
with Constant mean
    
    m(t) = c
and Squared Exponential covariance
    
                          (t - t')^2
    k(t, t') = a^2 exp(- ------------ )
                             2l^2
