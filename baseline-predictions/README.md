Files & Directories
===================

- `notebooks` - Jupyter notebooks with detailed specifications of the models and how they translate to code.
- `mass_dot_gov_datasets` - CSV files with data used for experiments.
    - CSV files must have a column named `date` with dates in ISO format, and a target column with integer counts.
- `src`
  - `arg_types.py` - Checks that filenames specified by command-line arguments have proper suffixes.
  - `gar_grid_search.py` - Performs grid search for single-site GAR model.
  - `GenPoisson.py` - Defines a Generalized Poisson distribution as a PyMC3 custom Discrete distribution.
  - `GenPoissonAutoregression.py` - Defines a generalized autoregressive (GAR) model with Generalized Poisson likelihood.
  - `GenPoissonGaussianProcess.py` - Defines a generalized Gaussian Process (GGP) model with Generalized Poisson likelihood.
  - `ggp_grid_search.py` - Performs grid search for GGP model.
  - `grid_search.py` - Launches grid searches for single-site models.
  - `plot_forecasts.py` - Plots summary statistics of forecasts against true observed counts.
  - `multi_site_gar.py` - Trains, evaluates, and makes forecasts for multi-site GAR model.
  - `poisson_vs_genpoisson.py` - Compares standard and generalized Poisson likelihoods on our model.

How We Ran Our Experiments
==========================
Required Libraries
---------
- pymc3: https://docs.pymc.io/
- argparse, numpy, matplotlib, pandas, json, datetime, theano, scipy, os

Experiment #1: Standard vs Generalized Poisson
-------------
On each dataset, trains and scores GAR model with W=1 first using Standard Poisson likelihood, then using Generalized Poisson likelihood.
Reads all files in the directory specified on line 17 of the script. To change the directory name and target column name, modify lines 17, 19, and 29.

`python poisson_vs_genpoisson.py`

Experiment #2: Single-Site GGP vs GAR
-------------
Divides sequence of counts into training, validation, and test windows. Runs a grid search over a predefined set of hyperparameters for each model, evaluating on validation set. Takes the best parameters and trains the training and validation set together, then makes forecasts on the test window.

`python grid_search.py <input_csv_file>`

`<input_csv_file>` must have a column `date` with dates in ISO format, and a target column with integer counts.

Use the flag `-a` to only run the GAR, or the flag `-g` to only run the GGP.
Otherwise defaults to running both models and producing side-by-side plots.

**Optional arguments and their defaults**

    -c, --target_col_name       'hospitalized_total_covid_patients_suspected_and_confirmed_including_icu'
    -m, --gar_model_file        'gar_model.json'
    -o, --ggp_model_file        'ggp_model.json'
    -p, --performance_plot_file 'performance.pdf'
    -f, --forecast_plot_file    'heldout_forecasts.pdf'

**Output**
* Plot of best score for each window size (GAR)
* Plot of best score for each timescale prior mean (GGP)
* JSON file for each model with best model parameters found
* Plot of summary statistics of forecasts against true observed counts

Experiment #3: Multi-site GAR
-------------
Trains and scores multi-site GAR model and makes forecasts on test window. Reads all files in directory specified on line 20 of the script. To change the directory name and target column name, modify lines 20, 23, and 33. Saves traceplot and forecast plots in directory specified on lines 80 and 125 of the script.

`python multi_site_gar.py`




