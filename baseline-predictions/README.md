How to Run
==========
For both (1) and (2)
    use `-a` to only run the GAR,
    or `-g` to only run the GGP.
Else, defaults to running both models and producing side-by-side plots.

(1) `python grid_search.py <input_csv_file>`

**Optional arguments (defaults)**

    -c, --target_col_name       'hospitalized_total_covid_patients_suspected_and_confirmed_including_icu'
    -m, --gar_model_file        'gar_model.json'
    -f, --ggp_model_file        'ggp_model.json'
    -p, --plot_file             'performance.png'

**Output**
* Plot of best score for each window size (GAR)
* Plot of best score for each timescale prior mean (GGP)
* Plot of forecasts on heldout set using best model
* JSON file for each model with best model parameters found

(2) `python run_simple_forecast.py <input_csv_file>`

Note: `<input_csv_file>` must have a column 'date' with dates in ISO format.

**Optional arguments (defaults)**

    -c, --target_col_name           'hospitalized_total_covid_patients_suspected_and_confirmed_including_icu'
    -m, --gar_model_file            'gar_model.json'
    -f, --ggp_model_file            'ggp_model.json'
    -o, --gar_csv_file_pattern      'gar_samples/output-*.csv'
    -u, --ggp_csv_file_pattern      'ggp_samples/output-*.csv'
    -n, --n_samples                 1000
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
          gar_grid_search.py       ggp_grid_search.py
               /       \                 /      \
              /         \               /        \
             /           \             /          \
       NegBinAR.py      plot_forecasts.py      NegBinGP.py   


(2)             

                    run_simple_forecast.py
                          /         \
                         /           \
                        /             \
                       /               \
                      /                 \
             gar_forecast.py          ggp_forecast.py
               /       \                 /      \
              /         \               /        \
             /           \             /          \
       NegBinAR.py      plot_forecasts.py      NegBinGP.py                 

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
We suppose that y is NegBin distributed over the exponential of a latent
time series modeled by an autoregressive process with W lags, i.e.,

    y(t) ~ NegBin( mu = exp(f(t)), alpha )
where for each t, f(t) is a linear combination of the past W timesteps, i.e.,
 
    f(t) ~ N( mu = b_0 + b_1 * f(t-1) + ... + b_W * f(t-W), sigma = 0.01)

Generalized Gaussian Process (GGP)
----------------------------------
We suppose that y is NegBin distributed over the exponential of a latent
Gaussian Process, i.e.,
    
    y(t) ~ NegBin( mu = exp(f(t)), alpha )
where f is modeled by a GP
    
    f(t) ~ N( m(t), k(t, t') )
with Constant mean
    
    m(t) = c
and Squared Exponential covariance
    
                          (t - t')^2
    k(t, t') = a^2 exp(- ------------ )
                             2l^2




