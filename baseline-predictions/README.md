How to Run
==========
For both (1) and (2)
    use `-l` to only run the GLM,
    or `-g` to only run the GGP.
Else, defaults to running both models and producing side-by-side plots.

(1) `python grid_search.py <input_csv_file>`

**Optional arguments (defaults)**

    -c, --target_col_name       'cases'
    -m, --glm_model_file        'glm_model.json'
    -f, --ggp_model_file        'ggp_model.json'
    -p, --plot_file             'performance.png'

**Output**
* Plot of best score for each window size (GLM)
* Plot of best score for each timescale prior mean (GGP)
* JSON file for each model with best model parameters found

(2) `python run_simple_forecast.py <input_csv_file>`

Note: `<input_csv_file>` must have a column 'date' with dates in ISO format.

**Optional arguments (defaults)**

    -c, --target_col_name           'cases'
    -m, --glm_model_file            'glm_model.json'
    -f, --ggp_model_file            'ggp_model.json'
    -o, --glm_csv_file_pattern      'glm_samples/output-*.csv'
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
                           /      \
                          /        \
                         /          \
                        /            \
                       /              \
                      /                \
                     /                  \
                    /                    \
        glm_grid_search.py          ggp_grid_search.py
                |                           |
                |                           |
                |                           |
    NegativeBinomialRegression.py   NegativeBinomialGP.py


(2)             

                    run_simple_forecast.py
                       /      \         \
                      /        \         \
                     /          \         \
                    /            \         \
                   /              \     plot_forecasts.py
                  /                \
                 /                  \
                /                    \
        glm_forecast.py            ggp_forecast.py
            |                           |
            |                           |
            |                           |
    NegativeBinomialRegression.py   NegativeBinomialGP.py


Model Summary
=============
Both models are implemented using PyMC3.
More information about model parameters and how they were selected can be
found in `glm_grid_search.py` and `ggp_grid_search.py`.

Goal
----
Given a series of counts (admissions, census, etc.)
    
    y(t) for t = 1, ..., T
where t is an integer representing a date in time.
Assuming T is today, want to predict counts
    
    y(T+1), y(T+2), ..., y(T+7)
for the week ahead.

Generalized Autoregressive Model (GLM)
--------------------------------------
We suppose that for all t, y(t) is NegBin distributed over a linear combination
of the counts from the past W timesteps, i.e.,
    
    y(t) ~ NegBin( mu = inner_prod(beta, y[t-W:t-1]), alpha )
for all t.

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




