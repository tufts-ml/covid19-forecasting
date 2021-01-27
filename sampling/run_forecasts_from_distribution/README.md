

# Running forecasts with parameters sampled from a prior distribution

## Step 1: draw samples from the prior

Run the following command:
    `python sample_prior.py --prior_file [configs/example_prior.json] --output_file [configs/samples_from_prior.json] --random_seed [101] --num_samples [100]`

- *prior_file* contains the prior in json format.
- *output_file* is where the samples will be stored, as a list of dictionaries, in json format.

## Step 2: run forecasts using drawn samples

Run the following command:
    `python run_forecasts_from_samples.py --config_dict [configs/example_params.json] --samples_file [configs/samples_from_prior.json]
                                          --output_file [output/results_under_prior_random_seed=101_sample=None.csv]
                                          --random_seed [101] --num_samples 100`

- *config_dict* is the semimarkov_forecaster's standard config file
- *samples_file* contains the samples of parameters in json format

The script will run a single simulation from each of the last *num_samples* samples in samples_file.
Make sure that *_random_seed=101_sample=None* is present in the *output_file*, with the random seed value matching *random_seed*.

## Step 3: compute summaries of forecasts

Run the following command:
    `python summarize_forecasts.py --input_dir [output] --output_dir [output] --output_template [sumary_under_prior_]
                                   --input_csv_file_pattern [results_under_prior_*.csv] --comma_sep_percentiles [1,2.5,5,10,25,50,75,90,95,97.5,99]
                                   --include_los [False]`

- *input_csv_file_pattern* identifies all the files containing the forecasts from the script run_forecasts_from_samples.py.
- *include_los* says whether we want to compute summaries for the los results as well.

The summaries include (one file for each one): mean, stddev, and all percentiles specified by *comma_sep_percentiles*