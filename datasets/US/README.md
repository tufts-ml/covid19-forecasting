

## Generating datasets for US states

To generate a dataset for any US state of interest for a specific timeframe, simply run the following command:  
`python generate_data_US.py --state [MA] --start_date [20201111] --end_training_date [20210111] --end_date [20210201]`

This will automatically create a new directory with the necessary data files.

We note that we have not tested the functionality of this code on all 50 US states. The available data entries vary by state, and we may not cover all cases. We cover the two most common cases: 1) all 3 hospital counts and death counts ara available (see example MA dataset); 2) same, but only aggretae ICU counts ara available (see example CA dataset)