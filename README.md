# covid19-forecasting

Simulation software for forecasting demand at various stages of hospitalization

PI: Michael C. Hughes

## Usage

Run the following python script, which consumes a JSON file of configuration options, and produces as output an comma-separated value file (CSV)

```
python run_forecast.py --config_file params_simple_example.json  --output_file results.csv --random_seed 8675309
```

## Summary

The config file specifies a "semi-Markov" model of patient progression through the hospital system.

At each timestep, a patient can be described by:
* a binary health state ('Recovering' or 'Declining')
* an ordinal location state (e.g. 'Presenting', 'InGeneralWard', 'InICU', 'OnVentInICU')



 
