
# Get raw data

Got to: <http://www.healthdata.org/covid/data-downloads>

Download the zip file corresponding to the release dated "2021-01-15"


# Get cleaned version targeted at specific state

Run the following, which will create 4 separate CSV files (one per state)

```
$ python extract_csv_of_ihme_forecast_for_state.py --target_location "California"
$ python extract_csv_of_ihme_forecast_for_state.py --target_location "Massachusetts"
$ python extract_csv_of_ihme_forecast_for_state.py --target_location "South Dakota"
$ python extract_csv_of_ihme_forecast_for_state.py --target_location "Utah"
```

The produced CSV file has following columns using same standardized stage names as the rest of our forecasting project. 
Each row represents the forecasted count at the designated stage.

```
location_name
date
n_Admitted_InGeneralWard_mean
n_Admitted_InGeneralWard_lower
n_Admitted_InGeneralWard_upper
n_InGeneralWard_mean
n_InGeneralWard_lower
n_InGeneralWard_upper
n_InICU_mean
n_InICU_lower
n_InICU_upper
n_OffVentInICU_mean
n_OffVentInICU_lower
n_OffVentInICU_upper
n_OnVentInICU_mean
n_OnVentInICU_lower
n_OnVentInICU_upper
n_Terminal_mean
n_Terminal_lower
n_Terminal_upper
n_TerminalSmooth_mean
n_TerminalSmooth_lower
n_TerminalSmooth_upper
```

