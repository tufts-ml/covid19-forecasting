The file "Hospitalization_all_locs.csv" contains all the IHME predictions by US State.
gvisani will manually upload the latest forecasts every 1/2 days until he finds a way to download it from the IHME website directly.

**Latest download:** April 21th 2020

## For the TMC specific forecasts
Forecasts are made by taking the IHME predictions for the entire state of Massachusetts, and multiplying them by TMC's market share of the state. 
The default value for market share is 3%, as indicated by TMC folks, but a different value can be used.

Each forecasts is split into 3 spearate csv files: mean, lower bound, and upper bound. These are the IHME bounds adjusted with TMC's market share.
