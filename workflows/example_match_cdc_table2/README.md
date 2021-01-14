Demo that we recover approximate summary stats given in CDC's Table 2:

<https://www.cdc.gov/coronavirus/2019-ncov/hcp/planning-scenarios.html#table-2>

Simulation uses 30000 patients to show proper numbers.


Parameters
----------

To see the concrete parameters used, see here:

https://github.com/tufts-ml/covid19-forecasting/blob/master/workflows/example_match_cdc_table2/params.json

To understand the reasoning behind these parameters, see here:

https://docs.google.com/presentation/d/1seAYpADSLvuYK6RXZKFLsPeRebhteveze46AK1OrIcw/edit?usp=sharing

Usage
-----

To run a simulation, just do

$ cd workflows/example_match_cdc_table2
$ snakemake --cores 1 all

*Expected output*

```
----------------------------------------
Loaded SemiMarkovModel from config_file:
----------------------------------------
State #0 InGeneralWard
    prob. 0.640 recover
    prob. 0.360 advance to state OffVentInICU
State #1 OffVentInICU
    prob. 0.380 recover
    prob. 0.620 advance to state OnVentInICU
State #2 OnVentInICU
    prob. 0.570 recover
    prob. 0.430 advance to state TERMINAL
random_seed=1001 <<<
--------------------------------------------
Simulating for  50 timesteps with seed  1001
Initial population size: 30000
--------------------------------------------
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 20739.24it/s]
----------------------------------------
Writing results to .../git/covid19-forecasting/example_output/match_cdc_table2/results-random_seed=1001.csv
----------------------------------------
```


Check 1: Overall probabilities of key outcomes
----------------------------------------------

We want to match the following facts from Table 2 (assuming age 50-64):

* among all simulated patients, chance of going to ICU is 36%
* among all simulated patients, chance of going to vent is 22%
* among all simulated patients, chance of dying is 10%

*Proof*: Inspect the percentile_los summary file for all subjects.

Check if the values in the mean row match the desired values above.

```
$ cut -d, -f1-4 ../../example_output/match_cdc_table2/los_results_summary-random_seed\=1001.csv | column -s, -t | head -n3
summary_name        is_Terminal  is_InICU  is_OnVent
count               30000.00     30000.00  30000.00
mean                0.10         0.36      0.22
```

Check 2: Durations of non-ICU stays
------------------------------------

We want to match the following facts from Table 2 (assuming age 50-64):

Among all patients who did NOT go to ICU,

* the 25/50/75 percentiles of length of stay are 2 / 4 / 7

*Proof*: Inspect the percentile_los summary file for subjects satisfying "isInICU0".

Check if the values match the desired values above.

```
$ cut -d, -f1-5 ../../example_output/match_cdc_table2/los_results_summary_isInICU0-random_seed\=1001.csv | column -s, -t | grep "summary\|_25\|_50\|_75"
summary_name        is_Terminal  is_InICU  is_OnVent  duration_All
percentile_25.000   0.00         0.00      0.00       2.00
percentile_50.000   0.00         0.00      0.00       4.00
percentile_75.000   0.00         0.00      0.00       7.00
```

Check 3: Durations of ICU stays
--------------------------------

We want to match the following facts from Table 2 (assuming age 50-64):

Among all patients who DID go to ICU,

* the 25/50/75 percentiles of length of stay are 8 / 14 / 25

*Proof*: Inspect the percentile_los summary file for subjects satisfying "isInICU1".

Check if the values match the desired values above.

We get very close:  10 / 14 / 21

```
$ cut -d, -f1-5 ../../example_output/match_cdc_table2/los_results_summary_isInICU1-random_seed\=1001.csv | column -s, -t | grep "summary\|_25\|_50\|_75"
summary_name        is_Terminal  is_InICU  is_OnVent  duration_All
percentile_25.000   0.00         1.00      0.00       10.00
percentile_50.000   0.00         1.00      1.00       14.00
percentile_75.000   1.00         1.00      1.00       21.00
```


Check 4: Durations on the ventilator
--------------------------------

We want to match the following facts from Table 2:

Among all patients who DID go to ICU,

* the 25/50/75 percentiles of days on the ventilator are 2 / 6 / 12

*Proof*: Inspect the percentile_los summary file for subjects satisfying "isOnVent1".

Check if the values match the desired values above.

We get very close:  3 / 6 / 12

```
$ cut -d, -f1-8 ../../example_output/match_cdc_table2/los_results_summary_isOnVent1-random_seed\=1001.csv | column -s, -t | grep "summary\|_25\|_50\|_75"
summary_name        is_Terminal  is_InICU  is_OnVent  duration_All  duration_InGeneralWard  duration_OffVentInICU  duration_OnVentInICU
percentile_25.000   0.00         1.00      1.00       12.00         4.00                    3.00                   3.00
percentile_50.000   0.00         1.00      1.00       18.00         6.00                    4.00                   6.00
percentile_75.000   1.00         1.00      1.00       25.00         9.00                    5.00                   12.00
```
