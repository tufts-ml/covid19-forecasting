'''
    This file is used when we have collected warm-started forecasts on the test set by extending the simulation on the training set
    (only possible when the test set directly follows the training set).

    Then, the forecasts on the test set are included in the concat_file, which is a csv file containing the data frames of all test
    forecasts. Each individual data frame is identifiable via the value in for column 'index'.
'''


import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--concat_file', default='toy_data_experiment/final_results/abc_14_admissions_experiment_ninetupledAdmissions_v2_last_test_forecasts_OnCDCTableReasonable.csv') # 'NHS_results/samples_from_prior_SpikedDurs.json')
    parser.add_argument('--num_samples', default=200)
    parser.add_argument('--output_file', default='toy_data_experiment/final_output/results_14_admissions_experiment_ninetupledAdmissions_v2_TrainingAndTesting_OnCDCTableReasonable_index=0.csv') #'NHS_output/results_new_durations_simulated_annealing_0_manchester_university_nhs_foundation_trust_TrainingOnTraining_for_cdc_table_random_seed=101_sample=None.csv')

    args, unknown_args = parser.parse_known_args()

    unk_keys = map(lambda s: s[2:].strip(), unknown_args[::2])
    unk_vals = unknown_args[1::2]
    unk_dict = dict(zip(unk_keys, unk_vals))

    concat_file = args.concat_file
    output_file = args.output_file
    num_samples = int(args.num_samples)

    concat_df = pd.read_csv(concat_file, index_col=0)

    max_index = max(concat_df['index'])
    dfs = []

    print("----------------------------------------")
    print("Splitting concat_df into %d data frames" % (max_index + 1))
    print("----------------------------------------")
    for index in tqdm(range(max_index + 1)):
        df = concat_df[concat_df['index'] == index]
        df = df.drop(columns='index')
        dfs.append(df[df['timestep'] >= 0])

    print("----------------------------------------")
    print("Checking length of %d data frames" % (num_samples))
    print("----------------------------------------")
    for index in tqdm(range(num_samples)):
        df = dfs[len(dfs) - 1 - index]
        if df['timestep'].shape[0] < 90:
            print('\nNot enough testing predictions')
            exit(1)

    print("----------------------------------------")
    print("Saving %d data frames to csv files" % (num_samples))
    print("----------------------------------------")
    for index in tqdm(range(num_samples)):
        df = dfs[len(dfs) - 1 - index]
        filename = output_file.replace('index=0', 'index=%d' % index)
        df.to_csv(filename)