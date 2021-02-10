import numpy as np
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--concat_file', default='NHS_results/abc_WarmStartOnTesting_2_south_tees_hospitals_nhs_foundation_trust_TrainingAndTesting_last_test_forecasts_for_cdc_table.csv') # 'NHS_results/samples_from_prior_SpikedDurs.json')
    parser.add_argument('--num_samples', default=100)
    parser.add_argument('--output_file', default='NHS_output/results_WarmStartOnTesting_2_south_tees_hospitals_nhs_foundation_trust_TrainingOnTesting_for_cdc_table_index=0.csv') #'NHS_output/results_new_durations_simulated_annealing_0_manchester_university_nhs_foundation_trust_TrainingOnTraining_for_cdc_table_random_seed=101_sample=None.csv')

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
    for index in range(max_index + 1):
        df = concat_df[concat_df['index'] == index]
        df = df.drop(columns='index')
        dfs.append(df)

    for index in range(num_samples):
        df = dfs[len(dfs) - 1 - index]
        filename = output_file.replace('index=0', 'index=%d' % index)
        df.to_csv(filename)