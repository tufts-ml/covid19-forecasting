import os
import json
import argparse
import pandas as pd
import tqdm
import glob
import numpy as np

# This function maps the statistics outputted by the hospital model
#   to the statistics of the true data
def compute_true_summary_statistics(csv_df, expected_columns):
    new_dict = {}

    for column in expected_columns:
        if column == 'timestep' or column == 'date':
            continue
        elif column == 'n_occupied_beds':
            new_dict[column] = csv_df['n_InGeneralWard'] + csv_df['n_OffVentInICU'] + csv_df['n_OnVentInICU']
        elif column == 'n_InICU':
            new_dict[column] = csv_df['n_OffVentInICU'] + csv_df['n_OnVentInICU']
        else:
            new_dict[column] = csv_df[column]

    return pd.DataFrame(new_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='results/US/MA-20201111-20210111-20210211')
    parser.add_argument('--output_dir', default='results/US/MA-20201111-20210111-20210211')
    parser.add_argument('--config_file', default='results/US/MA-20201111-20210111-20210211/config_after_abc.json')
    parser.add_argument('--output_template', default='after_abc')
    parser.add_argument('--true_stats', default='results/US/MA-20201111-20210111-20210211/daily_counts.csv')
    parser.add_argument('--input_summaries_template', default='summary_MA_NovToFeb_61days_OnCDCTableReasonable_')
    parser.add_argument('--coverages',
                            type=str,
                            default='2.5_97.5,10_90,25_75')
    parser.add_argument('--comma_sep_expected_columns',
                            default='n_InGeneralWard,n_OffVentInICU,n_OnVentInICU,n_InICU,n_occupied_beds,n_TERMINAL,n_TERMINAL_5daysSmoothed')
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = json.load(f)
    num_training_timesteps = config['num_training_timesteps']

    if args.comma_sep_expected_columns == 'None':
        expected_columns = None
    else:
        expected_columns = args.comma_sep_expected_columns.split(',')

    true_df = pd.read_csv(args.true_stats)
    true_df = true_df[true_df['timestep'] > num_training_timesteps]

    # drop columns from true_df that are not among the expected columns
    original_columns = true_df.columns
    for col in original_columns:
        if col not in expected_columns:
            true_df = true_df.drop(col, axis=1)

    true_counts = true_df.values

    print("------------------------------------------------------")
    print("Computing MAE for mean")
    print("------------------------------------------------------")
    mean_df = pd.read_csv(os.path.join(args.input_dir, "%smean.csv" % (args.input_summaries_template)))
    mean_df = mean_df[mean_df['timestep'] > num_training_timesteps]
    
    if 'n_TERMINAL_5daysSmoothed' in expected_columns:
        mean_df['n_TERMINAL_5daysSmoothed'] = np.copy(mean_df['n_TERMINAL'])

    mean_df = compute_true_summary_statistics(mean_df, expected_columns)
    original_columns = mean_df.columns
    for col in original_columns:
        if col not in expected_columns:
            mean_df = mean_df.drop(col, axis=1)

    mean_counts = mean_df.values
    mae_scores = np.mean(np.abs(mean_counts - true_counts), axis=0).reshape((1, len(expected_columns)))
    max_scores = np.max(np.abs(mean_counts - true_counts), axis=0).reshape((1, len(expected_columns)))
    mean_true_counts = np.mean(true_counts, axis=0).reshape((1, len(expected_columns)))
    scores = np.vstack([mae_scores, max_scores, mean_true_counts])
    rows = np.array(['Mean Absolute Error', 'Maximum Absolute Error', 'Mean of True Counts']).reshape((3, 1))
    
    df = pd.DataFrame(np.hstack([rows, scores]), columns=['Metric']+expected_columns)
    df.to_csv(
        os.path.join(args.output_dir, "mae_scores_%s.csv" % (args.output_template)),
        index=False, float_format='%.2f')

    print("------------------------------------------------------")
    print("Computing coverage for given ranges")
    print("------------------------------------------------------")

    T = true_counts.shape[0]
    coverages = args.coverages.split(',')
    results_dict = {'Coverages (%)': list(map(lambda x: '%d' % (float(x.split('_')[1]) - float(x.split('_')[0])), coverages))}
    for coverage in coverages:
        low, high = list(map(float, coverage.split('_')))

        low_df = pd.read_csv(os.path.join(args.input_dir, "%spercentile=%06.2f.csv" % (args.input_summaries_template, low)))
        low_df = low_df[low_df['timestep'] > num_training_timesteps]

        if 'n_TERMINAL_5daysSmoothed' in expected_columns:
            low_df['n_TERMINAL_5daysSmoothed'] = np.copy(low_df['n_TERMINAL'])

        low_df = compute_true_summary_statistics(low_df, expected_columns)
        original_columns = low_df.columns
        for col in original_columns:
            if col not in expected_columns:
                low_df = low_df.drop(col, axis=1)

        high_df = pd.read_csv(os.path.join(args.input_dir, "%spercentile=%06.2f.csv" % (args.input_summaries_template, high)))
        high_df = high_df[high_df['timestep'] > num_training_timesteps]

        if 'n_TERMINAL_5daysSmoothed' in expected_columns:
            high_df['n_TERMINAL_5daysSmoothed'] = np.copy(high_df['n_TERMINAL'])

        high_df = compute_true_summary_statistics(high_df, expected_columns)
        original_columns = high_df.columns
        for col in original_columns:
            if col not in expected_columns:
                high_df = high_df.drop(col, axis=1)

        low_counts = low_df.values
        high_counts = high_df.values

        is_in_low = true_counts > low_counts
        is_in_high = true_counts < high_counts
        is_in_range = np.logical_and(is_in_low, is_in_high)

        counts_in_range = np.sum(is_in_range, axis=0)
        assert len(expected_columns) == counts_in_range.shape[0]

        for i, column in enumerate(expected_columns):
            if column not in results_dict:
                results_dict[column] = []
            results_dict[column].append(counts_in_range[i] / T)

    df = pd.DataFrame(results_dict)
    df.to_csv(
        os.path.join(args.output_dir, "coverage_percentages_%s.csv" % (args.output_template)),
        index='coverages', float_format='%.2f')   