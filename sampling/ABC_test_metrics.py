import os
import json
import argparse
import pandas as pd
import tqdm
import glob
import numpy as np

# This function maps the statistics outputted by the hospital model
#   to the statistics of the true data
# 
# This function is meant to be customized as needed. It currently includes conversions for NHS data.
def compute_true_summary_statistics(csv_df, expected_columns):
    new_dict = {}

    for column in expected_columns:
        if column == 'timestep' or column == 'dates' or column == 'n_admits':
            continue
        elif column == 'n_discharges':
            new_dict[column] = csv_df['n_discharged_InGeneralWard'] + csv_df['n_discharged_OffVentInICU'] + csv_df['n_discharged_OnVentInICU']
        elif column == 'n_occupied_beds':
            new_dict[column] = csv_df['n_InGeneralWard'] + csv_df['n_OffVentInICU'] + csv_df['n_OnVentInICU']
        else:
            new_dict[column] = csv_df[column]

    return pd.DataFrame(new_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='NHS_output')
    parser.add_argument('--output_dir', default='NHS_results/metrics')
    parser.add_argument('--output_template', default='new_durations_simulated_annealing_9_south_tees_hospitals_nhs_foundation_trust_TrainingOnTraining_for_cdc_tableSpikedDurs')
    parser.add_argument('--true_stats', default='NHS_data/new_data/formatted_data/south_tees_hospitals_nhs_foundation_trust_Training.csv')
    parser.add_argument('--input_simulations_pattern', default='results_new_durations_simulated_annealing_9_south_tees_hospitals_nhs_foundation_trust_TrainingOnTraining_for_cdc_tableSpikedDurs_*.csv')
    parser.add_argument('--input_summaries_template', default='summary_new_durations_simulated_annealing_9_south_tees_hospitals_nhs_foundation_trust_TrainingOnTraining_for_cdc_tableSpikedDurs_')
    parser.add_argument('--coverages',
        type=str,
        default='2.5_97.5,10_90,25_75')
    parser.add_argument('--comma_sep_expected_columns',
                        default='n_discharges,n_occupied_beds,n_OnVentInICU')
    args = parser.parse_args()

    sims_csv_files = sorted(glob.glob(os.path.join(args.input_dir, args.input_simulations_pattern)))

    if args.comma_sep_expected_columns == 'None':
        expected_columns = None
    else:
        expected_columns = args.comma_sep_expected_columns.split(',')

    all_arrays = []
    print("------------------------------------------------------")
    print("Computing likelihood of true data under empirical pmf")
    print("------------------------------------------------------")   

    L = len(sims_csv_files)
    num_samples = len(sims_csv_files)
    print("----------------------------------------")
    print("Reading in data from %d simulations" % (L))
    print("----------------------------------------")
    for ll in tqdm.tqdm(range(L)):
        csv_df = pd.read_csv(sims_csv_files[ll], index_col='timestep')
        if expected_columns is None:
            expected_columns = csv_df.columns
        else:
            csv_df = compute_true_summary_statistics(csv_df, expected_columns)
            L = len(expected_columns)
            is_same_length = (L == len(csv_df.columns))
            if not is_same_length:
                raise ValueError("Bad columns: Length mismatch")
            for cc in range(L):
                if expected_columns[cc] != csv_df.columns[cc]:
                    raise ValueError("Bad columns: Element mismatch")
        
        all_arrays.append(csv_df.values)

    counts_TKS = np.dstack(all_arrays)

    true_df = pd.read_csv(args.true_stats, index_col='timestep')

    # drop columns from true_df that are not among the expected columns
    original_columns = true_df.columns
    for col in original_columns:
        if col not in expected_columns:
            true_df = true_df.drop(col, axis=1)

    true_counts = true_df.values
    pmf_counts = np.sum(np.equal(counts_TKS, np.dstack([true_counts for i in range(num_samples)])), axis=2)
    pmf_scores = pmf_counts / num_samples

    df = pd.DataFrame(pmf_scores, columns=expected_columns)
    df.to_csv(
        os.path.join(args.output_dir, "empirical_pmf_scores_%s.csv" % (args.output_template)),
        index=False, float_format='%.2f')

    print("------------------------------------------------------")
    print("Computing MAE for mean")
    print("------------------------------------------------------")

    mean_counts = compute_true_summary_statistics(pd.read_csv(os.path.join(args.input_dir, "%smean.csv" % (args.input_summaries_template)), index_col='timestep'), expected_columns).values
    mae_scores = np.mean(np.abs(mean_counts - true_counts), axis=0).reshape((1, len(expected_columns)))
    
    df = pd.DataFrame(mae_scores, columns=expected_columns)
    df.to_csv(
        os.path.join(args.output_dir, "mae_scores_%s.csv" % (args.output_template)),
        index=False, float_format='%.2f')

    print("------------------------------------------------------")
    print("Computing coverage for given ranges")
    print("------------------------------------------------------")

    T = true_counts.shape[0]
    # expected_columns = list(true_df.columns) # WARNING: must be in the order of the csv files!
    coverages = args.coverages.split(',')
    results_dict = {'coverages': coverages}
    for coverage in coverages:
        low, high = list(map(float, coverage.split('_')))
        low_counts = compute_true_summary_statistics(pd.read_csv(os.path.join(args.input_dir, "%spercentile=%06.2f.csv" % (args.input_summaries_template, low)), index_col='timestep'), expected_columns).values
        high_counts = compute_true_summary_statistics(pd.read_csv(os.path.join(args.input_dir, "%spercentile=%06.2f.csv" % (args.input_summaries_template, high)), index_col='timestep'), expected_columns).values

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