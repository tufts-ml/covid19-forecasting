import os
import json
import argparse
import pandas as pd
import tqdm
import glob
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='example_output') #'NHS_output')
    parser.add_argument('--output_dir', default='example_output') #'NHS_output')
    parser.add_argument('--output_template', default='summary_health_experiment_bad_durations_2_end_100-10_') #'summary_12000Iters_university_hospitals_birmingham_nhs_foundation_trust_TrainingOnTraining_20MaxEach_uniform_')
    parser.add_argument('--input_csv_file_pattern', default='results_health_experiment_bad_durations_2_end_100-10_*.csv') #'results_12000Iters_university_hospitals_birmingham_nhs_foundation_trust_TrainingOnTraining_20MaxEach_uniform_*.csv')
    parser.add_argument('--comma_sep_percentiles',
        type=str,
        default='1,2.5,5,10,25,50,75,90,95,97.5,99')
    args = parser.parse_args()

    percentiles = list(map(float, args.comma_sep_percentiles.split(',')))

    csv_files = sorted(glob.glob(os.path.join(args.input_dir, args.input_csv_file_pattern)))

    expected_columns = None
    all_arrays = []

    L = len(csv_files)
    print("----------------------------------------")
    print("Reading in data from %d simulations" % (L))
    print("----------------------------------------")
    for ll in tqdm.tqdm(range(L)):
        csv_df = pd.read_csv(csv_files[ll])
        if expected_columns is None:
            expected_columns = csv_df.columns
        else:
            L = len(expected_columns)
            is_same_length = (L == len(csv_df.columns))
            if not is_same_length:
                raise ValueError("Bad columns: Length mismatch")
            for cc in range(L):
                if expected_columns[cc] != csv_df.columns[cc]:
                    raise ValueError("Bad columns: Element mismatch")
        all_arrays.append(csv_df.values)

    P = len(percentiles)
    print("----------------------------------------")
    print("Computing summaries for %d percentiles" % (P))
    print("----------------------------------------")
    counts_TKS = np.dstack(all_arrays)
    for perc in percentiles:
        summary_TK = np.percentile(counts_TKS, perc, axis=2)
        df = pd.DataFrame(summary_TK, columns=expected_columns)
        df.to_csv(
            os.path.join(args.output_dir, "%spercentile=%06.2f.csv" % (args.output_template, perc)),
            index=False, float_format='%.2f')

    for func_name, func in [('mean', np.mean), ('stddev', np.std)]:
        summary_TK = func(counts_TKS, axis=2)
        df = pd.DataFrame(summary_TK, columns=expected_columns)
        df.to_csv(
            os.path.join(args.output_dir, "%s%s.csv" % (args.output_template, func_name)),
            index=False, float_format='%.2f')
