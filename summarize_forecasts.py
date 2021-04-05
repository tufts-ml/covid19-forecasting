import os
import json
import argparse
import pandas as pd
import tqdm
import glob
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='output/US/MA-20201111-20210111-20210211')
    parser.add_argument('--output_dir', default='results/US/MA-20201111-20210111-20210211')
    parser.add_argument('--output_template', default='PRETRAINED_summary_after_abc_')
    parser.add_argument('--input_csv_file_pattern', default='PRETRAINED_results_after_abc_random_seed*.csv')
    parser.add_argument('--comma_sep_percentiles',
                            type=str,
                            default='1,2.5,5,10,25,50,75,90,95,97.5,99')
    args = parser.parse_args()

    percentiles = list(map(float, args.comma_sep_percentiles.split(',')))

    csv_files = sorted(glob.glob(os.path.join(args.input_dir, args.input_csv_file_pattern)))

    expected_columns = None
    all_arrays = []

    summary_name_flag = False
    L = len(csv_files)
    print("----------------------------------------")
    print("Reading in data from %d simulations" % (L))
    print("----------------------------------------")
    for ll in tqdm.tqdm(range(L)):
        csv_df = pd.read_csv(csv_files[ll])
        csv_df = csv_df[csv_df['timestep'] >= 0]

        if 'summary_name' in csv_df.columns:
            summary_name_values = np.array(csv_df.loc[:, 'summary_name'])
            summary_name_flag = True
            csv_df = csv_df.drop(columns='summary_name')

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
        if summary_name_flag:
            summary_TK = np.transpose(np.vstack([summary_name_values, np.transpose(summary_TK)]))
            exp_columns = ['summary_name'] + list(expected_columns)
        else:
            exp_columns = expected_columns

        df = pd.DataFrame(summary_TK, columns=exp_columns)
        df.to_csv(
            os.path.join(args.output_dir, "%spercentile=%06.2f.csv" % (args.output_template, perc)),
            index=False, float_format='%.2f')

    for func_name, func in [('mean', np.mean), ('stddev', np.std)]:
        summary_TK = func(counts_TKS, axis=2)
        if summary_name_flag:
            summary_TK = np.transpose(np.vstack([summary_name_values, np.transpose(summary_TK)]))
            exp_columns = ['summary_name'] + list(expected_columns)
        else:
            exp_columns = expected_columns

        df = pd.DataFrame(summary_TK, columns=exp_columns)
        df.to_csv(
            os.path.join(args.output_dir, "%s%s.csv" % (args.output_template, func_name)),
            index=False, float_format='%.2f')
