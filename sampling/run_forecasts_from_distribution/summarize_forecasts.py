import os
import json
import argparse
import pandas as pd
import tqdm
import glob
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='output')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--output_template', default='summary_under_prior_')
    parser.add_argument('--input_csv_file_pattern', default='results_under_prior_*.csv')
    parser.add_argument('--comma_sep_percentiles',
        type=str,
        default='1,2.5,5,10,25,50,75,90,95,97.5,99')
    parser.add_argument('--include_los', type=bool, default=False)
    args = parser.parse_args()

    percentiles = list(map(float, args.comma_sep_percentiles.split(',')))

    def input_patterns(original_pattern, include_los):
        yield original_pattern

        if include_los:
            yield original_pattern.replace('results', 'los_results_per_patient', 1)

            for sane_query_suffix in ["",
                "_isTerminal1", "_isTerminal0",
                "_isInICU1", "_isInICU0",
                "_isOnVent1", "_isOnVent0",
                ]:

                yield original_pattern.replace('results', 'los_results_summary%s' % sane_query_suffix, 1)

    def output_patterns(original_pattern, include_los):
        yield original_pattern

        if include_los:
            yield original_pattern.replace('summary', 'summary_of_los_results_per_patient', 1)

            for sane_query_suffix in ["",
                "_isTerminal1", "_isTerminal0",
                "_isInICU1", "_isInICU0",
                "_isOnVent1", "_isOnVent0",
                ]:

                yield original_pattern.replace('summary', 'summary_of_los_results_summary%s' % sane_query_suffix, 1)

    for input_csv_file_pattern, output_template in zip(input_patterns(args.input_csv_file_pattern, args.include_los), output_patterns(args.output_template, args.include_los)):
        print(input_csv_file_pattern)
        
        csv_files = sorted(glob.glob(os.path.join(args.input_dir, input_csv_file_pattern)))

        expected_columns = None
        all_arrays = []

        summary_name_flag = False
        L = len(csv_files)
        print("----------------------------------------")
        print("Reading in data from %d simulations" % (L))
        print("----------------------------------------")
        for ll in tqdm.tqdm(range(L)):
            csv_df = pd.read_csv(csv_files[ll])

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
                os.path.join(args.output_dir, "%spercentile=%06.2f.csv" % (output_template, perc)),
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
                os.path.join(args.output_dir, "%s%s.csv" % (output_template, func_name)),
                index=False, float_format='%.2f')
