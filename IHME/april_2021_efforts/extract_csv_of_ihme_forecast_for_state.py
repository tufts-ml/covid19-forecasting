import argparse
import numpy as np
import pandas as pd
import os
import itertools

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='/data/ihme/2021-01-15')
    parser.add_argument(
        '--test_start_date',
        type=str,
        default='2020-12-11')
    parser.add_argument(
        '--test_stop_date',
        type=str,
        default='2021-02-11')
    parser.add_argument(
        '--target_location_name',
        type=str,
        default='Massachusetts')
    args = parser.parse_args()

    all_data_iterator = pd.read_csv(
        os.path.join(args.dataset_path, 'reference_hospitalization_all_locs.csv'),
        iterator=True, chunksize=10000)

    match_df_list = list()
    for df in all_data_iterator:
        match_df_list.append(
            df.query("location_name == '%s'" % args.target_location_name))
    match_df = pd.concat(match_df_list)
    start_date = pd.to_datetime(args.test_start_date)
    stop_date = pd.to_datetime(args.test_stop_date)

    match_df['date_ts'] = match_df['date'].astype(np.datetime64)
    date_vals = match_df['date_ts'].values
    target_bmask = np.logical_and(
            date_vals >= start_date,
            date_vals <= stop_date)
    test_match_df = match_df.iloc[target_bmask].copy()

    rel_column_names = (
        ['location_name', 'date']
        + ['allbed_mean', 'allbed_lower', 'allbed_upper']
        + ['ICUbed_mean', 'ICUbed_lower', 'ICUbed_upper']
        + ['InvVen_mean', 'InvVen_lower', 'InvVen_upper']
        + ['deaths_mean', 'deaths_lower', 'deaths_upper']
        + ['deaths_mean_smoothed', 'deaths_lower_smoothed', 'deaths_upper_smoothed']
        )

    stage_name = 'n_OnVentInICU'
    for suffix in ['_mean', '_lower', '_upper']:
         test_match_df[stage_name + suffix] = test_match_df['InvVen' + suffix]

    stage_name = 'n_InICU'
    for suffix in ['_mean', '_lower', '_upper']:
         test_match_df[stage_name + suffix] = \
                 test_match_df['ICUbed' + suffix]

    stage_name = 'n_OffVentInICU'
    for suffix in ['_mean', '_lower', '_upper']:
         test_match_df[stage_name + suffix] = \
                 test_match_df['ICUbed' + suffix] - test_match_df['InvVen' + suffix]

    stage_name = 'n_InGeneralWard'
    for suffix in ['_mean', '_lower', '_upper']:
        test_match_df[stage_name + suffix] = \
                test_match_df['allbed' + suffix] - test_match_df['ICUbed' + suffix]

    stage_name = 'n_Admitted_InGeneralWard'
    for suffix in ['_mean', '_lower', '_upper']:
        test_match_df[stage_name + suffix] = \
                test_match_df['admis' + suffix]

    stage_name = 'n_Terminal'
    for (smoothed_suff, smoothed_new_suff) in [('',''), ('_smoothed', 'Smooth')]:
        for suffix in ['_mean', '_lower', '_upper']:
            test_match_df[stage_name + smoothed_new_suff + suffix] = \
                test_match_df['deaths' + suffix + smoothed_suff]

    keep_names = (
        ['location_name', 'date']
        + ['n_Admitted_InGeneralWard' + suffix for suffix in ['_mean', '_lower', '_upper']]
        + ['n_InGeneralWard' + suffix for suffix in ['_mean', '_lower', '_upper']]
        + ['n_InICU' + suffix for suffix in ['_mean', '_lower', '_upper']]
        + ['n_OffVentInICU' + suffix for suffix in ['_mean', '_lower', '_upper']]
        + ['n_OnVentInICU' + suffix for suffix in ['_mean', '_lower', '_upper']]
        + ['n_Terminal' + xs + suffix for (xs,suffix) in 
            itertools.product(['', 'Smooth'], ['_mean', '_lower', '_upper'])]
        )
    test_match_df['date'] = [s.replace("-", "") for s in test_match_df['date'].values]    
    print(test_match_df[keep_names].head())

    
    test_match_df.to_csv(
        'ihme_forecasts_for_%s_%s-%s.csv' % (
            args.target_location_name.replace(' ', ''),
            args.test_start_date.replace('-', ''),
            args.test_stop_date.replace('-', '')),
        columns=keep_names,
        header=True,
        float_format='%.4f',
        index=False)
