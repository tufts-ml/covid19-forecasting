'''
Usage:
    python get_IHME_forecasts_for_TMC.py --market_share [0.03]
'''

import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--market_share', default='0.03')
    args = parser.parse_args()
    market_share = float(args.market_share)

    df = pd.read_csv("IHME_forecasts/Hospitalization_all_locs.csv")
    mass = df.loc[df['location_name'] == 'Massachusetts']

    for interval in ['mean', 'lower', 'upper']:
        adict = {}
        adict['date'] = mass['date']
        adict['n_InGeneralWard'] = (mass['allbed_' + interval] - mass['ICUbed_' + interval])*market_share
        adict['n_InICU'] = (mass['ICUbed_' + interval])*market_share
        adict['n_OnVentInICU'] = (mass['InvVen_' + interval])*market_share
        adict['n_TERMINAL'] = (mass['deaths_' + interval])*market_share
        df = pd.DataFrame(adict)
        df.to_csv("IHME_forecasts/IHME-for-TMC-mktshare=%.3f-%s.csv" % (market_share, interval))

