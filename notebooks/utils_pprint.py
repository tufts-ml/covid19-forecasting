import pandas as pd
import numpy as np

def pprint_samples(samps_v1_S, samps_v2_S=None, ideal_props={}):
    pd.set_option('display.precision', 3)
    row_dict_list = list()
    for key in ideal_props.keys():
        if key.startswith('p'):
            perc = float(key[1:])
            desired = ideal_props[key]
        else:
            continue
        observed1 = np.percentile(samps_v1_S, perc)
        if samps_v2_S is not None:
            observed2 = np.percentile(samps_v2_S, perc)
        else:
            observed2 = np.nan
        row_dict_list.append(dict(perc=perc, desired=desired, observed1=observed1, observed2=observed2))
    df = pd.DataFrame(row_dict_list)
    df.sort_values('perc', inplace=True)
    print(df.to_string(index=False))
    return df