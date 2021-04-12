


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import os
import argparse
from _ctypes import PyObj_FromPtr
import json
import re
from HospitalData_v20210330 import HospitalData

########################################################
# taken from https://stackoverflow.com/questions/13249415/how-to-implement-custom-indentation-when-pretty-printing-with-the-json-module
# prints config file in nice and readable format, so that it is easier to edit

class NoIndent(object):
    """ Value wrapper. """
    def __init__(self, value):
        self.value = value


class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

    def __init__(self, **kwargs):
        # Save copy of any keyword argument values needed for use here.
        self.__sort_keys = kwargs.get('sort_keys', None)
        super(MyEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
                else super(MyEncoder, self).default(obj))

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.
        json_repr = super(MyEncoder, self).encode(obj)  # Default JSON.

        # Replace any marked-up object ids in the JSON repr with the
        # value returned from the json.dumps() of the corresponding
        # wrapped Python object.
        for match in self.regex.finditer(json_repr):
            # see https://stackoverflow.com/a/15012814/355230
            id = int(match.group(1))
            no_indent = PyObj_FromPtr(id)
            json_obj_repr = json.dumps(no_indent.value, sort_keys=self.__sort_keys)

            # Replace the matched id string with json formatted representation
            # of the corresponding Python object.
            json_repr = json_repr.replace(
                            '"{}"'.format(format_spec.format(id)), json_obj_repr)

        return json_repr

########################################################

def generate_config(hObj, directory):

    # initialize config file
    config = {'states': NoIndent(['InGeneralWard', 'OffVentInICU', 'OnVentInICU'])}

    # warm start
    num_warm_up_days = 5

    init_InGeneralWard = int(hObj.get_InGeneralWard_counts()[0] * 1.03)
    init_InGeneralWard_div = init_InGeneralWard // num_warm_up_days
    init_InGeneralWard_mod = init_InGeneralWard % num_warm_up_days
    config['init_num_InGeneralWard'] = {}
    for prev_day in range(-num_warm_up_days+1, 1):
        config['init_num_InGeneralWard'][str(prev_day)] = init_InGeneralWard_div
        if prev_day == 0:
            config['init_num_InGeneralWard'][str(prev_day)] += init_InGeneralWard_mod
    config['init_num_InGeneralWard'] = NoIndent(config['init_num_InGeneralWard'])

    # handling case in which only aggregate ICU counts are shown
    if np.isnan(hObj.get_OffVentInICU_counts()).all() and np.isnan(hObj.get_OnVentInICU_counts()).all():
        init_InICU = int(hObj.get_ICU_counts()[0])
        init_OffVentInICU = int(round(init_InICU / 2))
        init_OnVentInICU = int(round((init_InICU / 2) * 1.03))
    else:
        init_OffVentInICU = int(hObj.get_OffVentInICU_counts()[0])
        init_OnVentInICU = int(hObj.get_OnVentInICU_counts()[0] * 1.03)

    init_OffVentInICU_div = init_OffVentInICU // num_warm_up_days
    init_OffVentInICU_mod = init_OffVentInICU % num_warm_up_days
    config['init_num_OffVentInICU'] = {}
    for prev_day in range(-num_warm_up_days+1, 1):
        config['init_num_OffVentInICU'][str(prev_day)] = init_OffVentInICU_div
        if prev_day == 0:
            config['init_num_OffVentInICU'][str(prev_day)] += init_OffVentInICU_mod
    config['init_num_OffVentInICU'] = NoIndent(config['init_num_OffVentInICU'])

    init_OnVentInICU_div = init_OnVentInICU // num_warm_up_days
    init_OnVentInICU_mod = init_OnVentInICU % num_warm_up_days
    config['init_num_OnVentInICU'] = {}
    for prev_day in range(-num_warm_up_days+1, 1):
        config['init_num_OnVentInICU'][str(prev_day)] = init_OnVentInICU_div
        if prev_day == 0:
            config['init_num_OnVentInICU'][str(prev_day)] += init_OnVentInICU_mod
    config['init_num_OnVentInICU'] = NoIndent(config['init_num_OnVentInICU'])

    # summary statistics names to be considered for ABC, and their relative weights (empirically deterimined, must average to 1.0)
    # these are contingent upon the available data. we handle here only the two most common cases
    # we assume 5-days smoothing for terminal counts for all states
    if np.isnan(hObj.get_OffVentInICU_counts()).all() and np.isnan(hObj.get_OnVentInICU_counts()).all():
        config['summary_statistics_names'] = NoIndent(["n_InGeneralWard", "n_InICU", "n_TERMINAL_5daysSmoothed"])
        config['summary_statistics_weights'] = NoIndent({"n_InGeneralWard": 0.7, "n_InICU": 1.0, "n_TERMINAL_5daysSmoothed": 1.3})
    else:
        config['summary_statistics_names'] = NoIndent(["n_InGeneralWard", "n_OffVentInICU", "n_OnVentInICU",  "n_TERMINAL_5daysSmoothed"])
        config['summary_statistics_weights'] = NoIndent({"n_InGeneralWard": 0.7, "n_OffVentInICU": 0.9, "n_OnVentInICU": 1.1, "n_TERMINAL_5daysSmoothed": 1.3})

    # various numbers
    config['num_past_timesteps'] = num_warm_up_days - 1

    dateFormat = "%Y%m%d"
    d1 = datetime.strptime(hObj._start_date, dateFormat)
    d2 = datetime.strptime(hObj._end_training_date, dateFormat)
    config['num_training_timesteps'] = abs((d2 - d1).days)

    d1 = datetime.strptime(hObj._start_date, dateFormat)
    d2 = datetime.strptime(hObj._end_date, dateFormat)
    config['num_timesteps'] = abs((d2 - d1).days)

    # admissions (General Ward only for this data)
    config['pmf_num_per_timestep_InGeneralWard'] = 'datasets/US/%s/daily_admissions.csv' % directory

    # Parameters!
    # Only placeholder values (except Die after declining OnVentInICU, which is always at 1.0)
    # Assumes 22 days for all durations
    # config["proba_Recovering_given_InGeneralWard"] = 0.1
    # config["proba_Recovering_given_OffVentInICU"] = 0.1
    # config["proba_Recovering_given_OnVentInICU"] = 0.1
    # config["proba_Die_after_Declining_InGeneralWard"] = 0.01
    # config["proba_Die_after_Declining_OffVentInICU"] = 0.02
    # config["proba_Die_after_Declining_OnVentInICU"] = 1.0 # this one MUST be fixed at 1.0
    # config["pmf_duration_Declining_InGeneralWard"] = NoIndent({"1": 0.045, "2": 0.045, "3": 0.045, "4": 0.045, "5": 0.045, "6": 0.045, "7": 0.045, "8": 0.045, "9": 0.045, "10": 0.045, "11": 0.045, "12": 0.045, "13": 0.045, "14": 0.045, "15": 0.045, "16": 0.045, "17": 0.045, "18": 0.045, "19": 0.045, "20": 0.045, "21": 0.045, "22": 0.055})
    # config["pmf_duration_Recovering_InGeneralWard"] = NoIndent({"1": 0.045, "2": 0.045, "3": 0.045, "4": 0.045, "5": 0.045, "6": 0.045, "7": 0.045, "8": 0.045, "9": 0.045, "10": 0.045, "11": 0.045, "12": 0.045, "13": 0.045, "14": 0.045, "15": 0.045, "16": 0.045, "17": 0.045, "18": 0.045, "19": 0.045, "20": 0.045, "21": 0.045, "22": 0.055})
    # config["pmf_duration_Declining_OffVentInICU"] = NoIndent({"1": 0.045, "2": 0.045, "3": 0.045, "4": 0.045, "5": 0.045, "6": 0.045, "7": 0.045, "8": 0.045, "9": 0.045, "10": 0.045, "11": 0.045, "12": 0.045, "13": 0.045, "14": 0.045, "15": 0.045, "16": 0.045, "17": 0.045, "18": 0.045, "19": 0.045, "20": 0.045, "21": 0.045, "22": 0.055})
    # config["pmf_duration_Recovering_OffVentInICU"] = NoIndent({"1": 0.045, "2": 0.045, "3": 0.045, "4": 0.045, "5": 0.045, "6": 0.045, "7": 0.045, "8": 0.045, "9": 0.045, "10": 0.045, "11": 0.045, "12": 0.045, "13": 0.045, "14": 0.045, "15": 0.045, "16": 0.045, "17": 0.045, "18": 0.045, "19": 0.045, "20": 0.045, "21": 0.045, "22": 0.055})
    # config["pmf_duration_Declining_OnVentInICU"] = NoIndent({"1": 0.045, "2": 0.045, "3": 0.045, "4": 0.045, "5": 0.045, "6": 0.045, "7": 0.045, "8": 0.045, "9": 0.045, "10": 0.045, "11": 0.045, "12": 0.045, "13": 0.045, "14": 0.045, "15": 0.045, "16": 0.045, "17": 0.045, "18": 0.045, "19": 0.045, "20": 0.045, "21": 0.045, "22": 0.055})
    # config["pmf_duration_Recovering_OnVentInICU"] = NoIndent({"1": 0.045, "2": 0.045, "3": 0.045, "4": 0.045, "5": 0.045, "6": 0.045, "7": 0.045, "8": 0.045, "9": 0.045, "10": 0.045, "11": 0.045, "12": 0.045, "13": 0.045, "14": 0.045, "15": 0.045, "16": 0.045, "17": 0.045, "18": 0.045, "19": 0.045, "20": 0.045, "21": 0.045, "22": 0.055})

    # save config file as json
    with open(os.path.join(directory, 'config.json'), 'w+') as f:
        f.write(json.dumps(config, cls=MyEncoder, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', default='MA')
    parser.add_argument('--start_date', default='20201111', type=str)
    parser.add_argument('--end_training_date', default='20210111', type=str)
    parser.add_argument('--end_date', default='20210211', type=str)
    args = parser.parse_args()

    state = args.state
    start_date = args.start_date
    end_training_date = args.end_training_date
    end_date = args.end_date

    dateFormat = '%Y%m%d'

    d1 = datetime.strptime(start_date, dateFormat)
    d2 = datetime.strptime(end_training_date, dateFormat)
    num_training_timesteps = abs((d2 - d1).days)

    d1 = datetime.strptime(start_date, dateFormat)
    d2 = datetime.strptime(end_date, dateFormat)
    num_timesteps = abs((d2 - d1).days)

    hObj = HospitalData(state, start_date, end_training_date, end_date)

    admissions = hObj.get_Admission_counts()

    d1 = datetime.strptime(start_date, dateFormat)
    temp_dates = np.array([d1 + timedelta(days=x) for x in range(num_timesteps + 1)])
    dates = []
    for date in temp_dates:
        dates.append(date.strftime(dateFormat))
    dates = np.array(dates)

    n_InGeneralWard = hObj.get_InGeneralWard_counts()
    n_OffVentInICU = hObj.get_OffVentInICU_counts()
    n_OnVentInICU = hObj.get_OnVentInICU_counts()
    n_InICU = hObj.get_ICU_counts()
    n_occupied_beds = n_InGeneralWard + n_InICU
    n_TERMINAL = hObj.get_Death_counts()

    # perform 5-days smoothing on TERMINAL counts, making sure that test information does not leak into training
    cumsum = np.cumsum(np.insert(n_TERMINAL, 0, 0)) 
    middle_smoothed = ((cumsum[5:] - cumsum[:-5]) / float(5)).round().astype(np.int32)
    first_smoothed = np.array([round(np.sum(n_TERMINAL[:3]) / 3), round(np.sum(n_TERMINAL[:4]) / 4)], dtype=np.int32)
    last_smoothed = np.array([round(np.sum(n_TERMINAL[-4:]) / 4), round(np.sum(n_TERMINAL[-3:]) / 3)], dtype=np.int32)
    n_TERMINAL_5daysSmoothed = np.append(first_smoothed, np.append(middle_smoothed, last_smoothed))
    n_TERMINAL_5daysSmoothed[num_training_timesteps] = round(np.sum(n_TERMINAL[num_training_timesteps-2 : num_training_timesteps+1]) / 3)
    n_TERMINAL_5daysSmoothed[num_training_timesteps-1] = round(np.sum(n_TERMINAL[num_training_timesteps-3 : num_training_timesteps+1]) / 4)


    # create directory if it does not yet exist
    directory = '%s-%s-%s-%s' % (state, start_date, end_training_date, end_date)
    if not os.path.exists(directory):
        os.mkdir(directory)

    data_dict = {'timestep': np.arange(admissions.shape[0]),
                 'date': dates,
                 'n_discharged_InGeneralWard': np.full(admissions.shape[0], np.nan),
                 'n_InGeneralWard': n_InGeneralWard,
                 'n_OffVentInICU': n_OffVentInICU,
                 'n_OnVentInICU': n_OnVentInICU,
                 'n_InICU': n_InICU,
                 'n_occupied_beds': n_occupied_beds,
                 'n_TERMINAL': n_TERMINAL,
                 'n_TERMINAL_5daysSmoothed': n_TERMINAL_5daysSmoothed}

    # save filtered and formatted data - still need to smooth death counts. We do it manually on the csv file using excel
    data_df = pd.DataFrame(data_dict)
    data_df.to_csv(os.path.join(directory, 'daily_counts.csv'))

    admissions_dict = {'timestep': np.arange(admissions.shape[0]),
                       'date': dates,
                       'n_admitted_InGeneralWard': np.append(np.array([0]), admissions[1:])} # admissions at timestep 0
                                                                                      # are handled by the warm-up schedule

    admissions_df = pd.DataFrame(admissions_dict)
    admissions_df.to_csv(os.path.join(directory, 'daily_admissions.csv'))

    print('%d admitted patients in this timeframe' % (np.sum(admissions)))

    # generate config file
    generate_config(hObj, directory)

    # timesteps = np.arange(admissions.shape[0])
    # plt.figure(figsize=(16, 4))
    # plt.title('n_admitted_InGeneralWard')
    # plt.plot(timesteps, admissions)
    # plt.show()

    # plt.figure(figsize=(16, 4))
    # plt.title('n_InGeneralWard')
    # plt.plot(timesteps, n_InGeneralWard)
    # plt.show()

    # plt.figure(figsize=(16, 4))
    # plt.title('n_OffVentInICU')
    # plt.plot(timesteps, n_OffVentInICU)
    # plt.show()

    # plt.figure(figsize=(16, 4))
    # plt.title('n_OnVentInICU')
    # plt.plot(timesteps, n_OnVentInICU)
    # plt.show()

    # plt.figure(figsize=(16, 4))
    # plt.title('n_InICU')
    # plt.plot(timesteps, n_InICU)
    # plt.show()

    # plt.figure(figsize=(16, 4))
    # plt.title('n_occupied_beds')
    # plt.plot(timesteps, n_occupied_beds)
    # plt.show()

    # plt.figure(figsize=(16, 4))
    # plt.title('n_TERMINAL')
    # plt.plot(timesteps, n_TERMINAL)
    # plt.show()
