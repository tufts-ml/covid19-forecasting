'''
Scrape per-hospital total census and ICU census from daily reports on mass.gov 

Reporting at hospital level started at April 29, 2020, with new data released every day
https://www.mass.gov/info-details/archive-of-covid-19-cases-in-massachusetts#april-2020-
'''

import os
from io import BytesIO, StringIO
from zipfile import ZipFile
from urllib.request import urlopen
import datetime
import requests
import pandas as pd
import numpy as np
import string
import time

num_per_month = dict(april=4, may=5, june=6, july=7, august=8, september=9, october=10, november=11, december=12)
max_days_per_month = dict(april=30, may=31, june=30, july=31, august=31, september=30, october=31, november=30, december=31)

today = datetime.datetime.now().date()
first_date_data_available = datetime.date.fromisoformat("{year}-{monthid:02d}-{day:02d}".format(year=2020, monthid=num_per_month['april'], day=29))

def sanitize_string(s):
    ''' Make string suitable for filesystem paths or readable code (remove punctuation and whitespace)

    Args
    ----
    s : str

    Returns
    -------
    s_clean : str

    Examples
    --------
    >>> sanitize_string("Boston Children's Hospital")
    boston_childrens_hospital
    '''
    punctuation_remover = str.maketrans('', '', string.punctuation)
    return '_'.join(filter(None, [word.strip().lower().translate(punctuation_remover) for word in s.split()]))

def sanitize_col_names(df, delimiter=None):
    mapper = dict()
    for key in df.columns:
        mapper[key] = '_'.join(filter(None, map(sanitize_string, key.split(delimiter))))
    return df.rename(columns=mapper), mapper

max_trials_per_day = 3

row_list = list()

for month in num_per_month.keys():
    for day in range(1, 1+max_days_per_month[month]):

        cur_date = datetime.date.fromisoformat("{year}-{monthid:02d}-{day:02d}".format(year=2020, monthid=num_per_month[month], day=day))

        if cur_date >= first_date_data_available and cur_date < today:

            if (day % 10) == 0:
                print("WAITING 63 seconds to avoid too many requests.")
                time.sleep(63)
        
            query_url = "https://www.mass.gov/doc/covid-19-raw-data-{month}-{day}-2020/download".format(month=month, day=day)
            print("Requesting data for date %s from URL: %s" % (str(cur_date), query_url))

            trial = 1
            request_successful = False
            while trial <= max_trials_per_day and (not request_successful):

                # Request file from mass.gov
                r = requests.get(query_url, stream=True)

                # Make status code for the request is good
                if r.status_code != 200:
                    print("trial %d: ERROR with status code %d ... trying again." % (trial, r.status_code))
                    trial += 1
                    time.sleep(2 + trial)

                    if r.status_code == 429:
                        print("STATUS CODE 429: Waiting 63 seconds to avoid rate limiting.")
                        time.sleep(63)
                    continue

                # Make sure zip file is good
                try:
                    with ZipFile(BytesIO(r.content)) as zipfile:
                        assert len(zipfile.namelist()) > 0
                except Exception as e:
                    trial += 1
                    print("trial %d: ERROR reading zip file... trying again." % (trial))
                    time.sleep(2 + trial)
                    continue
                request_successful = True

                with ZipFile(BytesIO(r.content)) as zipfile:
                    #print(zipfile.namelist())
                    for name in zipfile.namelist():
                        ## in May, format is to use ....External.xlsx
                        ## after late May, format is to use HospCensusBedAvailable.xlsx
                        if name.endswith('.xlsx') and (name.count("HospCensusBedAvailable") or name.count("External")):
                            print("--->>> Reading file: %s" % name)
                            try:
                                ## Will load each sheet as a df into a dict where keys are sheet names
                                df_per_sheet = pd.read_excel(zipfile.read(name), sheet_name=None)    
                                df = None
                                for key in df_per_sheet.keys():
                                    if key.lower().count('covid census'):
                                        df = df_per_sheet[key]
                                        break
                                if df is None:
                                    # intentionally call knowing this will fail so we get the right error
                                    df = pd.read_excel(zipfile.read(name), sheet_name='Hospital COVID Census')
                                #print("\t".join(df.columns))
                                #print(df.head())
                                if 'Hospital County and Zip Code' in df:
                                    a, b = zip(*map(lambda s: s.split('-'), df['Hospital County and Zip Code'].values))
                                    county_vals = [str.strip(x) for x in a]
                                    zip_vals = [str.strip(x) for x in b]
                                    df['Hospital County'] = county_vals
                                    df['Zip Code'] = zip_vals
                                df['date'] = str(cur_date)
                                sane_df, _ = sanitize_col_names(df)
                                row_list.append(sane_df)
                            except Exception as e:
                                print(str(e).split('\n')[0])
                                from IPython import embed; embed()
                                pass
                if not request_successful:
                    print("FAILED all %d request trials. Skipping this date." % max_trials_per_day)

    if cur_date >= today:
        break

all_df, mapper = sanitize_col_names(pd.concat(row_list), delimiter='_')
df = all_df.query("hospital_county == 'Suffolk'")
for hospital_name in df['hospital_name'].unique():
    site_df = df.query("hospital_name == '%s'" % hospital_name).copy()

    sane_name = sanitize_string(hospital_name)
    if sane_name.count('children'):
        continue
    output_csv_path = os.path.join('csv', '%s_%s_to_%s.csv' % (sane_name, all_df['date'].min(), all_df['date'].max()))

    del site_df['hospital_county_and_zip_code']
    del site_df['hospital_name']
    del site_df['hospital_county']
    del site_df['pdfpage']
    del site_df['zip_code']
    clean_site_df = site_df.set_index('date').copy()

    num_df = clean_site_df.select_dtypes('number')
    site_count = np.nansum(num_df.values)
    if site_count < 50:
        print("Skipping site %s with only %d total cases" % (sane_name, site_count))
        continue
    else:
        print("Recording site %s to CSV with %d total cases" % (sane_name, site_count))
    clean_site_df.to_csv(output_csv_path, index=True)

