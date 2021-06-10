
"""
HospitalData.py
Examples in Jupyter Notebook to plot hospitalization data within MA from '20200702' to '20210109' using HospitalData.py and joined_file_20210114.csv being within same folder
--------

"""

import os
import turicreate as tc
import numpy as np
from datetime import datetime,timedelta
from pathlib import Path


import csv
import requests
# from bs4 import BeautifulSoup
import urllib

DATA_FOLDER = 'data/'

class HospitalData(object):
    ''' Represents a Turicreate SFrame of csv data extracted from HHS hospitalization data
    https://healthdata.gov/dataset/covid-19-reported-patient-impact-and-hospital-capacity-state-timeseries
    

    Attributes
    ----------
    data: turicreate SFrame
        Sframe containing columns such as date, state, GeneralWard, OnVentICU, OffVentICU... full dataset, no filters
    filtered_data: turicreate SFrame
        Sframe containing columns such as date, state, GeneralWard, OnVentICU, OffVentICU... filtered dataset based on attr us_state, start_date, end_date
    csv_filename : str
        filepath to csv that was loaded/saved to initialize this HospitalData object
    us_state : str
        2 letter abbreviation for state of interest when using methods get_GeneralWard_counts(), get_OnVentICU_counts(), get_OffVentICU_counts()
    start_date : str
        YYYYMMDD str format of the start date of interest when using methods get_GeneralWard_counts(), get_OnVentICU_counts(), get_OffVentICU_counts()
    end_date : str
        YYYYMMDD str format of the end date of interest when using methods get_GeneralWard_counts(), get_OnVentICU_counts(), get_OffVentICU_counts()
    '''

    def __init__(self, csv_filename, us_state, start_date, end_date, population_data_obj = None):
        ''' Construct a HospitalData from provided input
        Args
        ----
        csv_filename : str
            filepath to csv that was loaded/saved to initialize this HospitalData object
        us_state : str
        2 letter abbreviation for state of interest when using methods get_GeneralWard_counts(), get_OnVentICU_counts(), get_OffVentICU_counts()
        start_date : str
            YYYYMMDD str format of the start date of interest when using methods get_GeneralWard_counts(), get_OnVentICU_counts(), get_OffVentICU_counts()
        end_date : str
            YYYYMMDD str format of the end date of interest when using methods get_GeneralWard_counts(), get_OnVentICU_counts(), get_OffVentICU_counts()
        Returns
        -------
        Newly constructed HospitalData instance
        '''
        self.data = tc.SFrame([])
        self.filtered_data = tc.SFrame([]) 
        self.csv_filename = DATA_FOLDER + csv_filename
        self._us_state = us_state # protected attrib
        self._start_date = start_date # protected attrib
        self._end_date = end_date # protected attrib
        self.population_data_obj = population_data_obj


        self.load_csv_if_exists()

    def get_HHS_data(self):
        ## download csv from hhs website to the current directory of this HospitalData.py file
        csv_name = DATA_FOLDER+'HHS_data_raw.csv'
        today = datetime.now().date()

        if not Path(csv_name).exists()  or datetime.fromtimestamp(os.path.getctime(csv_name)).date() != today  :
            print('DOWNLOADING FRESH HHS DATA.........')
            # hhs_url = 'https://healthdata.gov/dataset/covid-19-reported-patient-impact-and-hospital-capacity-state-timeseries'
            hhs_url = 'https://healthdata.gov/api/views/g62h-syeh/rows.csv?accessType=DOWNLOAD'
            response = requests.get(hhs_url)  
            with open(csv_name, 'w') as f:
                writer = csv.writer(f)
                for line in response.iter_lines():
                    writer.writerow(line.decode('utf-8').split(','))
            tc_data = tc.SFrame(csv_name)
            tc_data['date'] = tc_data.apply(lambda x: x['date'].replace('/',''))

            return tc_data
        else: 
            tc_data = tc.SFrame(csv_name)
            tc_data['date'] = tc_data.apply(lambda x: x['date'].replace('/',''))
            return tc_data
    

    def selective_HHS_data(self, hhs_data):
        ## selecting the relevant column names of the HHS data sources
        # column name meanings found here: https://healthdata.gov/covid-19-reported-patient-impact-and-hospital-capacity-state-data-dictionary
        columns_of_interest_hhs = ['previous_day_admission_adult_covid_confirmed', ]
        columns_of_interest_hhs = columns_of_interest_hhs+ [col + '_coverage' for col in columns_of_interest_hhs] # get the num of hospitals that reports each columns_of_interest
        columns_of_interest_hhs = ['date','state']+columns_of_interest_hhs
        hhs_data_truncated = hhs_data.select_columns(columns_of_interest_hhs)

        return hhs_data_truncated

    def load_csv_if_exists(self):
        self.data = self.selective_HHS_data(self.get_HHS_data())
        self.data.save(self.csv_filename, format='csv') 
        self.filtered_data = self.get_filtered_data()
        
    def get_filtered_data(self):
        # returns Sframe within specified [self.start_date, self.end_date] and self.us_state         
        self.data['selected_row'] = self.data.apply(lambda x: 1 if (x['state']==self.us_state and (int(x['date'])>=int(self._start_date) and int(x['date'])<=int(self._end_date)) ) else 0) #1 means row selected, 0 means row not selected for filtered data
        filtered_data = self.data.filter_by([1],'selected_row').sort(['date'],ascending=True)

        return filtered_data
    
    def get_Admission_counts(self):
        # returns admission counts within [self.start_date, self.end_date] of the self.us_state specified
        prev_admission = (self.filtered_data.apply(lambda x: float('nan') if (None in [x['previous_day_admission_adult_covid_confirmed']]) \
            else x['previous_day_admission_adult_covid_confirmed'])).to_numpy()
#         print(self.end_date)
        end_date_plus_1 = datetime.strftime(datetime.strptime(self.end_date,'%Y%m%d')+timedelta(days=1),'%Y%m%d')
        end_date_plus_1_entry = self.data.filter_by([int(end_date_plus_1)],'date').filter_by([self.us_state],'state')
        print(end_date_plus_1_entry)
        current_admission = np.append(prev_admission[1:],end_date_plus_1_entry['previous_day_admission_adult_covid_confirmed'])
        return current_admission


    @property
    def start_date(self):
        return self._start_date

    @property
    def end_date(self):
        return self._end_date

    @property
    def us_state(self):
        return self._us_state

    # if a user decides to update any of the protected state, then HospitalData.filtered_data should correspondingly be updated 
    @start_date.setter
    def start_date(self, new_value):
        self._start_date = new_value
        if int(self._end_date) > int(self._start_date):
            self.filtered_data = self.get_filtered_data()

    @end_date.setter
    def end_date(self, new_value):
        self._end_date = new_value
        if int(self._end_date) > int(self._start_date):
            self.filtered_data = self.get_filtered_data()

    @us_state.setter
    def us_state(self, new_value):
        self._us_state = new_value
        self.filtered_data = self.get_filtered_data()

        
# h = HospitalData('HHS4.csv', 'MA', '20201001', '20201101', population_data_obj = None)