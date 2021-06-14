import os
import turicreate as tc
import autograd.numpy as np
from datetime import datetime,timedelta
from pathlib import Path
import csv
import requests
import urllib


class PopulationData(object):
    ''' Represents a Turicreate SFrame of csv data extracted from covidEstim project state-level estimates.csv

    Attributes
    ----------
    data: turicreate SFrame
        Sframe containing columns such as date, state,infections, symptomatic, severe, Rt... full dataset, no filters
    filtered_data: turicreate SFrame
        Sframe containing columns such as date, state,infections, symptomatic, severe, Rt... filtered dataset based on attr us_state, start_date to most-recently-covidEstimated-date
    forecasted_data: turicreate SFrame
        Sframe containing columns such as date, state,infections, symptomatic, severe, Rt... forecasted dataset from most-recently-covidEstimated-date to end_date
    csv_filename : str
        filepath to csv that was loaded/saved to initialize this PopulatiOndata object
    us_state : str
        2 letter abbreviation for state of interest when using methods get_GeneralWard_counts(), get_OnVentICU_counts(), get_OffVentICU_counts()
    start_date : str
        YYYYMMDD str format of the start date of interest when using methods get_GeneralWard_counts(), get_OnVentICU_counts(), get_OffVentICU_counts()
    end_date : str
        YYYYMMDD str format of the end date of interest when using methods get_GeneralWard_counts(), get_OnVentICU_counts(), get_OffVentICU_counts()
    '''

    def __init__(self, csv_filename, us_state, start_date, end_date):
        ''' Construct a PopulatiOndata from provided input
        Args
        ----
        csv_filename : str
            filepath to csv that was loaded/saved to initialize this PopulatiOndata object
        us_state : str
        full state name of interest 
        start_date : str
            YYYYMMDD str format of the start date of interest 
        end_date : str
            YYYYMMDD str format of the end date of interest 
        Returns
        -------
        Newly constructed PopulatiOndata instance
        '''

        self.csv_filename = csv_filename
        self._us_state = us_state # protected attrib
        self._start_date = start_date # protected attrib
        self._end_date = end_date # protected attrib
        

        self.load_csv_if_exists()
        self.filtered_data = self.get_filtered_data() # captures the time interval from start_date to latest estimates.csv dates



    def load_csv_if_exists(self):
        csv_name = self.csv_filename
        today = datetime.now().date()        
        if not Path(csv_name).exists() or datetime.fromtimestamp(os.path.getctime(csv_name)).date() != today  :
            url = 'https://covidestim.s3.us-east-2.amazonaws.com/latest/state/estimates.csv'
            response = requests.get(url)  
            with open(csv_name, 'w') as f:
                writer = csv.writer(f)
                for line in response.iter_lines():
                    writer.writerow(line.decode('utf-8').split(','))
            self.data = tc.SFrame.read_csv(csv_name, verbose=False)
            self.data['date']=self.data.apply(lambda x: int(str(x['date']).replace('-','')))
            
        else:
            self.data = tc.SFrame.read_csv(csv_name, verbose=False)
            self.data['date']=self.data.apply(lambda x: int(str(x['date']).replace('-','')))

        self.data['ailing'] = self.data['severe'] # just renaming severe to ailing to match our naming conventions in manuscripts/documentations
        self.data = self.data.select_columns(['state','date','Rt','infections','symptomatic','ailing' ])

    def get_filtered_data(self):
        # returns Sframe within specified [self.start_date, self.end_date] and self.us_state         
        self.data['selected_row'] = self.data.apply(lambda x: 1 if (x['state']==self.us_state and (x['date']>=int(self._start_date) and x['date']<=int(self._end_date)) ) else 0) #1 means row selected, 0 means row not selected for filtered data
    
        return self.data.filter_by([1],'selected_row').sort(['date'],ascending=True)




    #####################

    @property
    def start_date(self):
        return self._start_date

    @property
    def end_date(self):
        return self._end_date

    @property
    def us_state(self):
        return self._us_state

    # if a user decides to update any of the protected state, then PopulatiOndata.filtered_data should correspondingly be updated 
    @start_date.setter
    def start_date(self, new_value):
        self._start_date = new_value
        self.filtered_data = self.get_filtered_data()


    @end_date.setter
    def end_date(self, new_value):
        self._end_date = new_value
        self.filtered_data = self.get_filtered_data()


    @us_state.setter
    def us_state(self, new_value):
        self._us_state = new_value
        self.filtered_data = self.get_filtered_data()
