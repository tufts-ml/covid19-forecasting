
"""
HospitalData.py
Examples in Jupyter Notebook to plot hospitalization data within MA from '20200702' to '20210109' using HospitalData.py and joined_file_20210114.csv being within same folder
--------

"""

import os
import turicreate as tc
import numpy as np
from datetime import datetime
from pathlib import Path


class HospitalData(object):
    ''' Represents a Turicreate SFrame of csv data extracted from HHS (+ covidtracking.com) hospitalization data
    https://healthdata.gov/dataset/covid-19-reported-patient-impact-and-hospital-capacity-state-timeseries
    https://api.covidtracking.com/v1/states/daily.csv

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

    def __init__(self, csv_filename, us_state, start_date, end_date):
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
        self.csv_filename = csv_filename
        self._us_state = us_state # protected attrib
        self._start_date = start_date # protected attrib
        self._end_date = end_date # protected attrib


        self.load_csv_if_exists()

    def get_HHS_data(self, source_csv=None):
        ## download csv from hhs website to the current directory of this HospitalData.py file
        if source_csv==None:
            ##TODO do a curl download to get the latest reported_hospital_utilization_timeseries_ YYYYMMDD .csv file
            downloaded_csv = 'reported_hospital_utilization_timeseries_20210110_1007.csv'
            # column name meanings found here: https://healthdata.gov/covid-19-reported-patient-impact-and-hospital-capacity-state-data-dictionary
            return tc.SFrame(downloaded_csv)

        else:
            hhs_data = tc.SFrame(source_csv)
            columns_of_interest_hhs = ['staffed_icu_adult_patients_confirmed_covid', 'inpatient_beds_used_covid', 'previous_day_admission_adult_covid_confirmed']
            columns_of_interest_hhs = columns_of_interest_hhs+ [col + '_coverage' for col in columns_of_interest_hhs] # get the num of hospitals that reports each columns_of_interest
            columns_of_interest_hhs = ['date','state']+columns_of_interest_hhs
            hhs_data_truncated = hhs_data.select_columns(columns_of_interest_hhs)

            hhs_data_truncated['date'] = hhs_data_truncated.apply(lambda x: int(x['date'].replace("-", "")) ) #for example: "MA" + "2020-01-01".replace("-","")
            # print(hhs_data_truncated['date'].unique(), 'unique dates')
            return hhs_data_truncated

    def load_csv_if_exists(self):
        ## if csv exists, then load
        ## else extract data from internet and save as self.csv_filename 

        if os.path.isfile(self.csv_filename):
            #load
            # self.data = tc.SFrame(self.csv_filename)
            self.data = self.get_HHS_data(source_csv=self.csv_filename)
            
        else:
            self.data = self.get_HHS_data(source_csv=None)
        
        self.filtered_data = self.get_filtered_data()
        

    def get_filtered_data(self):
        # returns Sframe within specified [self.start_date, self.end_date] and self.us_state 
        self.data['selected_row'] = self.data.apply(lambda x: 1 if (x['state']==self.us_state and (x['date']>=int(self._start_date) and x['date']<=int(self._end_date)) ) else 0) #1 means row selected, 0 means row not selected for filtered data
        return self.data.filter_by([1],'selected_row').sort(['date'],ascending=True)
    

    def get_InGeneralWard_counts(self):
        # returns Sframe column of general ward counts within [self.start_date, self.end_date] of the self.us_state specified

        # float('nan') is used to replace any blank entries from web data
        return (self.filtered_data.apply(lambda x: float('nan') if (None in [x['inpatient_beds_used_covid'],x['staffed_icu_adult_patients_confirmed_covid']]) \
            else x['inpatient_beds_used_covid']-x['staffed_icu_adult_patients_confirmed_covid'])).to_numpy()

    def get_ICU_counts(self):
        # returns Sframe column of staffed_icu_adult_patients_confirmed_covid counts within [self.start_date, self.end_date] of the self.us_state specified
        return (self.filtered_data.apply(lambda x: float('nan') if (None in [x['staffed_icu_adult_patients_confirmed_covid']]) \
            else x['staffed_icu_adult_patients_confirmed_covid'])).to_numpy()

    ######fcns useful for ParamsGenerator##########
    def get_init_num(self, hospital_state): #hospital state means Presenting,InGeneralWard,OffVentInICU,OnVentInICU,etc
        # "init_num_Presenting": 0,
        # "init_num_InGeneralWard": 10,
        # "init_num_OffVentInICU": 10,
        # "init_num_OnVentInICU": 10,
        if hospital_state in ['Presenting','OffVentInICU','OnVentInICU']:
            return int(0) 
        else:
            state_counts = eval('self.get_' +str(hospital_state)+ '_counts()')
            # print(state_counts, 'for ', hospital_state)
        return int(state_counts[0]) # must be int type to make json happy

    def get_num_timesteps_in_days(self):
        # "num_timesteps": 21,
        dateFormat = "%Y%m%d"
        d1 = datetime.strptime(self._start_date, dateFormat)
        d2 = datetime.strptime(self._end_date, dateFormat)
        return abs((d2 - d1).days)

    def get_pmf_num_per_timestep_InGeneralWard(self,errorFactor=1):
        # returns a filename.csv of pmf_num_per_timestep_Presenting
         
        # admission seems to be much more than inpatient beds, so maybe theres a factor of 7-10 counted in admissions, but not counted for in ICU and inpatient bed
        
        previous_day_admission_adult_covid_confirmed = self.filtered_data.apply(lambda x: float('nan') if x['previous_day_admission_adult_covid_confirmed']==None else x['previous_day_admission_adult_covid_confirmed']/errorFactor ).to_numpy()
        # presenting_per_day = np.append(previous_day_admission_adult_covid_confirmed[1:],np.array(np.NaN)) # to keep length consistent
        presenting_per_day = np.append(previous_day_admission_adult_covid_confirmed[1:],np.array(0)) # to keep length consistent
        

        pmf_SFrame = tc.SFrame({'timestep':list(range(len(presenting_per_day))),'num_InGeneralWard':presenting_per_day})
        # need to add turicreate sFrame to save CSV in proper timestep, num_[state] column-named formats
        # np.savetxt("pmf_num_per_timestep_InGeneralWard.csv", presenting_per_day, delimiter=",")
        
        csv_filename = str(Path(__file__).resolve().parents[0])+ '/'+self.us_state +'_'+'pmf_num_per_timestep_InGeneralWard.csv' 
        pmf_SFrame.save(csv_filename, format='csv')
        return csv_filename


    # called within ParamsGenerator class
    def extract_param_value(self,param_name):
        # function that maps param_name to param_value based on the supplemental_obj being HospitalData or CovidEstim Obj
        states = ["InGeneralWard", "OffVentInICU", "OnVentInICU"]
        
        if 'num_timesteps' == param_name:
            return self.get_num_timesteps_in_days()

        if 'pmf_num_per_timestep_InGeneralWard' == param_name:
            return self.get_pmf_num_per_timestep_InGeneralWard(errorFactor=1)

        for hospital_state in states: 
            if 'init_num_'+hospital_state == param_name:
                return self.get_init_num(hospital_state)
        
        # if does not hit any of above conditional cases
        raise Exception(param_name+' failed compatibility with extract_param_value of :'+type(self))
    ######END fcns useful for ParamsGenerator##########

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
        self.filtered_data = self.get_filtered_data()

    @end_date.setter
    def end_date(self, new_value):
        self._end_date = new_value
        self.filtered_data = self.get_filtered_data()

    @us_state.setter
    def us_state(self, new_value):
        self._us_state = new_value
        self.filtered_data = self.get_filtered_data()

