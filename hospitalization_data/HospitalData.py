
"""
HospitalData.py
Examples in Jupyter Notebook to plot hospitalization data within MA from '20200702' to '20210109' using HospitalData.py and joined_file_20210114.csv being within same folder
--------
from HospitalData import HospitalData
hObj = HospitalData('joined_file_20210114.csv', 'MA','20200702','20210109')

import matplotlib.pyplot as plt
import numpy as np
import os
%matplotlib inline

plt.figure(figsize=(15,4))
plt.plot(hObj.filtered_data['staffed_icu_adult_patients_confirmed_covid'],color='b',label='ICU')
# plt.plot(np.array(OffVentICU)+np.array(OnVentICU),color='k',label='ICU (Off+OnVent)')
plt.plot(hObj.get_OffVentICU_counts(),color='b',linestyle=':',label='OffVentICU')
plt.plot(hObj.get_OnVentICU_counts(),color='b',linestyle='--',label='OnVentICU')
plt.plot(hObj.get_GeneralWard_counts(),color='r',label='General Ward?')
plt.plot(hObj.filtered_data['inpatient_beds_used_covid'],color='m',label='General Ward+OffVentICU+OnVentICU')

# admission seems to be much more than inpatient beds, so maybe theres a factor of 7-10 counted in admissions, but not counted for in ICU and inpatient bed
pmf_num_per_timestep_Presenting_filename = hObj.get_pmf_num_per_timestep_Presenting_filename(errorFactor=7)
import turicreate
num_per_timestep_Presenting = turicreate.SFrame(pmf_num_per_timestep_Presenting_filename)

pres = num_per_timestep_Presenting.to_numpy()
plt.plot(pres,color='k',linestyle=':',label='presented admissions')
plt.plot(np.nancumsum(pres),color='k',label='cumulative admissions')

dates = list(hObj.filtered_data['date'])
def sparsify_dates_labels(dates_list):
    for dd, date in enumerate(dates_list):
        if dd%4 == 0:
            dates_list[dd] = str(dates_list[dd])[-4:]
            pass
        else:
            dates_list[dd] = ''
    return dates_list
    
plt.xticks(ticks=list(range(len(dates))),labels=sparsify_dates_labels(dates),rotation='vertical');
plt.xlabel('date in format MMDD')
plt.legend()
plt.title(hObj.us_state+' hospitalization data in covid_forecasting variable names');
plt.show()


"""

import os
import turicreate as tc
import numpy as np
from datetime import datetime


class HospitalData(object):
    ''' Represents a Turicreate SFrame of csv data extracted from CDC (+ covidtracking.com) hospitalization data
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

    def get_CDC_data(self):
        ## download csv from cdc website to the current directory of this HospitalData.py file

        ##TODO do a curl download to get the latest reported_hospital_utilization_timeseries_ YYYYMMDD .csv file
        downloaded_csv = 'reported_hospital_utilization_timeseries_20210110_1007.csv'
        return tc.SFrame(downloaded_csv)

    def get_covidtracking_data(self):
        ## download csv from covidtracking website to the current directory of this HospitalData.py file

        ##TODO do a curl download to get the latest daily_historic.csv file
        downloaded_csv = 'daily_historic.csv'
        return tc.SFrame(downloaded_csv)

    def join_CDC_covidtracking_data(self, cdc_data, ctrack_data):
        ## joining the relevant column names of the 2 data sources
        
        # column name meanings found here: https://healthdata.gov/covid-19-reported-patient-impact-and-hospital-capacity-state-data-dictionary
        columns_of_interest_cdc = ['staffed_icu_adult_patients_confirmed_covid', 'inpatient_beds_used_covid', 'previous_day_admission_adult_covid_confirmed']
        columns_of_interest_cdc = columns_of_interest_cdc+ [col + '_coverage' for col in columns_of_interest_cdc] # get the num of hospitals that reports each columns_of_interest
        columns_of_interest_cdc = ['date','state']+columns_of_interest_cdc
        cdc_data_truncated = cdc_data.select_columns(columns_of_interest_cdc)

        columns_of_interest_ctrack = ['date','state','inIcuCurrently', 'onVentilatorCurrently']
        ctrack_data_truncated = ctrack_data.select_columns(columns_of_interest_ctrack)

        # for both of the truncated Sframe, create a state_and_date column such that we can do a joining of rows (wherever they share the same state and same date)
        cdc_data_truncated['state_and_date'] = cdc_data_truncated.apply(lambda x: x['state'] + x['date'].replace("-", "")) #for example: "MA" + "2020-01-01".replace("-","")
        ctrack_data_truncated['state_and_date'] = ctrack_data_truncated.apply(lambda x: x['state'] + str(x['date'])) #for example: "MA" + str(20200101)

        # delete the 'date' and 'state' column of cdc_data_truncated because we will be getting those 2 columns from  ctrack_data_truncated after join
        cdc_data_truncated = cdc_data_truncated.remove_columns(column_names=['state','date'])

        #debugging
        print(cdc_data_truncated.column_names(), 'cdc columns available before merge')
        print(ctrack_data_truncated.column_names(), 'covidtracking columns available before merge')

        return cdc_data_truncated.join(ctrack_data_truncated, on=['state_and_date'], how='left')

    def load_csv_if_exists(self):
        ## if csv exists, then load
        ## else extract data from internet and save as self.csv_filename 

        if os.path.isfile(self.csv_filename):
            #load
            self.data = tc.SFrame(self.csv_filename)
            
        else:
            
            #dowload 2 datasets and join
            self.data = self.join_CDC_covidtracking_data(self.get_CDC_data(),self.get_covidtracking_data())
            #compute the prob_OnVent_givenICU to impute for the fraction OnVent vs OffVent from the limited staffed_icu_adult_patients_confirmed_covid of CDC data
            self.data['prob_OnVent_givenICU'] = self.data.apply(lambda x: float('nan') if (None in [x['onVentilatorCurrently'],x['inIcuCurrently']] or x['inIcuCurrently']==0) \
                else x['onVentilatorCurrently']/x['inIcuCurrently'])
            self.data.save(self.csv_filename, format='csv') 
        
        self.filtered_data = self.get_filtered_data()
        # print(self.filtered_data.select_columns(['date','staffed_icu_adult_patients_confirmed_covid']))

    def get_filtered_data(self):
        # returns Sframe within specified [self.start_date, self.end_date] and self.us_state 
        self.data['selected_row'] = self.data.apply(lambda x: 1 if (x['state']==self.us_state and (x['date']>=int(self._start_date) and x['date']<=int(self._end_date)) ) else 0) #1 means row selected, 0 means row not selected for filtered data
        return self.data.filter_by([1],'selected_row').sort(['date'],ascending=True)
    

    def get_GeneralWard_counts(self):
        # returns Sframe column of general ward counts within [self.start_date, self.end_date] of the self.us_state specified

        # float('nan') is used to replace any blank entries from web data
        return (self.filtered_data.apply(lambda x: float('nan') if (None in [x['inpatient_beds_used_covid'],x['staffed_icu_adult_patients_confirmed_covid']]) \
            else x['inpatient_beds_used_covid']-x['staffed_icu_adult_patients_confirmed_covid'])).to_numpy()

    def get_OnVentICU_counts(self):
        # returns Sframe column of OnVentICU counts within [self.start_date, self.end_date] of the self.us_state specified
        # calculated/imputed by x['staffed_icu_adult_patients_confirmed_covid']*x['prob_OnVent_givenICU']
        return (self.filtered_data.apply(lambda x: float('nan') if (None in [x['onVentilatorCurrently'],x['inIcuCurrently'],x['staffed_icu_adult_patients_confirmed_covid']]) \
            else x['staffed_icu_adult_patients_confirmed_covid']*x['prob_OnVent_givenICU'])).to_numpy()

    def get_OffVentICU_counts(self):
        # returns Sframe column of OffVentICU counts within [self.start_date, self.end_date] of the self.us_state specified
        # calculated/imputed by x['staffed_icu_adult_patients_confirmed_covid']*(1-x['prob_OnVent_givenICU']
        return (self.filtered_data.apply(lambda x: float('nan') if (None in [x['onVentilatorCurrently'],x['inIcuCurrently'],x['staffed_icu_adult_patients_confirmed_covid']]) \
            else x['staffed_icu_adult_patients_confirmed_covid']*(1-x['prob_OnVent_givenICU']))).to_numpy()


    ######fcns useful for ParamsGenerator##########
    def get_init_num(self, hospital_state): #hospital state means Presenting,InGeneralWard,OffVentInICU,OnVentInICU,etc
        # "init_num_Presenting": 0,
        # "init_num_InGeneralWard": 10,
        # "init_num_OffVentInICU": 10,
        # "init_num_OnVentInICU": 10,
        if hospital_state == 'Presenting':
            return 0 
        else:
            state_counts = eval('self.get_' +str(hospital_state)+ 'counts()')
        return state_counts[0]

    def get_num_timesteps_in_days(self):
        # "num_timesteps": 21,
        dateFormat = "%Y%m%d"
        d1 = datetime.strptime(self._start_date, dateFormat)
        d2 = datetime.strptime(self._end_date, dateFormat)
        return abs((d2 - d1).days)

    def get_pmf_num_per_timestep_Presenting_filename(self,errorFactor=1):
        # returns a filename.csv of pmf_num_per_timestep_Presenting
         
        # admission seems to be much more than inpatient beds, so maybe theres a factor of 7-10 counted in admissions, but not counted for in ICU and inpatient bed
        
        previous_day_admission_adult_covid_confirmed = self.filtered_data.apply(lambda x: float('nan') if x['previous_day_admission_adult_covid_confirmed']==None else x['previous_day_admission_adult_covid_confirmed']/errorFactor ).to_numpy()
        presenting_per_day = np.append(previous_day_admission_adult_covid_confirmed[1:],np.array(np.NaN)) # to keep length consistent
        np.savetxt("pmf_num_per_timestep_Presenting.csv", presenting_per_day, delimiter=",")
        return 'pmf_num_per_timestep_Presenting.csv'


    # called within ParamsGenerator class
    def extract_param_value(self,param_name):
        # function that maps param_name to param_value based on the supplemental_obj being HospitalData or CovidEstim Obj
        states = ["Presenting", "InGeneralWard", "OffVentInICU", "OnVentInICU"]
        
        if 'num_timesteps' == param_name:
            return self.get_num_timesteps_in_days()

        if 'pmf_num_per_timestep_Presenting' == param_name:
                return self.get_pmf_num_per_timestep_Presenting_filename(errorFactor=7)

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

