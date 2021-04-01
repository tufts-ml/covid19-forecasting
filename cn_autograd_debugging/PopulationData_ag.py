
"""
PopulationData.py

--------

"""

import os
import turicreate as tc
import autograd.numpy as np
from datetime import datetime,timedelta
from pathlib import Path

# from scipy.stats import gamma
# import autograd.scipy.stats.gamma as gamma
# import autograd.scipy.stats as stat

from autograd.scipy.special import gamma as gamma_fcn
# from autograd.scipy.special import gammainc
from autograd_gamma import gammainc
from autograd.scipy.stats import beta,norm

# from autograd.scipy.stats import gamma as gamma_dist
import autograd

import csv
import requests
from bs4 import BeautifulSoup
import urllib



factorial=lambda x: np.prod([i for i in list(range(1,x+1))])

def log_gamma_pdf(x,alpha,beta):
    return np.log((beta**alpha)*(x**(alpha-1))*np.exp(-beta*x)/gamma_fcn(alpha))
    # print(gamma_dist.logpdf(x, alpha, loc=0, scale = 1 / beta), 'DEBUGGING')
    # return gamma_dist.logpdf(x, alpha, loc=0, scale = 1 / beta)
    # return np.log(gamma_at_x({'alpha':alpha,'beta',beta},x))
def gamma_at_x(params,x):
#     return np.exp(log_gamma_pdf(x=x,alpha=params['alpha'], beta = params['beta']))
    if x==0:
        return 0.0 
    if x==1:
        return cdf_at_x(params,x)
    elif x>1:
        return cdf_at_x(params,x) - cdf_at_x(params,x-1)

def cdf_at_x(params,x):
    return (gammainc(params['alpha'],params['beta']*x))    
    

def pmf_at_x(params,x):
    k = x-params['loc']
    return (params['mu']**(k))*np.exp(-params['mu'])/ factorial(k)   

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

    def __init__(self, csv_filename, us_state, start_date, end_date, forecast=True, params={}, debug_mode = False, training_mode=False, params_for_training = {'t':None, 'forecast_type':None} ):
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
        self.debug_mode=debug_mode
        # print('lambda j: (gamma(a=3.41, scale=1/0.605)).pdf(j) ', lambda j: (gamma(a=3.41, scale=1/0.605)).pdf(j), (lambda j: (gamma(a=1.62, scale=1/0.218)).pdf(j)) )
        self.csv_filename = csv_filename
        self._us_state = us_state # protected attrib
        self._start_date = start_date # protected attrib
        self._end_date = end_date # protected attrib
        if params == {}:
            # self.params = {'T_serial':5.8, 'Rt_shift':0.2, 'days_of_imposed_restrictions':[],'days_of_relaxed_restrictions':[],\
            #             'prob_sympt':0.65,'prob_severe':0.05,'prob_hosp':0.75, \
            #             'prob_soujourn_inf_fcn':(lambda j: (gamma(a=3.41, scale=1/0.605)).pdf(j) ),'prob_soujourn_symp_fcn':(lambda j: (gamma(a=1.62, scale=1/0.218)).pdf(j))}

            self.params = {'T_serial':5.8, 'Rt_shift':0.2, 'days_of_imposed_restrictions':[],'days_of_relaxed_restrictions':[],\
            'prob_sympt':0.65,'prob_severe':0.05,'prob_hosp':0.75, \
            'prob_soujourn_inf_alpha':3.41, 'prob_soujourn_inf_beta':0.605, \
            'prob_soujourn_symp_alpha':1.62, 'prob_soujourn_symp_beta':0.218}

        else:
            self.params = params

        self.load_csv_if_exists()
        self.filtered_data = self.get_filtered_data() # captures the time interval from start_date to latest estimates.csv dates

        self.training_mode = training_mode
        self.params_for_training = params_for_training
        self.loss_per_iteration = []

        if forecast:
            if self.debug_mode:
                self.future_data = tc.SFrame('covidestim.csv').filter_by([us_state],'state').sort(['date'],ascending=True)
                self.future_data['date'] = self.future_data.apply(lambda x: x['date'].replace('-',''))
            self.dates_to_forecast = self.get_dates_to_forecast()
            self.forecasted_data = self.get_forecasted_data() # captures the time interval from latest estimates.csv dates to end_date
        self.forecast = forecast

    def load_csv_if_exists(self):
        csv_name = 'covidestim.csv'
        today = datetime.now().date()        
        if not Path(csv_name).exists() or datetime.fromtimestamp(os.path.getctime(csv_name)).date() != today  :
            url = 'https://covidestim.s3.us-east-2.amazonaws.com/latest/state/estimates.csv'
            response = requests.get(url)  
            with open(csv_name, 'w') as f:
                writer = csv.writer(f)
                for line in response.iter_lines():
                    writer.writerow(line.decode('utf-8').split(','))
            self.data = tc.SFrame(csv_name)
            self.data['date']=self.data.apply(lambda x: int(str(x['date']).replace('-','')))
        elif self.debug_mode:
            self.data = tc.SFrame(self.csv_filename)
            self.data['date']=self.data.apply(lambda x: int(str(x['date']).replace('-','')))
        else:
            self.data = tc.SFrame(csv_name)
            self.data['date']=self.data.apply(lambda x: int(str(x['date']).replace('-','')))

    def get_filtered_data(self):
        # returns Sframe within specified [self.start_date, self.end_date] and self.us_state         
        self.data['selected_row'] = self.data.apply(lambda x: 1 if (x['state']==self.us_state and (x['date']>=int(self._start_date) and x['date']<=int(self._end_date)) ) else 0) #1 means row selected, 0 means row not selected for filtered data
        return self.data.filter_by([1],'selected_row').sort(['date'],ascending=True)

    def get_forecasted_data(self, params=None):

        # if params is not None(using default workplan parmas) and self.training_mode:
        if params is not None:

            self.params['T_serial']= 5.8

            # self.params['T_serial']= params['T_serial']
            self.params['prob_hosp']= params['prob_hosp']
            self.params['prob_sympt']= params['prob_sympt']
            self.params['prob_severe']= params['prob_severe']

            self.params['prob_soujourn_inf_alpha']= params['prob_soujourn_inf_alpha']
            self.params['prob_soujourn_inf_beta']= params['prob_soujourn_inf_beta']
            self.params['prob_soujourn_symp_alpha']= params['prob_soujourn_symp_alpha']
            self.params['prob_soujourn_symp_beta']= params['prob_soujourn_symp_beta']

            self.params['days_of_relaxed_restrictions'] = []
            self.params['days_of_imposed_restrictions'] = []


        # EQ 2-1 ##################################################################
        day0 = self.filtered_data['date']
        dates_to_forecast = self.get_dates_to_forecast() # alist of dates which we are forecasting for
        
        flow_sus_to_inf_list=[self.filtered_data['infections'][-1]]
        Rt_list=[self.filtered_data['Rt'][-1]]
        T_serial_list = [self.params['T_serial']]

        #modify Rt based on government restrictionss
        Rt_0 = self.filtered_data['Rt'][-1]
        Rt_sf = self.future_data.filter_by([d.replace('-','') for d in dates_to_forecast], 'date').sort(['date'])['Rt'].to_numpy()

        
        for i,d in enumerate(dates_to_forecast):
            flow_sus_to_inf_list+= [flow_sus_to_inf_list[-1]*(Rt_list[-1]**(1/T_serial_list[-1]))]
            Rt_list+=[Rt_sf[i]]
            T_serial_list += [T_serial_list[-1]]
        
        flow_sus_to_inf_list = flow_sus_to_inf_list[1:]
        flow_sus_to_inf_list_return = flow_sus_to_inf_list
        flow_sus_to_inf_list = list(self.filtered_data['infections'][-20:].to_numpy())+flow_sus_to_inf_list
        
        # if self.training_mode and self.params_for_training['forecast_type'] == 'infections':
        #     return flow_sus_to_inf_list_return[self.params_for_training['t']]
        
        # EQ 2-2 ##################################################################
        flow_inf_to_symp_list = []
        for i,d in enumerate(self.dates_to_forecast):
            flow_sus_to_inf_list
            flow_inf_to_symp_list+= [np.sum(np.array(\
                [self.params['prob_sympt']*(flow_sus_to_inf_list[:i+20])[-j]*gamma_at_x({'alpha':self.params['prob_soujourn_inf_alpha'],'beta':self.params['prob_soujourn_inf_beta']},j) for j in range(1,1+len(flow_sus_to_inf_list[:i+20]))]))]                   
                # [self.params['prob_sympt']*(flow_sus_to_inf_list[:i+20])[-j]*self.params['prob_soujourn_inf_fcn'](j) for j in range(1,1+len(flow_sus_to_inf_list[:i+20]))]))]                   
        
        flow_inf_to_symp_list_return = flow_inf_to_symp_list
        flow_inf_to_symp_list = list(self.filtered_data['symptomatic'][-40:].to_numpy())+flow_inf_to_symp_list
        # if self.training_mode and self.params_for_training['forecast_type'] == 'symptomatic': 
        #     return flow_inf_to_symp_list_return[self.params_for_training['t']]

        # EQ 2-3 ##################################################################
        flow_symp_to_severe_list = []
        for i,d in enumerate(self.dates_to_forecast):
            flow_symp_to_severe_list+= [np.sum(np.array(\
                [self.params['prob_severe']*(flow_inf_to_symp_list[:i+40])[-j]*gamma_at_x({'alpha':self.params['prob_soujourn_symp_alpha'],'beta':self.params['prob_soujourn_symp_beta']},j) for j in range(1,1+len(flow_inf_to_symp_list[:i+40]))]))]                   
                # [self.params['prob_severe']*(flow_inf_to_symp_list[:i+40])[-j]*self.params['prob_soujourn_symp_fcn'](j) for j in range(1,1+len(flow_inf_to_symp_list[:i+40]))]))]                   
        
        flow_symp_to_severe_list_return = flow_symp_to_severe_list
        # if self.training_mode and self.params_for_training['forecast_type'] == 'severe':
        #     return flow_symp_to_severe_list_return[self.params_for_training['t']]

        # EQ in section 2.2C ##################################################################
        flow_symp_to_severe_list_right_shifted = [self.filtered_data['severe'][-1]]+flow_symp_to_severe_list[:-1]        
        flow_severe_to_hosp_list = [flow_symp_to_severe*self.params['prob_hosp'] for flow_symp_to_severe in flow_symp_to_severe_list_right_shifted]
        flow_severe_to_hosp_list_return = flow_severe_to_hosp_list
        # if self.training_mode and self.params_for_training['forecast_type'] == 'hosp':
        #     return flow_severe_to_hosp_list_return[self.params_for_training['t']]

        if self.training_mode:
            return {'date':self.dates_to_forecast,'infections':flow_sus_to_inf_list_return, 'symptomatic':flow_inf_to_symp_list_return, 'severe':flow_symp_to_severe_list_return, 'hosp':flow_severe_to_hosp_list_return}
        return tc.SFrame({'date':self.dates_to_forecast,'infections':flow_sus_to_inf_list_return, 'symptomatic':flow_inf_to_symp_list_return, 'severe':flow_symp_to_severe_list_return, 'hosp':flow_severe_to_hosp_list_return})
    


    def get_loss_given_truthdict_and_params(self, params, truth_dict, lambda_reg):
        # pop_states = ['infections', 'symptomatic','severe','hosp']
        pop_states = ['hosp'] # we are only penalizing loss based on observable hosp admission and not other non observables 'infections', 'symptomatic','severe'
        pred_dict = self.get_forecasted_data(params)
        loss = 0

        # print(pred_dict['hosp'], 'PRED from get loss fcn')
        # print(np.array(pred_dict['hosp']).shape, np.array(truth_dict['hosp']).shape)
        for i,k in enumerate(pop_states):
            loss+=np.sum(np.abs(np.array(pred_dict[k]) - np.array(truth_dict[k]))*np.linspace(0.1, 1, num=len(pred_dict[k])) )
        
        try:
            self.loss_per_iteration[-1] = float(loss._value)
        except:
            self.loss_per_iteration[-1] = float(np.nan)

        return  loss + (lambda_reg*self.get_log_prior_penalty(params))

    def get_log_prior_penalty(self,params):
        log_prior_penalty = 0
        prior_params_ab = {'prob_sympt':[9,2],'prob_severe':[2,8],'prob_hosp':[2,6]}

        for k,v in prior_params_ab.items():
            a = v[0]
            b = v[1]
            log_prior_penalty += beta.logpdf(params[k], a, b)
    
        prior_params_ab2 = {'prob_soujourn_inf_alpha':[4.693986464958364, 1.0832804882575846],\
        'prob_soujourn_inf_beta':[1.351693639556777, 0.5813118009202929],\
        'prob_soujourn_symp_alpha':[2.2019589211697066, 0.741949951339325],\
        'prob_soujourn_symp_beta': [1.115102258598192, 0.527992011918313]}
        for k,v in prior_params_ab2.items():
            a = v[0]
            b = v[1]
            log_prior_penalty += log_gamma_pdf(params[k],a,b)

        return -log_prior_penalty

    def get_grad_of_loss(self, params, truth_dict, lambda_reg):
        grad_of_loss = autograd.grad(self.get_loss_given_truthdict_and_params)
        return grad_of_loss(params,truth_dict,lambda_reg)

    def get_dates_to_forecast(self):

        most_recent_estimated_date = str(self.filtered_data['date'][-1])
        dates_to_forecast = []+[most_recent_estimated_date]
        for t in range((datetime.strptime(self.end_date, "%Y%m%d") - datetime.strptime(most_recent_estimated_date, "%Y%m%d")).days ):
            dates_to_forecast+=[(datetime.strptime(dates_to_forecast[-1], "%Y%m%d")+timedelta(days=1)).strftime("%Y%m%d")]

        return dates_to_forecast  
    



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
        if self.forecast:
            self.forecasted_data = self.get_forecasted_data()

    @end_date.setter
    def end_date(self, new_value):
        self._end_date = new_value
        self.filtered_data = self.get_filtered_data()
        if self.forecast:
            self.forecasted_data = self.get_forecasted_data()

    @us_state.setter
    def us_state(self, new_value):
        self._us_state = new_value
        self.filtered_data = self.get_filtered_data()
        if self.forecast:
            self.forecasted_data = self.get_forecasted_data()

