
"""
PopulationData.py

--------

"""

import os
import turicreate as tc
import autograd.numpy as np
from datetime import datetime,timedelta
from pathlib import Path

from autograd.scipy.special import gamma as gamma_fcn
from autograd_gamma import gammainc
from autograd.scipy.stats import beta,norm,poisson

import autograd
from scipy import stats as sc_stats

import csv
import requests
import urllib

from autograd.scipy.special import expit as sigmoid
from autograd.scipy.special import logit as sigmoid_inv

RESULTS_FOLDER = 'results/'
# DATA_FOLDER = 'data/'

############### HELPER AUTOGRADABLE MATH FUNCTIONS #############
softplus = lambda x: np.log(1+np.exp(x))
# plt.plot(range(-10,10,1),softplus(range(-10,10,1)))

softplus_inv = lambda x: np.log(np.exp(x) -1)
# plt.plot(range(-10,10,1),softplus_inv(softplus(range(-10,10,1))))

# https://en.wikipedia.org/wiki/Gamma_distribution with alpha,beta hyperparameters
def log_gamma_pdf(x,alpha,beta):
    return alpha*np.log(beta) + (alpha-1)*np.log(x) - (beta*x) - np.log(gamma_fcn(alpha))

def gamma_at_x(params,x):
    if x==0:
        return 0.0 
    if x==1:
        return cdf_at_x(params,x)
    elif x>1:
        return cdf_at_x(params,x) - cdf_at_x(params,x-1)

def cdf_at_x(params,x):
    return (gammainc(params['alpha'],params['beta']*x))
############### END helper functions #############    

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

        self.csv_filename = csv_filename
        self._us_state = us_state # protected attrib
        self._start_date = start_date # protected attrib
        self._end_date = end_date # protected attrib
        if params == {}: # set default parameter values 
            # self.params = {'T_serial':5.8, 'Rt_shift':0.2, 'days_of_imposed_restrictions':[],'days_of_relaxed_restrictions':[],\
            #             'prob_sympt':0.65,'prob_severe':0.05,'prob_hosp':0.75, \
            #             'prob_soujourn_inf_fcn':(lambda j: (gamma(a=3.41, scale=1/0.605)).pdf(j) ),'prob_soujourn_symp_fcn':(lambda j: (gamma(a=1.62, scale=1/0.218)).pdf(j))}

            self.params = {'T_serial':5.8, 'Rt_shift':0.2, 'days_of_imposed_restrictions':[],'days_of_relaxed_restrictions':[],\
            'prob_sympt_s':sigmoid_inv(0.65),'prob_severe_s':sigmoid_inv(0.05),'prob_hosp_s':sigmoid_inv(0.75), \
            'prob_soujourn_inf_alpha_s':softplus_inv(3.41), 'prob_soujourn_inf_beta_s':softplus_inv(0.605), \
            'prob_soujourn_symp_alpha_s':softplus_inv(1.62), 'prob_soujourn_symp_beta_s':softplus_inv(0.218)}

        else:
            self.params = params

        self.load_csv_if_exists()
        self.filtered_data = self.get_filtered_data() # captures the time interval from start_date to latest estimates.csv dates

        self.training_mode = training_mode
        self.params_for_training = params_for_training
        self.loss_per_iteration = []

        if forecast:
            if self.debug_mode:
                self.future_data = tc.SFrame(self.csv_filename).filter_by([us_state],'state').sort(['date'],ascending=True)
                # print(self.future_data['date'], 'future data date column from covidEstim')
                # self.future_data['date'] = self.future_data.apply(lambda x: x['date'].replace('-',''))

            self.dates_to_forecast = self.get_dates_to_forecast()
            self.forecasted_data = self.get_forecasted_data() # captures the time interval from latest estimates.csv dates to end_date
        self.forecast = forecast

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

    def get_forecasted_data(self, params=None, save_admissions=False):

        # if params is not None(using default workplan parmas) and self.training_mode:
        if params is not None:
            
            self.params['T_serial']= 5.8

            # self.params['T_serial']= params['T_serial']
            self.params['prob_hosp_s']= params['prob_hosp_s']
            self.params['prob_sympt_s']= params['prob_sympt_s']
            self.params['prob_severe_s']= params['prob_severe_s']

            self.params['prob_soujourn_inf_alpha_s']= params['prob_soujourn_inf_alpha_s']
            self.params['prob_soujourn_inf_beta_s']= params['prob_soujourn_inf_beta_s']
            self.params['prob_soujourn_symp_alpha_s']= params['prob_soujourn_symp_alpha_s']
            self.params['prob_soujourn_symp_beta_s']= params['prob_soujourn_symp_beta_s']

            self.params['days_of_relaxed_restrictions'] = []
            self.params['days_of_imposed_restrictions'] = []


        # EQ 2-1 ##################################################################
        day0 = self.filtered_data['date']
        dates_to_forecast = self.get_dates_to_forecast() # alist of dates which we are forecasting for
        

        RT_before_forecast = self.filtered_data['Rt'][-60:]
        slope, intercept, r_value, p_value, std_err = sc_stats.linregress(range(-60,0,1),RT_before_forecast)

        flow_sus_to_inf_list=[self.filtered_data['infections'][-1]]
        Rt_list=[self.filtered_data['Rt'][-1]]
        T_serial_list = [self.params['T_serial']]

        Rt_sf = (np.array(list(range(len(dates_to_forecast))))*slope + intercept)

        for i,d in enumerate(dates_to_forecast):
            flow_sus_to_inf_list+= [flow_sus_to_inf_list[-1]*(Rt_list[-1]**(1/T_serial_list[-1]))]

            Rt_list+=[Rt_sf[i]]
            T_serial_list += [T_serial_list[-1]]
        
        flow_sus_to_inf_list = flow_sus_to_inf_list[1:]
        flow_sus_to_inf_list_return = flow_sus_to_inf_list
        flow_sus_to_inf_list = list(self.filtered_data['infections'][-20:].to_numpy())+flow_sus_to_inf_list
        
        
        # EQ 2-2 ##################################################################
        flow_inf_to_symp_list = []
        for i,d in enumerate(dates_to_forecast):
            flow_sus_to_inf_list
            flow_inf_to_symp_list+= [np.sum(np.array(\
                [sigmoid(self.params['prob_sympt_s'])*(flow_sus_to_inf_list[:i+20])[-j]*gamma_at_x({'alpha':softplus(self.params['prob_soujourn_inf_alpha_s']),'beta':softplus(self.params['prob_soujourn_inf_beta_s'])},j) for j in range(1,1+len(flow_sus_to_inf_list[:i+20]))]))]                   
        
        flow_inf_to_symp_list_return = flow_inf_to_symp_list
        flow_inf_to_symp_list = list(self.filtered_data['symptomatic'][-40:].to_numpy())+flow_inf_to_symp_list

        
        # EQ 2-3 ##################################################################
        flow_symp_to_severe_list = []
        for i,d in enumerate(dates_to_forecast):
            flow_symp_to_severe_list+= [np.sum(np.array(\
                [sigmoid(self.params['prob_severe_s'])*(flow_inf_to_symp_list[:i+40])[-j]*gamma_at_x({'alpha':softplus(self.params['prob_soujourn_symp_alpha_s']),'beta':softplus(self.params['prob_soujourn_symp_beta_s'])},j) for j in range(1,1+len(flow_inf_to_symp_list[:i+40]))]))]                   

        
        flow_symp_to_severe_list_return = flow_symp_to_severe_list

        # EQ in section 2.2C ##################################################################
        flow_symp_to_severe_list_right_shifted = [self.filtered_data['severe'][-1]]+flow_symp_to_severe_list[:-1]        
        flow_severe_to_hosp_list = [flow_symp_to_severe*sigmoid(self.params['prob_hosp_s']) for flow_symp_to_severe in flow_symp_to_severe_list]
        flow_severe_to_hosp_list_return = flow_severe_to_hosp_list

        if self.training_mode:
            return {'date':dates_to_forecast,'Rt':Rt_list[1:],'infections':flow_sus_to_inf_list_return, 'symptomatic':flow_inf_to_symp_list_return, 'severe':flow_symp_to_severe_list_return, 'hosp':flow_severe_to_hosp_list_return}

        if save_admissions:
            print('SAVE ADMISSIONS')
            tc.SFrame({'date':dates_to_forecast,'n_admitted_InGeneralWard':[int(adm) for adm in flow_severe_to_hosp_list_return]}).save(RESULTS_FOLDER+'forecasted_admissions.csv', format='csv')
        return tc.SFrame({'date':dates_to_forecast,'Rt':Rt_list[1:],'infections':flow_sus_to_inf_list_return, 'symptomatic':flow_inf_to_symp_list_return, 'severe':flow_symp_to_severe_list_return, 'hosp':flow_severe_to_hosp_list_return})
    

############ FUNCTIONS FOR GRADIENT DESCENT TRAINING    ############

    def get_loss_given_truthdict_and_params(self, params, truth_dict, lambda_reg):
        # pop_states = ['infections', 'symptomatic','severe','hosp']
        pop_states = ['hosp'] # we are only penalizing loss based on observable hosp admission and not other non observables 'infections', 'symptomatic','severe'
        pred_dict = self.get_forecasted_data(params)
        log_loss = 0

        for i,k in enumerate(pop_states):
            # log_loss+= np.sum(np.array([poisson.logpmf(t,p) for (p,t) in zip(pred_dict[k],truth_dict[k])]))
            num_preds = len(pred_dict[k])
            # for val in pred_dict[k]:

            #     print(val._value, 'predicted for', k, ' ' )
                # np.max([1,p])
            log_loss+= np.sum(np.array([2*(e/num_preds)*poisson.logpmf(t,p+1e-9) for e,(p,t) in enumerate(zip(pred_dict[k],truth_dict[k]))]))
        

        reg_penalty = (lambda_reg*self.get_log_prior_penalty(params))
        try:
            self.loss_per_iteration[-1] = float(-log_loss._value + reg_penalty._value)
        except:
            self.loss_per_iteration[-1] = float(np.nan)
        # return  -log_loss 
        return  -log_loss + reg_penalty

    def get_log_prior_penalty(self,params):
        log_prior_penalty = 0

        prior_params_ab = {'prob_sympt_s':[5.5,2],'prob_severe_s':[2,5],'prob_hosp_s':[2,4]}

        for k,v in prior_params_ab.items():
            a = v[0]
            b = v[1]
            log_prior_penalty += beta.logpdf(sigmoid(params[k]), a, b)
    

        prior_params_ab2 = {'prob_soujourn_inf_alpha_s':[4.693986464958364, 1.0832804882575846],\
        'prob_soujourn_inf_beta_s':[1.351693639556777, 0.5813118009202929],\
        'prob_soujourn_symp_alpha_s':[2.2019589211697066, 0.741949951339325],\
        'prob_soujourn_symp_beta_s': [1.115102258598192, 0.527992011918313]}
        for k,v in prior_params_ab2.items():
            a = v[0]
            b = v[1]
            log_prior_penalty += log_gamma_pdf(softplus(params[k]),a,b) ## TODO redefine this fn as autogradable

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

