
"""
PopulationModel.py

--------

"""

import os
import turicreate as tc
import autograd.numpy as np
from datetime import datetime,timedelta


from autograd.scipy.special import gamma as gamma_fcn
from autograd_gamma import gammainc
from autograd.scipy.stats import beta,norm,poisson

import autograd
from scipy import stats as sc_stats



from autograd.scipy.special import expit as sigmoid
from autograd.scipy.special import logit as sigmoid_inv

from PopulationData import PopulationData

import matplotlib.pyplot as plt
import copy

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




class PopulationModel(object):
    ''' Represents a Population model that can predict number of hospital-admissibles (and # of people who are infected, symptomatic, ailing)

    Attributes
    ----------
    warmup_data: PopulationData object
        contains the covidestim.org SFrame dataset to help initialize the I,S,A,H values for the model at t=0 (i.e warmup_data.end_date
    forecast_duration: int
        during forecasting (i.e not training) the model will predict forecast_duration days beyond t=0, where t=0 is the  warmup_data.end_date

        '''
    def __init__(self, warmup_data_obj, forecast_duration = 60, params={}, priors={}): 
        self.warmup_data = copy.deepcopy(warmup_data_obj) # Population Data obj
        self.forecast_duration = forecast_duration
        self.set_params(params)
        self.set_priors(priors)

    def get_forecasted_data(self, params=None, save_admissions=False):
        if params is None:
            params = self.params


        # EQ 2-1 ##################################################################
        day0 = self.warmup_data.filtered_data['date']
        dates_to_forecast = self.get_dates_to_forecast() # alist of dates which we are forecasting for

        flow_sus_to_inf_list=[self.warmup_data.filtered_data['infections'][-1]]
        Rt_list=[self.warmup_data.filtered_data['Rt'][-1]]
        T_serial_list = [params['T_serial']]

        if self.training_mode:
            Rt_sf = self.warmup_data.data.filter_by(dates_to_forecast, 'date').sort(['date'])['Rt'].to_numpy().tolist()   

        else: # forecasting mode
            lookback_duration = 60
            RT_before_forecast = self.warmup_data.filtered_data['Rt'][-lookback_duration:]
            slope, intercept, r_value, p_value, std_err = sc_stats.linregress(range(-lookback_duration,0,1),RT_before_forecast)
            Rt_sf = (np.array(list(range(len(dates_to_forecast))))*slope + intercept)


        for i,d in enumerate(dates_to_forecast):
            flow_sus_to_inf_list+= [flow_sus_to_inf_list[-1]*(Rt_list[-1]**(1/T_serial_list[-1]))]

            Rt_list+=[Rt_sf[i]]
            T_serial_list += [T_serial_list[-1]]
        
        flow_sus_to_inf_list = flow_sus_to_inf_list[1:]
        flow_sus_to_inf_list_return = flow_sus_to_inf_list
        flow_sus_to_inf_list = list(self.warmup_data.filtered_data['infections'][-20:].to_numpy())+flow_sus_to_inf_list
        
        
        # EQ 2-2 ##################################################################
        flow_inf_to_symp_list = []
        for i,d in enumerate(dates_to_forecast):
            flow_sus_to_inf_list
            flow_inf_to_symp_list+= [np.sum(np.array(\
                [sigmoid(params['prob_sympt_s'])*(flow_sus_to_inf_list[:i+20])[-j]*gamma_at_x({'alpha':softplus(params['prob_soujourn_inf_alpha_s']),'beta':softplus(params['prob_soujourn_inf_beta_s'])},j) for j in range(1,1+len(flow_sus_to_inf_list[:i+20]))]))]                   
        
        flow_inf_to_symp_list_return = flow_inf_to_symp_list
        flow_inf_to_symp_list = list(self.warmup_data.filtered_data['symptomatic'][-40:].to_numpy())+flow_inf_to_symp_list

        
        # EQ 2-3 ##################################################################
        flow_symp_to_ailing_list = []
        for i,d in enumerate(dates_to_forecast):
            flow_symp_to_ailing_list+= [np.sum(np.array(\
                [sigmoid(params['prob_ailing_s'])*(flow_inf_to_symp_list[:i+40])[-j]*gamma_at_x({'alpha':softplus(params['prob_soujourn_symp_alpha_s']),'beta':softplus(params['prob_soujourn_symp_beta_s'])},j) for j in range(1,1+len(flow_inf_to_symp_list[:i+40]))]))]                   

        
        flow_symp_to_ailing_list_return = flow_symp_to_ailing_list

        # EQ in section 2.2C ##################################################################
        # flow_symp_to_ailing_list_right_shifted = [self.warmup_data.filtered_data['ailing'][-1]]+flow_symp_to_ailing_list[:-1]        
        flow_ailing_to_hosp_list = [flow_symp_to_ailing*sigmoid(params['prob_hosp_s']) for flow_symp_to_ailing in flow_symp_to_ailing_list]
        flow_ailing_to_hosp_list_return = flow_ailing_to_hosp_list

        if self.training_mode:
            return {'date':dates_to_forecast,'Rt':Rt_list[1:],'infections':flow_sus_to_inf_list_return, 'symptomatic':flow_inf_to_symp_list_return, 'ailing':flow_symp_to_ailing_list_return, 'hospitalized':flow_ailing_to_hosp_list_return}

        if save_admissions:
            print('\n','***forecasted hospital-admissible numbers saved to ', RESULTS_FOLDER+'forecasted_admissions.csv')
            tc.SFrame({'date':dates_to_forecast,'n_admitted_InGeneralWard':[int(adm) for adm in flow_ailing_to_hosp_list_return]}).save(RESULTS_FOLDER+'forecasted_admissions.csv', format='csv')
        return tc.SFrame({'date':dates_to_forecast,'Rt':Rt_list[1:],'infections':flow_sus_to_inf_list_return, 'symptomatic':flow_inf_to_symp_list_return, 'ailing':flow_symp_to_ailing_list_return, 'hospitalized':flow_ailing_to_hosp_list_return})
    
    def set_params(self,params):
        if params == {}: # set default parameter (prior-pdf-mean-centered) values 
        #(Note that we are using the transformed (via softplus and sigmoid) values 
        #of default values because we want to constrain these learnable parameters 
        #certain acceptable ranges. i.e transition probabilities are only within the range of 0 to 1 (enforced by sigmoid))
            # self.params = {'T_serial':5.8, 'Rt_shift':0.2, 'days_of_imposed_restrictions':[],'days_of_relaxed_restrictions':[],\
            # 'prob_sympt_s':sigmoid_inv(0.65),'prob_ailing_s':sigmoid_inv(0.05),'prob_hosp_s':sigmoid_inv(0.75), \
            # 'prob_soujourn_inf_alpha_s':softplus_inv(3.41), 'prob_soujourn_inf_beta_s':softplus_inv(0.605), \
            # 'prob_soujourn_symp_alpha_s':softplus_inv(1.62), 'prob_soujourn_symp_beta_s':softplus_inv(0.218)}


            self.params = {'T_serial':5.8, 'Rt_shift':0.2, 'days_of_imposed_restrictions':[],'days_of_relaxed_restrictions':[],\
            'prob_sympt_s':sigmoid_inv(0.65),'prob_ailing_s':sigmoid_inv(0.05),'prob_hosp_s':sigmoid_inv(0.45), \
            'prob_soujourn_inf_alpha_s':softplus_inv(3.41), 'prob_soujourn_inf_beta_s':softplus_inv(0.605), \
            'prob_soujourn_symp_alpha_s':softplus_inv(1.62), 'prob_soujourn_symp_beta_s':softplus_inv(0.218)}

        else:
            for key, value in params.items():
                self.params[key] = value

    def set_priors(self,priors):
        
        if priors == {}: # set default hyperparameters that define the parameters' prior pdfs
        #(Note 'prob_sympt_s':[5.5,2],'prob_ailing_s':[2,5],'prob_hosp_s':[2,4] each has the [a,b] hyperparameters of beta.logpdf 
        # 'prob_soujourn_inf_alpha_s':[4.693986464958364, 1.0832804882575846] and other sojourns has the [a,b] hyperparameters of log_gamma_pdf

            self.priors = {'T_serial':None, 'Rt_shift':None, 'days_of_imposed_restrictions':None,'days_of_relaxed_restrictions':None,\
            'prob_sympt_s':[5.5,2],'prob_ailing_s':[2,5],'prob_hosp_s':[2,4], \
            'prob_soujourn_inf_alpha_s':[4.693986464958364, 1.0832804882575846],\
            'prob_soujourn_inf_beta_s':[1.351693639556777, 0.5813118009202929],\
            'prob_soujourn_symp_alpha_s':[2.2019589211697066, 0.741949951339325],\
            'prob_soujourn_symp_beta_s': [1.115102258598192, 0.527992011918313]}

        else:
            for key, value in priors.items():
                self.priors[key] = value
        

############ FUNCTIONS FOR GRADIENT DESCENT TRAINING    ############
    def fit(self,training_data_obj, init_params = {},\
     n_iters = 100, step_size_txn=0.001, step_size_soj=0.001, n_steps_between_print=5, lambda_reg = 1, epsilon_stop=5e-10, plots=False):
        self.training_mode = True
        self.plots = plots
        self.n_steps_between_print=n_steps_between_print
        self.training_data = training_data_obj # PopulationData obj
        self.set_params(init_params)

        #ground truth data that we are trying to fit to
        truth_dict = {'date':self.training_data.filtered_data['date'],\
                                    'hospitalized':self.training_data.get_Admission_counts()}

        def gradient_descent():

            def add_to_dict(d,g): 
            # add_to_dict({'a':1,'b':3}, 77) --> {'a':78,'b':80}
                for k in d.keys():
                    d[k]=d[k]+g[k]
                return d

            def scale_dict_vals(d,s_txn,s_soj):
            # scale_dict_vals({'a':1,'b':3}, 2) --> {'a':2,'b':6}
                for k in d.keys():
                    if k in ['T_serial','prob_sympt_s','prob_ailing_s','prob_hosp_s']: # scaling factor for transition parameters
                        d[k] = s_txn*d[k]
                    elif k in ['prob_soujourn_inf_alpha_s', 'prob_soujourn_inf_beta_s', 'prob_soujourn_symp_alpha_s', 'prob_soujourn_symp_beta_s']: # scaling factor for sojourn parameters
                        d[k] = s_soj*d[k]
                    elif k in ['Rt_shift', 'days_of_imposed_restrictions', 'days_of_relaxed_restrictions', 'prob_soujourn_symp_beta_s']: # scaling factor for sojourn parameters
                        pass
                    else:
                        print('UNHANDLED SCALE DICT k=', k)
                return d

            self.gradients_per_iter = {k: [0]*n_iters for k in list(self.params.keys())}
            self.loss_per_iter = [0]*n_iters

            for n in range(n_iters):
                self.iter = n
                if n == 0:
                    new_params = self.params
                else:
                    new_params = add_to_dict(new_params, scale_dict_vals(new_gradients, -1*step_size_txn,-1*step_size_soj)) # gradient update via 2 separate learning rates -step_size_txn and -step_size_soj
               
                new_gradients = self.get_gradients_of_loss(new_params, truth_dict,lambda_reg= lambda_reg)
               
                for k in self.gradients_per_iter.keys(): # store each partial gradient 
                    self.gradients_per_iter[k][n] = new_gradients[k]

                if self.iter%self.n_steps_between_print == 0:
                    print('----------------------Iter ',self.iter,'----------------------')
                    print(self.loss_per_iter[self.iter], 'loss at iteration ')
                    print(self.gradients_per_iter['prob_sympt_s'][self.iter], 'symptomatic gradient at iteration ')
            
            return new_params

        # run gradient descent
        new_params = gradient_descent()

    def get_loss_given_truthdict_and_params(self, params, truth_dict, lambda_reg):
        pred_dict = self.get_forecasted_data(params)
        log_loss = 0

        num_preds = len(pred_dict['hospitalized'])
        log_loss+= np.sum(np.array([2*(e/num_preds)*poisson.logpmf(truth,pred+1e-9) for e,(pred,truth) in enumerate(zip(pred_dict['hospitalized'],truth_dict['hospitalized']))]) )
        
        reg_penalty = (lambda_reg*self.get_log_prior_penalty(params))

        if self.plots and self.iter%self.n_steps_between_print == 0:
            plt.figure(figsize=(15,4))
            plt.plot(np.array(pred_dict['hospitalized'])._value, label='model-prediction')
            plt.plot(truth_dict['hospitalized'], label='ground-truth')
            plt.grid()
            x_dates = truth_dict['date']
            x_dates = [x if e%3==0 else '' for e,x in enumerate(x_dates)]   
            plt.xticks(range(len(x_dates)),x_dates, rotation=90)
            plt.ylabel('hospital-admissible people')
            plt.title('current model fit to ground-truth-training-dataset')
            plt.legend()
            plt.show()

        self.loss_per_iter[self.iter] = float(-log_loss._value + reg_penalty._value)
        return  -log_loss + reg_penalty

    def get_log_prior_penalty(self,params):
        log_prior_penalty = 0
        
        for k,v in self.priors.items():
            if k in ['T_serial', 'Rt_shift', 'days_of_imposed_restrictions', 'days_of_relaxed_restrictions']:
                continue

            a = v[0]
            b = v[1]
            if k in ['prob_sympt_s','prob_ailing_s','prob_hosp_s']:
                log_prior_penalty += beta.logpdf(sigmoid(params[k]), a, b)
                
            elif k in ['prob_soujourn_inf_alpha_s','prob_soujourn_inf_beta_s','prob_soujourn_symp_alpha_s','prob_soujourn_symp_beta_s']:
                log_prior_penalty += log_gamma_pdf(softplus(params[k]),a,b) ## TODO redefine this fn as autogradable
                
        return -log_prior_penalty

    def get_gradients_of_loss(self, params, truth_dict, lambda_reg):
        grad_of_loss = autograd.grad(self.get_loss_given_truthdict_and_params)
        return grad_of_loss(params,truth_dict,lambda_reg)

    def get_dates_to_forecast(self):
        # print(self.warmup_data.filtered_data['date'], 'final warm up date')
        most_recent_estimated_date = str(self.warmup_data.filtered_data['date'][-1])
        dates_to_forecast = [int(most_recent_estimated_date)]
        if self.training_mode:
            final_date = datetime.strptime(self.training_data.end_date, "%Y%m%d")
        else:
            final_date = datetime.strptime(self.warmup_data.end_date, "%Y%m%d")+timedelta(days=self.forecast_duration)
        for t in range((final_date - datetime.strptime(most_recent_estimated_date, "%Y%m%d")).days ):
            dates_to_forecast+=[int( (datetime.strptime(str(dates_to_forecast[-1]), "%Y%m%d")+timedelta(days=1)).strftime("%Y%m%d") )]

        return dates_to_forecast
