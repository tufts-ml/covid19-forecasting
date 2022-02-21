import argparse
import numpy as np
import tensorflow as tf

import os
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense

import tensorflow_probability as tfp
from scipy.stats import beta, truncnorm


# Local imports from model.py, data.py
from model import CovidModel, LogPoissonProb, get_logging_callbacks, Comp, Vax
from data import read_data, create_warmup

import scipy

import matplotlib
import matplotlib.pyplot as plt

def run_model(learning_rate=None, fix_variance=None, data_dir=None, log_dir=None):

    transition_window = 10

    warmup_start = '20210421'
    warmup_end = '20210430'
    train_start = '20210501'
    train_end = '20210731'
    test_start = '20210801'
    test_end = '20210831'

    state = 'Massachusetts'
    state_abbrev = 'MA'


    covid_estim_date = '20210901'
    hhs_date = '20210903'
    owid_date = '20210903'

    df = read_data(data_dir=data_dir,
                   covid_estim_date=covid_estim_date,
                   hhs_date=hhs_date,
                   owid_date=owid_date,
                   state=state, state_abbrev=state_abbrev)

    warmup_asymp, warmup_mild, count_gen, count_icu = create_warmup(df,
                                                                    warmup_start,
                                                                    warmup_end,
                                                                    0, 0, 0, 0)

    vax_statuses = [Vax.yes, Vax.no]

    warmup_A_params = {}
    warmup_M_params = {}
    warmup_G_params = {}
    warmup_GR_params = {}
    init_count_G = {}
    warmup_I_params = {}
    warmup_IR_params = {}
    init_count_I = {}

    for vax_status in [status.value for status in vax_statuses]:
        warmup_A_params[vax_status] = {}
        warmup_A_params[vax_status]['prior'] = []
        warmup_A_params[vax_status]['posterior_init'] = []

        warmup_M_params[vax_status] = {}
        warmup_M_params[vax_status]['prior'] = []
        warmup_M_params[vax_status]['posterior_init'] = []

        warmup_G_params[vax_status] = {}
        warmup_G_params[vax_status]['prior'] = []
        warmup_G_params[vax_status]['posterior_init'] = []
        warmup_GR_params[vax_status] = {}
        warmup_GR_params[vax_status]['prior'] = []
        warmup_GR_params[vax_status]['posterior_init'] = []

        warmup_I_params[vax_status] = {}
        warmup_I_params[vax_status]['prior'] = []
        warmup_I_params[vax_status]['posterior_init'] = []
        warmup_IR_params[vax_status] = {}
        warmup_IR_params[vax_status]['prior'] = []
        warmup_IR_params[vax_status]['posterior_init'] = []

        init_count_G[vax_status] = {}
        init_count_G[vax_status]['prior'] = {}
        init_count_G[vax_status]['posterior_init'] = {}

        init_count_I[vax_status] = {}
        init_count_I[vax_status]['prior'] = {}
        init_count_I[vax_status]['posterior_init'] = {}

        # Set priors

    init_count_G[0]['prior'] = {'loc': 0.9*count_gen[0], 'scale': count_gen[0]*0.9/10}
    init_count_I[0]['prior'] = {'loc': 0.9*count_icu[0], 'scale': count_icu[0]*0.9 / 10}
    init_count_G[1]['prior'] = {'loc': 0.1 * count_gen[0], 'scale': count_gen[0] * 0.1 / 10}
    init_count_I[1]['prior'] = {'loc': 0.1 * count_icu[0], 'scale': count_icu[0] * 0.1 / 10}

    warmup_A_params[0]['prior'] = {'intercept': warmup_asymp[0][0] / 2,
                                            'slope': 0,
                                            'scale': warmup_asymp[0][0] / 2 / 10}
    warmup_A_params[1]['prior'] = {'intercept': warmup_asymp[1][0] / 2,
                                            'slope': 0,
                                            'scale': warmup_asymp[1][0] / 2 / 10}

    warmup_M_params[0]['prior'] = {'intercept': warmup_mild[0][0] / 2,
                                            'slope': 0,
                                            'scale': warmup_mild[0][0] / 2 / 10}
    warmup_M_params[1]['prior'] = {'intercept': warmup_mild[1][0] / 2,
                                   'slope': 0,
                                   'scale': warmup_mild[1][0] / 2 / 10}

    warmup_G_params[0]['prior'] = {'intercept': count_gen[0] / 5,
                                            'slope': 0,
                                            'scale': count_gen[0] / 5}
    warmup_G_params[1]['prior'] = {'intercept': count_gen[0]*0.1 / 5,
                                   'slope': 0,
                                   'scale': count_gen[0]*0.1 / 5}

    warmup_GR_params[0]['prior'] = {'intercept': count_gen[0]/5/2,
                                             'slope': 0,
                                             'scale': count_gen[0]/5/2}
    warmup_GR_params[1]['prior'] = {'intercept': count_gen[0]*0.1 / 5 / 2,
                                             'slope': 0,
                                             'scale': count_gen[0]*0.1 / 5 / 2}

    warmup_I_params[0]['prior'] = {'intercept': count_icu[0] /5,
                                            'slope': 0,
                                            'scale': count_icu[0] / 5}
    warmup_I_params[1]['prior'] = {'intercept': count_icu[0] *0.1/ 5,
                                   'slope': 0,
                                   'scale': count_icu[0]*0.1 / 5}

    warmup_IR_params[0]['prior'] = {'intercept': count_icu[0] / 5/2,
                                   'slope': 0,
                                   'scale': count_icu[0] / 5/2}
    warmup_IR_params[1]['prior'] = {'intercept': count_icu[0] * 0.1 / 5/2,
                                   'slope': 0,
                                   'scale': count_icu[0] * 0.1 / 5/2}

    x_train = tf.cast(df.loc[train_start:train_end, 'Rt'].values, dtype=tf.float32)
    x_test = tf.cast(df.loc[train_start:test_end, 'Rt'].values, dtype=tf.float32)

    y_train = {}
    y_train['G_in'] = tf.cast(df.loc[train_start:train_end, 'general_ward_in'], dtype=tf.float32)
    y_train['G_count'] = tf.cast(df.loc[train_start:train_end, 'general_ward_count'], dtype=tf.float32)
    y_train['I_count'] = tf.cast(df.loc[train_start:train_end, 'icu_count'], dtype=tf.float32)
    y_train['D_in'] = tf.cast(df.loc[train_start:train_end, 'deaths_covid'], dtype=tf.float32)

    y_test = {}
    y_test['G_in'] = tf.cast(df.loc[train_start:test_end, 'general_ward_in'], dtype=tf.float32)
    y_test['G_count'] = tf.cast(df.loc[train_start:test_end, 'general_ward_count'], dtype=tf.float32)
    y_test['I_count'] = tf.cast(df.loc[train_start:test_end, 'icu_count'], dtype=tf.float32)
    y_test['D_in'] = tf.cast(df.loc[train_start:test_end, 'deaths_covid'], dtype=tf.float32)

    T_serial = {}
    T_serial['prior'] = {'loc': 5.8, 'scale': 1}

    epsilon = {}
    epsilon['prior'] = {'a': 1, 'b': 1}

    delta = {}
    delta['prior'] = {'a': 1, 'b': 1}

    rho_M = {}
    lambda_M = {}
    nu_M = {}
    rho_G = {}
    lambda_G = {}
    nu_G = {}

    rho_I = {}
    lambda_I = {}
    nu_I = {}
    lambda_I_bar = {}
    nu_I_bar = {}

    rho_D = {}
    lambda_D = {}
    nu_D = {}
    lambda_D_bar = {}
    nu_D_bar = {}

    for vax_status in [status.value for status in vax_statuses]:
        rho_M[vax_status] = {}
        rho_M[vax_status]['prior'] = {'a': 1, 'b': 1}
        # rho_M[vax_status]['prior'] = {'a': 31.8, 'b': 10.3}

        lambda_M[vax_status] = {}
        lambda_M[vax_status]['prior'] = {'loc': 4.7, 'scale': 1}

        nu_M[vax_status] = {}
        nu_M[vax_status]['prior'] = {'loc': 3.1, 'scale': 1.2}

        rho_G[vax_status] = {}
        rho_G[vax_status]['prior'] = {'a': 1, 'b': 1}
        # rho_G[vax_status]['prior'] = {'a': 31.8, 'b': 10.3}
        lambda_G[vax_status] = {}
        lambda_G[vax_status]['prior'] = {'loc': 3.3, 'scale': 1.0}
        nu_G[vax_status] = {}
        nu_G[vax_status]['prior'] = {'loc': 9.0, 'scale': 3}

        rho_I[vax_status] = {}
        rho_I[vax_status]['prior'] = {'a': 1, 'b': 1}
        # rho_I[vax_status]['prior'] = {'a': 31.8, 'b': 10.3}
        lambda_I[vax_status] = {}
        lambda_I[vax_status]['prior'] = {'loc': 3.3, 'scale': 1.0}
        nu_I[vax_status] = {}
        nu_I[vax_status]['prior'] = {'loc': 9.0, 'scale': 3}
        lambda_I_bar[vax_status] = {}
        lambda_I_bar[vax_status]['prior'] = {'loc': 3.3, 'scale': 1.0}
        nu_I_bar[vax_status] = {}
        nu_I_bar[vax_status]['prior'] = {'loc': 9.0, 'scale': 3}

        rho_D[vax_status] = {}
        rho_D[vax_status]['prior'] = {'a': 1, 'b': 1}
        # rho_D[vax_status]['prior'] = {'a': 31.8, 'b': 10.3}
        lambda_D[vax_status] = {}
        lambda_D[vax_status]['prior'] = {'loc': 3.3, 'scale': 1.0}
        nu_D[vax_status] = {}
        nu_D[vax_status]['prior'] = {'loc': 9.0, 'scale': 3}
        lambda_D_bar[vax_status] = {}
        lambda_D_bar[vax_status]['prior'] = {'loc': 3.3, 'scale': 1.0}
        nu_D_bar[vax_status] = {}
        nu_D_bar[vax_status]['prior'] = {'loc': 9.0, 'scale': 3}

    T_serial_scale = 1.0
    delta_scale = 0.2
    epsilon_scale = 0.3

    rho_M_scale = 0.1
    lambda_M_scale = 1.0
    nu_M_scale = 1.2

    rho_G_scale = 0.1
    lambda_G_scale = 1.0
    nu_G_scale = 0.2

    rho_I_scale = 0.1
    lambda_I_scale = 1.0
    nu_I_scale = 0.2
    lambda_I_bar_scale = 1.0
    nu_I_bar_scale = 0.2

    rho_D_scale = 0.1
    lambda_D_scale = 1.0
    nu_D_scale = 0.2
    lambda_D_bar_scale = 1.0
    nu_D_bar_scale = 0.2

    T_serial['posterior_init'] = {'loc': tfp.math.softplus_inverse(5.3),
                                  'scale': tf.cast(tfp.math.softplus_inverse(T_serial_scale), dtype=tf.float32)}
    delta['posterior_init'] = {'loc': tf.cast(np.log(0.03 / (1 - 0.03)), dtype=tf.float32),
                               'scale': tf.cast(tfp.math.softplus_inverse(delta_scale), dtype=tf.float32)}
    epsilon['posterior_init'] = {'loc': tf.cast(np.log(0.2 / (1 - 0.5)), dtype=tf.float32),
                                 'scale': tf.cast(tfp.math.softplus_inverse(epsilon_scale), dtype=tf.float32)}


    rho_M_unvax =0.5
    rho_M_vax = 0.45
    rho_M[0]['posterior_init'] = {'loc': tf.cast(np.log(rho_M_unvax / (1 - rho_M_unvax)), dtype=tf.float32),
                                           'scale': tf.cast(tfp.math.softplus_inverse(rho_M_scale),
                                                            dtype=tf.float32)}
    rho_M[1]['posterior_init'] = {'loc': tf.cast(np.log(rho_M_vax / (1 - rho_M_vax)), dtype=tf.float32),
                                  'scale': tf.cast(tfp.math.softplus_inverse(rho_M_scale),
                                                   dtype=tf.float32)}

    lambda_M[0]['posterior_init'] = {'loc': tf.cast(tfp.math.softplus_inverse(3.0), dtype=tf.float32),
                                              'scale': tf.cast(tfp.math.softplus_inverse(lambda_M_scale),
                                                               dtype=tf.float32)}
    lambda_M[1]['posterior_init'] = {'loc': tf.cast(tfp.math.softplus_inverse(3.0), dtype=tf.float32),
                                              'scale': tf.cast(tfp.math.softplus_inverse(lambda_M_scale),
                                                               dtype=tf.float32)}

    nu_M[0]['posterior_init'] = {'loc': tf.cast(tfp.math.softplus_inverse(5.0), dtype=tf.float32),
                                          'scale': tf.cast(tfp.math.softplus_inverse(nu_M_scale), dtype=tf.float32)}
    nu_M[1]['posterior_init'] = {'loc': tf.cast(tfp.math.softplus_inverse(5.0), dtype=tf.float32),
                                 'scale': tf.cast(tfp.math.softplus_inverse(nu_M_scale), dtype=tf.float32)}

    rho_G_unvax = 0.1
    rho_G_vax = 0.01
    rho_G[0]['posterior_init'] = {'loc': tf.cast(np.log(rho_G_unvax / (1 - rho_G_unvax)), dtype=tf.float32),
                                           'scale': tf.cast(tfp.math.softplus_inverse(rho_G_scale),
                                                            dtype=tf.float32)}
    rho_G[1]['posterior_init'] = {'loc': tf.cast(np.log(rho_G_vax / (1 - rho_G_vax)), dtype=tf.float32),
                                  'scale': tf.cast(tfp.math.softplus_inverse(rho_G_scale),
                                                   dtype=tf.float32)}

    lambda_G[0]['posterior_init'] = {'loc': tf.cast(tfp.math.softplus_inverse(3.3), dtype=tf.float32),
                                              'scale': tf.cast(tfp.math.softplus_inverse(lambda_G_scale),
                                                               dtype=tf.float32)}
    lambda_G[1]['posterior_init'] = {'loc': tf.cast(tfp.math.softplus_inverse(3.3), dtype=tf.float32),
                                     'scale': tf.cast(tfp.math.softplus_inverse(lambda_G_scale),
                                                      dtype=tf.float32)}

    nu_G[0]['posterior_init'] = {'loc': tf.cast(tfp.math.softplus_inverse(9.0), dtype=tf.float32),
                                          'scale': tf.cast(tfp.math.softplus_inverse(nu_G_scale), dtype=tf.float32)}
    nu_G[1]['posterior_init'] = {'loc': tf.cast(tfp.math.softplus_inverse(9.0), dtype=tf.float32),
                                 'scale': tf.cast(tfp.math.softplus_inverse(nu_G_scale), dtype=tf.float32)}

    rho_I_unvax = 0.1
    rho_I_vax = 0.01
    rho_I[0]['posterior_init'] = {'loc': tf.cast(np.log(rho_I_unvax / (1 - rho_I_unvax)), dtype=tf.float32),
                                           'scale': tf.cast(tfp.math.softplus_inverse(rho_I_scale),
                                                            dtype=tf.float32)}
    rho_I[1]['posterior_init'] = {'loc': tf.cast(np.log(rho_I_vax / (1 - rho_I_vax)), dtype=tf.float32),
                                 'scale': tf.cast(tfp.math.softplus_inverse(rho_I_scale),
                                                  dtype=tf.float32)}

    lambda_I[0]['posterior_init'] = {'loc': tf.cast(tfp.math.softplus_inverse(3.3), dtype=tf.float32),
                                              'scale': tf.cast(tfp.math.softplus_inverse(lambda_I_scale),
                                                               dtype=tf.float32)}
    lambda_I[1]['posterior_init'] = {'loc': tf.cast(tfp.math.softplus_inverse(3.3), dtype=tf.float32),
                                     'scale': tf.cast(tfp.math.softplus_inverse(lambda_I_scale),
                                                      dtype=tf.float32)}

    nu_I[0]['posterior_init'] = {'loc': tf.cast(tfp.math.softplus_inverse(9.0), dtype=tf.float32),
                                          'scale': tf.cast(tfp.math.softplus_inverse(nu_I_scale), dtype=tf.float32)}
    nu_I[1]['posterior_init'] = {'loc': tf.cast(tfp.math.softplus_inverse(9.0), dtype=tf.float32),
                                 'scale': tf.cast(tfp.math.softplus_inverse(nu_I_scale), dtype=tf.float32)}

    lambda_I_bar[0]['posterior_init'] = {'loc': tf.cast(tfp.math.softplus_inverse(3.3), dtype=tf.float32),
                                                  'scale': tf.cast(tfp.math.softplus_inverse(lambda_I_bar_scale),
                                                                   dtype=tf.float32)}
    lambda_I_bar[1]['posterior_init'] = {'loc': tf.cast(tfp.math.softplus_inverse(3.3), dtype=tf.float32),
                                         'scale': tf.cast(tfp.math.softplus_inverse(lambda_I_bar_scale),
                                                          dtype=tf.float32)}
    nu_I_bar[0]['posterior_init'] = {'loc': tf.cast(tfp.math.softplus_inverse(9.0), dtype=tf.float32),
                                              'scale': tf.cast(tfp.math.softplus_inverse(nu_I_bar_scale),
                                                               dtype=tf.float32)}
    nu_I_bar[1]['posterior_init'] = {'loc': tf.cast(tfp.math.softplus_inverse(9.0), dtype=tf.float32),
                                     'scale': tf.cast(tfp.math.softplus_inverse(nu_I_bar_scale),
                                                      dtype=tf.float32)}

    rho_D_unvax = 0.1
    rho_D_vax = 0.01
    rho_D[0]['posterior_init'] = {'loc': tf.cast(np.log(rho_D_unvax / (1 - rho_D_unvax)), dtype=tf.float32),
                                           'scale': tf.cast(tfp.math.softplus_inverse(rho_D_scale),
                                                            dtype=tf.float32)}
    rho_D[1]['posterior_init'] = {'loc': tf.cast(np.log(rho_D_vax / (1 - rho_D_vax)), dtype=tf.float32),
                                  'scale': tf.cast(tfp.math.softplus_inverse(rho_D_scale),
                                                   dtype=tf.float32)}

    lambda_D[0]['posterior_init'] = {'loc': tf.cast(tfp.math.softplus_inverse(3.3), dtype=tf.float32),
                                              'scale': tf.cast(tfp.math.softplus_inverse(lambda_D_scale),
                                                               dtype=tf.float32)}
    lambda_D[1]['posterior_init'] = {'loc': tf.cast(tfp.math.softplus_inverse(3.3), dtype=tf.float32),
                                     'scale': tf.cast(tfp.math.softplus_inverse(lambda_D_scale),
                                                      dtype=tf.float32)}
    nu_D[0]['posterior_init'] = {'loc': tf.cast(tfp.math.softplus_inverse(9.0), dtype=tf.float32),
                                          'scale': tf.cast(tfp.math.softplus_inverse(nu_D_scale), dtype=tf.float32)}
    nu_D[1]['posterior_init'] = {'loc': tf.cast(tfp.math.softplus_inverse(9.0), dtype=tf.float32),
                                 'scale': tf.cast(tfp.math.softplus_inverse(nu_D_scale), dtype=tf.float32)}
    lambda_D_bar[0]['posterior_init'] = {'loc': tf.cast(tfp.math.softplus_inverse(3.3), dtype=tf.float32),
                                                  'scale': tf.cast(tfp.math.softplus_inverse(lambda_D_bar_scale),
                                                                   dtype=tf.float32)}
    lambda_D_bar[1]['posterior_init'] = {'loc': tf.cast(tfp.math.softplus_inverse(3.3), dtype=tf.float32),
                                         'scale': tf.cast(tfp.math.softplus_inverse(lambda_D_bar_scale),
                                                          dtype=tf.float32)}
    nu_D_bar[0]['posterior_init'] = {'loc': tf.cast(tfp.math.softplus_inverse(9.0), dtype=tf.float32),
                                              'scale': tf.cast(tfp.math.softplus_inverse(nu_D_bar_scale),
                                                               dtype=tf.float32)}
    nu_D_bar[1]['posterior_init'] = {'loc': tf.cast(tfp.math.softplus_inverse(9.0), dtype=tf.float32),
                                     'scale': tf.cast(tfp.math.softplus_inverse(nu_D_bar_scale),
                                                      dtype=tf.float32)}

    init_count_G[0]['posterior_init'] = {
        'loc': tf.cast(tfp.math.softplus_inverse(count_gen[0]*0.9 / 100), dtype=tf.float32),
        'scale': tf.cast(tfp.math.softplus_inverse(count_gen[0]*0.9 / 100 / 10), dtype=tf.float32)}
    init_count_G[1]['posterior_init'] = {
        'loc': tf.cast(tfp.math.softplus_inverse(count_gen[0] * 0.1 / 100), dtype=tf.float32),
        'scale': tf.cast(tfp.math.softplus_inverse(count_gen[0] * 0.1 / 100 / 10), dtype=tf.float32)}
    init_count_I[0]['posterior_init'] = {
        'loc': tf.cast(tfp.math.softplus_inverse(count_icu[0] *0.9/ 100), dtype=tf.float32),
        'scale': tf.cast(tfp.math.softplus_inverse(count_icu[0]*0.9 / 100 / 10), dtype=tf.float32)}
    init_count_I[1]['posterior_init'] = {
        'loc': tf.cast(tfp.math.softplus_inverse(count_icu[0] * 0.1 / 100), dtype=tf.float32),
        'scale': tf.cast(tfp.math.softplus_inverse(count_icu[0] * 0.1 / 100 / 10), dtype=tf.float32)}

    # must be positive so reverse softplus the mean
    warmup_A_params[0]['posterior_init'] = {
        'intercept': tf.cast(tfp.math.softplus_inverse(1000.0 / 100 ), dtype=tf.float32),
        'slope': tf.cast(0.0, dtype=tf.float32),
        'scale': tf.cast(tfp.math.softplus_inverse(250.0 / 100 ), dtype=tf.float32)}
    warmup_A_params[1]['posterior_init'] = {
        'intercept': tf.cast(tfp.math.softplus_inverse(1000.0 / 100 ), dtype=tf.float32),
        'slope': tf.cast(0.0, dtype=tf.float32),
        'scale': tf.cast(tfp.math.softplus_inverse(250.0 / 100 ), dtype=tf.float32)}

    warmup_M_params[0]['posterior_init'] = {
        'intercept': tf.cast(tfp.math.softplus_inverse(500.0 / 100 ), dtype=tf.float32),
        'slope': tf.cast(0.0, dtype=tf.float32),
        'scale': tf.cast(tfp.math.softplus_inverse(50.0 / 100 ), dtype=tf.float32)}
    warmup_M_params[1]['posterior_init'] = {
        'intercept': tf.cast(tfp.math.softplus_inverse(500.0 / 100 ), dtype=tf.float32),
        'slope': tf.cast(0.0, dtype=tf.float32),
        'scale': tf.cast(tfp.math.softplus_inverse(50.0 / 100  ), dtype=tf.float32)}

    warmup_G_params[0]['posterior_init'] = {
        'intercept': tf.cast(tfp.math.softplus_inverse(250 / 100 ), dtype=tf.float32),
        'slope': tf.cast(0.0, dtype=tf.float32),
        'scale': tf.cast(tfp.math.softplus_inverse(25 / 100 ), dtype=tf.float32)}
    warmup_G_params[1]['posterior_init'] = {
        'intercept': tf.cast(tfp.math.softplus_inverse(25 / 100), dtype=tf.float32),
        'slope': tf.cast(0.0, dtype=tf.float32),
        'scale': tf.cast(tfp.math.softplus_inverse(10 / 100 ), dtype=tf.float32)}

    warmup_GR_params[0]['posterior_init'] = {
        'intercept': tf.cast(tfp.math.softplus_inverse(200.0 / 100 ), dtype=tf.float32),
        'slope': tf.cast(0.0, dtype=tf.float32),
        'scale': tf.cast(tfp.math.softplus_inverse(25.0 / 100 ), dtype=tf.float32)}
    warmup_GR_params[1]['posterior_init'] = {
        'intercept': tf.cast(tfp.math.softplus_inverse(12.5 / 100), dtype=tf.float32),
        'slope': tf.cast(0.0, dtype=tf.float32),
        'scale': tf.cast(tfp.math.softplus_inverse(5/ 100), dtype=tf.float32)}

    warmup_I_params[0]['posterior_init'] = {
        'intercept': tf.cast(tfp.math.softplus_inverse(90 / 100 ), dtype=tf.float32),
        'slope': tf.cast(0.0, dtype=tf.float32),
        'scale': tf.cast(tfp.math.softplus_inverse(30.0 / 100 ), dtype=tf.float32)}
    warmup_I_params[1]['posterior_init'] = {
        'intercept': tf.cast(tfp.math.softplus_inverse(10 / 100), dtype=tf.float32),
        'slope': tf.cast(0.0, dtype=tf.float32),
        'scale': tf.cast(tfp.math.softplus_inverse(3 / 100), dtype=tf.float32)}

    warmup_IR_params[0]['posterior_init'] = {
        'intercept': tf.cast(tfp.math.softplus_inverse(45 / 100), dtype=tf.float32),
        'slope': tf.cast(0.0, dtype=tf.float32),
        'scale': tf.cast(tfp.math.softplus_inverse(20.0 / 100 ), dtype=tf.float32)}
    warmup_IR_params[1]['posterior_init'] = {
        'intercept': tf.cast(tfp.math.softplus_inverse(5 / 100), dtype=tf.float32),
        'slope': tf.cast(0.0, dtype=tf.float32),
        'scale': tf.cast(tfp.math.softplus_inverse(2 / 100), dtype=tf.float32)}

    model = CovidModel([Vax.no, Vax.yes], [Comp.A, Comp.M, Comp.G, Comp.GR, Comp.I, Comp.IR, Comp.D],
                       transition_window,
                       T_serial, epsilon, delta,
                       rho_M, lambda_M, nu_M,
                       rho_G, lambda_G, nu_G,
                       rho_I, lambda_I, nu_I,
                       lambda_I_bar, nu_I_bar,
                       rho_D, lambda_D, nu_D,
                       lambda_D_bar, nu_D_bar,
                       warmup_A_params,
                       warmup_M_params,
                       warmup_G_params, warmup_GR_params, init_count_G,
                       warmup_I_params, warmup_IR_params, init_count_I, posterior_samples=1000,
                       debug_disable_theta=False, fix_variance=fix_variance)

    pre_training_preds = model.call(x_train)

    logging_callbacks = get_logging_callbacks(log_dir)
    loss = LogPoissonProb()
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,  # beta_1=0.1, beta_2=0.1
    )

    model.compile(loss=loss, optimizer=optimizer, run_eagerly=True)
    model.fit(x=np.asarray([x_train]),
              y=np.asarray([(y_train['G_count'], y_train['G_in'], y_train['I_count'], y_train['D_in'])]),
              epochs=1000, batch_size=0,
              callbacks=logging_callbacks)

    preds=tf.reduce_mean(model.call(x_test), axis=-1)

    icu_plot_loc = os.path.join(log_dir, 'icu.png')
    plt.figure(figsize=(8, 6))
    preds = tf.reduce_mean(model.call(x_test), axis=-1)
    plt.plot(df.loc[train_start:test_end].index.values, y_test['I_count'], label='ICU')
    plt.plot(df.loc[train_start:test_end].index.values, preds[0][2], label='ICU')
    month_ticks = matplotlib.dates.MonthLocator(interval=1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(month_ticks)
    plt.title('ICU Count')
    plt.savefig(icu_plot_loc)


    gen_plot_loc = os.path.join(log_dir, 'gen_c.png')
    plt.figure(figsize=(8, 6))
    plt.plot(df.loc[train_start:test_end].index.values, y_test['G_count'], )
    plt.plot(df.loc[train_start:test_end].index.values, preds[0][0])
    month_ticks = matplotlib.dates.MonthLocator(interval=1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(month_ticks)
    plt.legend()
    plt.title('Gen Count')
    plt.savefig(gen_plot_loc)

    death_plot_loc = os.path.join(log_dir, 'death.png')
    plt.figure(figsize=(8, 6))
    plt.plot(df.loc[train_start:test_end].index.values, y_test['D_in'], )
    plt.plot(df.loc[train_start:test_end].index.values, preds[0][3])
    month_ticks = matplotlib.dates.MonthLocator(interval=1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(month_ticks)
    plt.legend()
    plt.title('Death Influx')
    plt.savefig(death_plot_loc)

    return



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run a covid model')

    parser.add_argument('--fix_variance', help='fix variance if present', action='store_true')
    parser.add_argument('--log_dir', help='Path to directory to write logs into', type=str)
    parser.add_argument('--data_dir', help='Wheres the data at', type=str)
    parser.add_argument('--learning_rate', help='Learning rate', type=float,
                        default=1e-1)

    args = parser.parse_args()
    run_model(**vars(args))
