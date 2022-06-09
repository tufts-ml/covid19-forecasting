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
from model_config import ModelConfig
from data import read_data, create_warmup

import scipy

import matplotlib
import matplotlib.pyplot as plt

def run_model(model_config_path=None, learning_rate=None, fix_variance=None, data_dir=None, log_dir=None,
              state=None, state_abbrev=None, rescale_state=None, rescale_state_abbrev=None):

    transition_window = 10

    warmup_start = '20210421'
    warmup_end = '20210430'
    train_start = '20210501'
    train_end = '20210731'
    test_start = '20210801'
    test_end = '20210831'


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

    config = ModelConfig.from_json(model_config_path)

    if rescale_state is not None and rescale_state_abbrev is not None:
        df_rescale = read_data(data_dir=data_dir,
                          covid_estim_date=covid_estim_date,
                          hhs_date=hhs_date,
                          owid_date=owid_date,
                          state=rescale_state, state_abbrev=rescale_state_abbrev)

        rescale_I_count_before = df_rescale.loc[train_start, 'icu_count']
        rescale_G_count_before = df_rescale.loc[train_start, 'general_ward_count']
        rescale_G_in_before = df_rescale.loc[train_start, 'general_ward_in']

        I_count_before = df.loc[train_start, 'icu_count']
        G_count_before = df.loc[train_start, 'general_ward_count']
        G_in_before = df.loc[train_start, 'general_ward_in']

        I_count_scale = I_count_before / rescale_I_count_before
        G_count_scale = G_count_before / rescale_G_count_before
        G_in_scale = G_in_before / rescale_G_in_before

        config.init_count_G.value[0]['loc'] = config.init_count_G.value[0]['loc'] * G_count_scale
        config.init_count_G.value[1]['loc'] = config.init_count_G.value[1]['loc'] * G_count_scale
        config.init_count_I.value[0]['loc'] = config.init_count_I.value[0]['loc'] * I_count_scale
        config.init_count_I.value[1]['loc'] = config.init_count_I.value[1]['loc'] * I_count_scale

        config.warmup_A.value[0]['intercept'] = config.warmup_A.value[0]['intercept'] * G_in_scale
        config.warmup_A.value[1]['intercept'] = config.warmup_A.value[1]['intercept'] * G_in_scale
        config.warmup_A.value[0]['slope'] = config.warmup_A.value[0]['slope'] * G_in_scale
        config.warmup_A.value[1]['slope'] = config.warmup_A.value[1]['slope'] * G_in_scale

        config.warmup_M.value[0]['intercept'] = config.warmup_M.value[0]['intercept'] * G_in_scale
        config.warmup_M.value[1]['intercept'] = config.warmup_M.value[1]['intercept'] * G_in_scale
        config.warmup_M.value[0]['slope'] = config.warmup_M.value[0]['slope'] * G_in_scale
        config.warmup_M.value[1]['slope'] = config.warmup_M.value[1]['slope'] * G_in_scale

        config.warmup_G.value[0]['intercept'] = config.warmup_G.value[0]['intercept'] * G_in_scale
        config.warmup_G.value[1]['intercept'] = config.warmup_G.value[1]['intercept'] * G_in_scale
        config.warmup_G.value[0]['slope'] = config.warmup_G.value[0]['slope'] * G_in_scale
        config.warmup_G.value[1]['slope'] = config.warmup_G.value[1]['slope'] * G_in_scale

        config.warmup_GR.value[0]['intercept'] = config.warmup_GR.value[0]['intercept'] * G_in_scale
        config.warmup_GR.value[1]['intercept'] = config.warmup_GR.value[1]['intercept'] * G_in_scale
        config.warmup_GR.value[0]['slope'] = config.warmup_GR.value[0]['slope'] * G_in_scale
        config.warmup_GR.value[1]['slope'] = config.warmup_GR.value[1]['slope'] * G_in_scale

        config.warmup_I.value[0]['intercept'] = config.warmup_I.value[0]['intercept'] * G_in_scale
        config.warmup_I.value[1]['intercept'] = config.warmup_I.value[1]['intercept'] * G_in_scale
        config.warmup_I.value[0]['slope'] = config.warmup_I.value[0]['slope'] * G_in_scale
        config.warmup_I.value[1]['slope'] = config.warmup_I.value[1]['slope'] * G_in_scale

        config.warmup_IR.value[0]['intercept'] = config.warmup_IR.value[0]['intercept'] * G_in_scale
        config.warmup_IR.value[1]['intercept'] = config.warmup_IR.value[1]['intercept'] * G_in_scale
        config.warmup_IR.value[0]['slope'] = config.warmup_IR.value[0]['slope'] * G_in_scale
        config.warmup_IR.value[1]['slope'] = config.warmup_IR.value[1]['slope'] * G_in_scale


    elif rescale_state is not None or rescale_state_abbrev is not None:
        raise ValueError("Only provided one of rescale state and rescale state abbrev")

    vax_statuses = [Vax.yes, Vax.no]

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



    model = CovidModel([Vax.no, Vax.yes], [Comp.A, Comp.M, Comp.G, Comp.GR, Comp.I, Comp.IR, Comp.D],
                       transition_window,
                       config, posterior_samples=100,
                       debug_disable_theta=False, fix_variance=fix_variance)

    pre_training_preds = model.call(x_train)

    logging_callbacks = get_logging_callbacks(log_dir, df, x_test, y_test, state_abbrev, train_start, train_end, test_start, test_end)
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

    model.config.to_json(os.path.join(log_dir,'final_config.json'))

    return



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run a covid model')

    parser.add_argument('--fix_variance', help='fix variance if present', action='store_true')
    parser.add_argument('--log_dir', help='Path to directory to write logs into', type=str)
    parser.add_argument('--data_dir', help='Wheres the data at', type=str)
    parser.add_argument('--state', help='Full state name to model', type=str)
    parser.add_argument('--state_abbrev', help='2 letter state abbreviation to model', type=str)
    parser.add_argument('--rescale_state', help='Optional, state the config was trained on',
                        type=str, required=False)
    parser.add_argument('--rescale_state_abbrev', help='Optional, 2 letter abbrev of state the config was trained on',
                        type=str, required=False)
    parser.add_argument('--model_config_path', help='Path to model config json', type=str)
    parser.add_argument('--learning_rate', help='Learning rate', type=float,
                        default=1e-1)

    args = parser.parse_args()
    run_model(**vars(args))
