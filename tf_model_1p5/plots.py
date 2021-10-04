import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20}) # set plot font sizes

import numpy as np
import pandas as pd
from scipy.stats import beta, truncnorm
import tensorflow as tf

from model import Compartments


def plot_to_grid(df, warmup_start, warmup_end, train_start, train_end, test_start, test_end,
                 ax, title, in_sample_preds=None,
                 oos_preds=None,
                 truth=None,
                 plot_legend=False, plot_ticks=False):
    provided_vals = []

    if truth is not None:
        ax.plot(truth[0],
                truth[1], label='truth')
        provided_vals += [truth[1]]

    if in_sample_preds is not None:
        ax.plot(in_sample_preds[0],
                in_sample_preds[1], label='in-sample preds')
        provided_vals += [in_sample_preds[1]]
    if oos_preds is not None:
        ax.plot(oos_preds[0],
                oos_preds[1], label='oos_preds')
        provided_vals += [oos_preds[1]]

    max_y = max([max(vals) for vals in provided_vals if vals is not None])
    if truth is not None:
        max_y = max(truth[1])

    h1 = ax.fill_between(df.loc[warmup_start:warmup_end].index.values, 0, max_y, alpha=0.15, color='red',
                         label='warmup')
    h2 = ax.fill_between(df.loc[train_start:train_end].index.values, 0, max_y, alpha=0.15, color='green', label='train')
    h3 = ax.fill_between(df.loc[test_start:test_end].index.values, 0, max_y, alpha=0.15, color='yellow', label='test')

    if plot_legend:
        ax.legend()
    if plot_ticks:
        month_ticks = matplotlib.dates.MonthLocator(interval=1)
        ax.xaxis.set_major_locator(month_ticks)
        # cut off last tick
        ax.set_xticks(ax.get_xticks()[:-1])
    else:
        ax.set_xticks([])

    ax.title.set_text(title)
    ax.set_ylim(0, max_y)


def plot_beta_prior(title, learned_values, alpha_param, beta_param, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111)
    x = np.linspace(0, 1, 1000)
    pdf_vals_0 = beta.pdf(x, alpha_param[0], beta_param[0])
    pdf_vals_1 = beta.pdf(x, alpha_param[1], beta_param[1])
    ax.plot(x, pdf_vals_0 , label='beta prior no vax')
    ax.plot(x, pdf_vals_1 , label='beta prior  vax')
    ax.plot([learned_values[0], learned_values[0]], [0, max(pdf_vals_0)],
            label=f'{title} non-vax',
            linestyle='--', linewidth=5)
    ax.plot([learned_values[1], learned_values[1]], [0, max(pdf_vals_1)],
            label=f'{title} vax',
            linestyle='--', linewidth=5)
    ax.legend(prop={'size': 10})
    ax.title.set_text(title)


def plot_pi_prior(title, learned_values, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111)
    days = len(learned_values[0])
    x = range(1, days + 1)
    ax.plot(x, learned_values[0], '.',
            label=f'{title} non-vax', markersize=15)
    ax.plot(x, learned_values[1], '.',
            label=f'{title} vax', markersize=15)
    ax.legend(prop={'size': 10})
    ax.title.set_text(title)


def make_all_plots(df, model,
                   alpha_bar_M, beta_bar_M,
                   alpha_bar_X, beta_bar_X,
                   alpha_bar_G, beta_bar_G,
                   warmup_start, warmup_end,
                   train_start, train_end,
                   test_start, test_end,
                   train_preds, test_preds,
                   vax_asymp_risk, vax_mild_risk, vax_extreme_risk,
                   forecasted_fluxes,
                   loss_fn,
                   save_path=None):
    fig = plt.figure(figsize=(15, 44))

    ax_hosp_tot = plt.subplot2grid((8, 3), (0, 0), colspan=3, rowspan=2)
    ax_rt = plt.subplot2grid((8, 3), (2, 0))
    ax_hosp_nonvax = plt.subplot2grid((8, 3), (2, 1))
    ax_hosp_vax = plt.subplot2grid((8, 3), (2, 2))
    ax_a_tot = plt.subplot2grid((8, 3), (3, 0))
    ax_a_nonvax = plt.subplot2grid((8, 3), (3, 1))
    ax_a_vax = plt.subplot2grid((8, 3), (3, 2))
    ax_m_tot = plt.subplot2grid((8, 3), (4, 0))
    ax_m_nonvax = plt.subplot2grid((8, 3), (4, 1))
    ax_m_vax = plt.subplot2grid((8, 3), (4, 2))
    ax_x_tot = plt.subplot2grid((8, 3), (5, 0))
    ax_x_nonvax = plt.subplot2grid((8, 3), (5, 1))
    ax_x_vax = plt.subplot2grid((8, 3), (5, 2))
    ax_rho_M = plt.subplot2grid((8, 3), (6, 0))
    ax_rho_X = plt.subplot2grid((8, 3), (6, 1))
    ax_rho_G = plt.subplot2grid((8, 3), (6, 2))
    ax_pi_M = plt.subplot2grid((8, 3), (7, 0))
    ax_pi_X = plt.subplot2grid((8, 3), (7, 1))
    ax_pi_G = plt.subplot2grid((8, 3), (7, 2))

    all_days = df.loc[warmup_start:test_end].index.values
    warmup_days = df.loc[warmup_start:warmup_end].index.values
    train_days = df.loc[train_start:train_end].index.values
    test_days = df.loc[test_start:test_end].index.values
    train_test_days = df.loc[train_start:test_end].index.values

    # Make big G total plot
    plot_to_grid(df, warmup_start, warmup_end, train_start, train_end, test_start, test_end, ax_hosp_tot, 'G, Total',
                 in_sample_preds=(train_days, train_preds),
                 oos_preds=(train_test_days, test_preds),
                 truth=(all_days,
                        df.loc[warmup_start:test_end, 'general_ward'].values),
                 plot_legend=True, plot_ticks=True)

    train_loss_G = loss_fn(tf.convert_to_tensor(df.loc[train_start:train_end, 'general_ward'].values, dtype=tf.float32), train_preds)
    test_loss_G = loss_fn(tf.convert_to_tensor(df.loc[test_start:test_end, 'general_ward'].values, dtype=tf.float32), test_preds[-len(test_days):])

    plot_to_grid(df, warmup_start, warmup_end, train_start, train_end, test_start, test_end, ax_rt, 'Rt',
                 truth=(all_days, df.loc[warmup_start:test_end, 'Rt'].values))

    plot_to_grid(df, warmup_start, warmup_end, train_start, train_end, test_start, test_end, ax_hosp_nonvax, 'G, No Vax',
                 truth=(all_days, df.loc[warmup_start:test_end, 'general_ward'].values),
                 oos_preds=(train_test_days, forecasted_fluxes[Compartments.general_ward.value][0].stack().numpy()))
    plot_to_grid(df, warmup_start, warmup_end, train_start, train_end, test_start, test_end, ax_hosp_vax, 'G, Vax',
                 truth=(all_days, df.loc[warmup_start:test_end, 'general_ward'].values),
                 oos_preds=(train_test_days, forecasted_fluxes[Compartments.general_ward.value][1].stack().numpy()))

    asymp_vax_all_days = (df.loc[warmup_start:test_end, 'vax_pct'] * (1 - vax_asymp_risk) * \
                          df.loc[warmup_start:test_end, 'asymp']).values
    asymp_no_vax_all_days = df.loc[warmup_start:test_end, 'asymp'].values - asymp_vax_all_days

    mild_vax_all_days = (df.loc[warmup_start:test_end, 'vax_pct'] * (1 - vax_mild_risk) * \
                         df.loc[warmup_start:test_end, 'mild']).values
    mild_no_vax_all_days = df.loc[warmup_start:test_end, 'mild'].values - mild_vax_all_days

    extreme_vax_all_days = (df.loc[warmup_start:test_end, 'vax_pct'] * (1 - vax_extreme_risk) * \
                            df.loc[warmup_start:test_end, 'extreme']).values
    extreme_no_vax_all_days = df.loc[warmup_start:test_end, 'extreme'].values - extreme_vax_all_days

    plot_to_grid(df, warmup_start, warmup_end, train_start, train_end, test_start, test_end, ax_a_tot, 'A, Total',
                 truth=(all_days, df.loc[warmup_start:test_end, 'asymp'].values),
                 oos_preds=(all_days, forecasted_fluxes[Compartments.asymp.value][0].stack() + forecasted_fluxes[Compartments.asymp.value][1].stack()))

    train_loss_A = loss_fn( tf.convert_to_tensor(forecasted_fluxes[Compartments.asymp.value][0].stack() +
                          forecasted_fluxes[Compartments.asymp.value][1].stack(), dtype=tf.float32)[len(warmup_days):len(warmup_days)+len(train_days)],
                        tf.convert_to_tensor(df.loc[train_start:train_end, 'asymp'].values,dtype=tf.float32))

    test_loss_A = loss_fn(tf.convert_to_tensor(forecasted_fluxes[Compartments.asymp.value][0].stack() +
                         forecasted_fluxes[Compartments.asymp.value][1].stack(), dtype=tf.float32)[
                        len(warmup_days) + len(train_days):],
                        tf.convert_to_tensor(df.loc[test_start:test_end, 'asymp'].values,dtype=tf.float32))

    plot_to_grid(df, warmup_start, warmup_end, train_start, train_end, test_start, test_end, ax_a_nonvax, 'A, No vax',
                 truth=(all_days, asymp_no_vax_all_days),
                 oos_preds=(all_days, forecasted_fluxes[Compartments.asymp.value][0].stack()))
    plot_to_grid(df, warmup_start, warmup_end, train_start, train_end, test_start, test_end, ax_a_vax, 'A, Vax',
                 truth=(all_days, asymp_no_vax_all_days),
                 oos_preds=(all_days, forecasted_fluxes[Compartments.asymp.value][1].stack()))

    plot_to_grid(df, warmup_start, warmup_end, train_start, train_end, test_start, test_end, ax_m_tot, 'M, Total',
                 truth=(all_days, df.loc[warmup_start:test_end, 'mild'].values),
                 oos_preds=(all_days, forecasted_fluxes[Compartments.mild.value][0].stack() + forecasted_fluxes[Compartments.mild.value][1].stack()))

    train_loss_M = loss_fn(tf.convert_to_tensor(forecasted_fluxes[Compartments.mild.value][0].stack() +
                                                forecasted_fluxes[Compartments.mild.value][1].stack(),
                                                dtype=tf.float32)[len(warmup_days):len(warmup_days) + len(train_days)],
                           tf.convert_to_tensor(df.loc[train_start:train_end, 'mild'].values, dtype=tf.float32))

    test_loss_M = loss_fn(tf.convert_to_tensor(forecasted_fluxes[Compartments.mild.value][0].stack() +
                                               forecasted_fluxes[Compartments.mild.value][1].stack(),
                                               dtype=tf.float32)[
                          len(warmup_days) + len(train_days):],
                          tf.convert_to_tensor(df.loc[test_start:test_end, 'mild'].values, dtype=tf.float32))

    plot_to_grid(df, warmup_start, warmup_end, train_start, train_end, test_start, test_end, ax_m_nonvax, 'M, No vax',
                 truth=(all_days, mild_no_vax_all_days),
                 oos_preds=(all_days, forecasted_fluxes[Compartments.mild.value][0].stack()))
    plot_to_grid(df, warmup_start, warmup_end, train_start, train_end, test_start, test_end, ax_m_vax, 'M, Vax',
                 truth=(all_days, mild_no_vax_all_days),
                 oos_preds=(all_days, forecasted_fluxes[Compartments.mild.value][1].stack()))

    plot_to_grid(df, warmup_start, warmup_end, train_start, train_end, test_start, test_end, ax_x_tot, 'X, Total',
                 truth=(all_days, df.loc[warmup_start:test_end, 'extreme'].values),
                 oos_preds=(all_days, forecasted_fluxes[Compartments.extreme.value][0].stack() + forecasted_fluxes[Compartments.extreme.value][1].stack()))

    train_loss_X = loss_fn(tf.convert_to_tensor(forecasted_fluxes[Compartments.extreme.value][0].stack() +
                                                forecasted_fluxes[Compartments.extreme.value][1].stack(),
                                                dtype=tf.float32)[len(warmup_days):len(warmup_days) + len(train_days)],
                           tf.convert_to_tensor(df.loc[train_start:train_end, 'extreme'].values, dtype=tf.float32))

    test_loss_X = loss_fn(tf.convert_to_tensor(forecasted_fluxes[Compartments.extreme.value][0].stack() +
                                               forecasted_fluxes[Compartments.extreme.value][1].stack(),
                                               dtype=tf.float32)[
                          len(warmup_days) + len(train_days):],
                          tf.convert_to_tensor(df.loc[test_start:test_end, 'extreme'].values, dtype=tf.float32))
    plot_to_grid(df, warmup_start, warmup_end, train_start, train_end, test_start, test_end, ax_x_nonvax, 'X, No vax',
                 truth=(all_days, extreme_no_vax_all_days),
                 oos_preds=(all_days, forecasted_fluxes[Compartments.extreme.value][0].stack()))
    plot_to_grid(df, warmup_start, warmup_end, train_start, train_end, test_start, test_end, ax_x_vax, 'X, Vax',
                 truth=(all_days, extreme_no_vax_all_days),
                 oos_preds=(all_days, forecasted_fluxes[Compartments.extreme.value][1].stack()))

    plot_beta_prior('Rho M', model.rho_M, alpha_bar_M, beta_bar_M, ax=ax_rho_M)
    plot_beta_prior('Rho X', model.rho_X, alpha_bar_X, beta_bar_X, ax=ax_rho_G)
    plot_beta_prior('Rho G', model.rho_G, alpha_bar_G, beta_bar_G, ax=ax_rho_X)
    plot_pi_prior('pi_M', model.pi_M, ax=ax_pi_M)
    plot_pi_prior('pi_X', model.pi_X, ax=ax_pi_X)
    plot_pi_prior('pi_G', model.pi_G, ax=ax_pi_G)

    if save_path is not None:
        plt.savefig(save_path)

    results = pd.DataFrame.from_records([{
        'train_loss_G': train_loss_G.numpy(),
        'test_loss_G': test_loss_G.numpy(),
        'train_loss_A': train_loss_A.numpy(),
        'test_loss_A': test_loss_A.numpy(),
        'train_loss_M': train_loss_M.numpy(),
        'test_loss_M': test_loss_M.numpy(),
        'train_loss_X': train_loss_X.numpy(),
        'test_loss_X': test_loss_X.numpy(),
    }])

    return results



