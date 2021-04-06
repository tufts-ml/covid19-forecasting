import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import scipy
from scipy import stats
import seaborn as sns

sns.set_style("whitegrid")

def plot_stat(stats_filepath, stat, last_N_not_nan=None):
    stats = pd.read_csv(stats_filepath, index_col=0)
    data = np.array(stats[stat])

    if last_N_not_nan:
        data = data[~np.isnan(data)][-last_N_not_nan:]

    sns.set_context("notebook", font_scale=1.25)
    plt.figure(figsize=(10, 6))
    plt.plot(data, marker='.')
    plt.ylabel(stat)
    plt.xlabel('Iterations')
    plt.tight_layout()
    plt.show()



def sample_params_from_prior(prior_dict, states, num_samples=100):
    samples = {}
    for state in states:
        samples['proba_Recovering_given_%s' % state] = [np.random.dirichlet(prior_dict['prior_Health_given_%s' % state])[1] for _ in range(num_samples)]
        
        if state != 'OnVentInICU':
            samples['proba_Die_after_Declining_%s' % state] = [np.random.dirichlet(prior_dict['prior_Die_after_Declining_%s' % state])[1] for _ in range(num_samples)]
        
        for health in ['Recovering', 'Declining']:
            lam_dict = prior_dict['prior_duration_%s_%s' % (health, state)]['lam']
            tau_dict = prior_dict['prior_duration_%s_%s' % (health, state)]['tau']
            
            a = lam_dict['lower']
            b = lam_dict['upper']
            lam_mean = lam_dict['mean']
            lam_stddev = lam_dict['stddev']
            alpha = (a - lam_mean) / lam_stddev
            beta = (b - lam_mean) / lam_stddev

            tau_mean = tau_dict['mean']
            tau_stddev = tau_dict['stddev']
            
            lam = scipy.stats.truncnorm(alpha, beta, loc=lam_mean, scale=lam_stddev)
            tau = scipy.stats.norm(loc=tau_mean, scale=tau_stddev)
            
            samples['pmf_duration_%s_%s' % (health, state)] = []
            for _ in range(num_samples):
                lam_sample = lam.rvs(size=1)[0]
                tau_sample = tau.rvs(size=1)[0]
                samples['pmf_duration_%s_%s' % (health, state)].append(scipy.special.softmax(scipy.stats.poisson.logpmf(np.arange(int(b)), lam_sample) / np.power(10, tau_sample)))
            
            samples['pmf_duration_%s_%s' % (health, state)] = np.vstack(samples['pmf_duration_%s_%s' % (health, state)])
    
    return samples

# Arguments: 
#   - config file that contains a pointer to a samples file
#   - number of samples wish to be taken from the samples file
# Note: the *last* N samples are taken, so we assume that the last samples are considered to be the most relevant
def gather_params(config_file, num_samples=None):
    with open(config_file, 'r') as f:
        config = json.load(f)

    states = config['states']

    with open(config['samples_file'], 'r') as f:
        last_thetas = json.load(f)['last_samples']
        
        if num_samples is not None:
            last_thetas = last_thetas[-num_samples:]
    
    # initialize results
    results = {}
    for theta in last_thetas:
        for state in states:
            results['proba_Recovering_given_%s' % state] = []
            if state != 'OnVentInICU':
                results['proba_Die_after_Declining_%s' % state] = []
            for health in ['Recovering', 'Declining']:
                results['pmf_duration_%s_%s' % (health, state)] = {}
                max_dur = len(config['pmf_duration_%s_%s' % (health, state)])
                durations = [str(x) for x in range(1, max_dur+1)] + ['lam', 'tau']
                for dur in durations:
                    results['pmf_duration_%s_%s' % (health, state)][dur] = []

    # fill up results
    for theta in last_thetas:
        for state in states:
            results['proba_Recovering_given_%s' % state].append(theta['proba_Recovering_given_%s' % state])
            if state != 'OnVentInICU':
                results['proba_Die_after_Declining_%s' % state].append(theta['proba_Die_after_Declining_%s' % state])
            for health in ['Recovering', 'Declining']:
                max_dur = len(config['pmf_duration_%s_%s' % (health, state)])
                durations = [str(x) for x in range(1, max_dur+1)] + ['lam', 'tau']
                for dur in durations:
                    results['pmf_duration_%s_%s' % (health, state)][dur].append(theta['pmf_duration_%s_%s' % (health, state)][dur])
    return results

def plot_params(params_list, filename_prior='priors/abc_prior_config_OnCDCTableReasonable.json', filename_to_save=None, filename_true_params=None, plot_disjointly=False):
    if not isinstance(params_list, list):
        params_list = [params_list]

    num_samples = len(params_list)
    states = ['InGeneralWard', 'OffVentInICU', 'OnVentInICU']

    param_to_prior_dict = {}
    for state in states:
        param_to_prior_dict['proba_Recovering_given_%s' % state] = 'prior_Health_given_%s' % state
        param_to_prior_dict['proba_Die_after_Declining_%s' % state] = 'prior_Die_after_Declining_%s' % state
        param_to_prior_dict['pmf_duration_Declining_%s' % state] = 'prior_duration_Declining_%s' % state
        param_to_prior_dict['pmf_duration_Recovering_%s' % state] = 'prior_duration_Recovering_%s' % state

    param_names = list(params_list[0].keys())
    
    param_distributions = {}
    for params in params_list:
        for name in param_names:
            if name not in param_distributions:
                if type(params[name]) == dict:
                    param_distributions[name] = {}
                    for key in params[name]:
                        param_distributions[name][key] = []
                else:
                    param_distributions[name] = []

            if type(params[name]) == dict:
                for key in params[name]:
                    param_distributions[name][key].append(np.asarray(params[name][key]))
            else:
                param_distributions[name].append(np.asarray(params[name]))
    
    # flatten everything
    for name in param_distributions:
        if type(param_distributions[name]) == dict:
            for key in param_distributions[name]:
                temp = np.array([])
                for arr in param_distributions[name][key]:
                    temp = np.append(temp, arr)
                param_distributions[name][key] = np.copy(temp)
        else:
            temp = np.array([])
            for arr in param_distributions[name]:
                temp = np.append(temp, arr)
            param_distributions[name] = np.copy(temp)

    if filename_true_params is not None:
        with open(filename_true_params, 'r') as f:
            true_params = json.load(f)

    with open(filename_prior, 'r') as f:
        prior_dict = json.load(f)

    prior_samples = sample_params_from_prior(prior_dict, states, num_samples=1000)

    sns.set_context("notebook", font_scale=1.075)

    if not plot_disjointly:
        fig, ax_grid = plt.subplots(nrows=3, ncols=4, figsize=(16, 10))
        ax_grid = ax_grid.flatten()

    i = 0    
    for param in param_distributions:
        if not plot_disjointly:
            ax = ax_grid[i]
        if 'duration' in param:
            durations = [dur for dur in param_distributions[param] if dur not in ['lam', 'tau']]
            mean_durs = [np.mean(param_distributions[param][dur]) for dur in durations]
            upper_durs = [np.percentile(param_distributions[param][dur], 97.5) for dur in durations]
            lower_durs = [np.percentile(param_distributions[param][dur], 2.5) for dur in durations]

            if filename_true_params is not None:
                true_durs = [true_params[param][dur] for dur in durations]
            
            prior_mean_durs = np.mean(prior_samples[param], axis=0)
            prior_upper_durs = np.percentile(prior_samples[param], 97.5, axis=0)
            prior_lower_durs = np.percentile(prior_samples[param], 2.5, axis=0)

            if plot_disjointly:
                plt.figure(figsize=(5, 4))
                if filename_true_params is not None:
                    plt.plot(durations, true_durs, color='k', label='true')
                plt.plot(durations, prior_mean_durs, color='red', label='prior')
                plt.fill_between(durations, prior_lower_durs, prior_upper_durs, color='red', alpha=0.3)
                plt.plot(durations, mean_durs, color='blue', label='posterior\nacross %d runs' % (num_samples))
                plt.fill_between(durations, lower_durs, upper_durs, color='blue', alpha=0.3)        
                plt.title(param)
                plt.xticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
                plt.ylim([0.0, 0.5])
                plt.legend()
                plt.tight_layout()
                if filename_to_save:
                    plt.savefig(filename_to_save + "_%s.pdf" % (param), bbox_inches='tight', pad_inches=0)
                # plt.show()
            else:
                if filename_true_params is not None:
                    ax.plot(durations, true_durs, color='k', label='true')
                
                ax.plot(durations, prior_mean_durs, color='red', label='prior')
                ax.fill_between(durations, prior_lower_durs, prior_upper_durs, color='red', alpha=0.3)
                ax.plot(durations, mean_durs, color='blue', label='posterior\nacross %d runs' % (num_samples))
                ax.fill_between(durations, lower_durs, upper_durs, color='blue', alpha=0.3)        
                ax.set_title(param)
                ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
                ax.set_ylim([0.0, 0.5])
                ax.legend()
        else:

            if plot_disjointly:
                plt.figure(figsize=(5, 4))
                if filename_true_params is not None:
                    plt.axvline(true_params[param], color='k', label='true')
                plt.hist(prior_samples[param], density=True, stacked=True, alpha=0.3, rwidth=1.0, edgecolor='r',  facecolor='r', label='prior')
                plt.hist(param_distributions[param], density=True, stacked=True, alpha=0.5, edgecolor='blue', facecolor='blue', label='posterior\nacross %d runs' % (num_samples))
                plt.title(param)
                if 'Die' in param:
                    plt.xlim((0, 0.1))
                else:
                    plt.xlim((0, 1))
                plt.legend()
                plt.tight_layout()
                if filename_to_save:
                    plt.savefig(filename_to_save + "_%s.pdf" % (param), bbox_inches='tight', pad_inches=0)
                # plt.show()
            else:
                if filename_true_params is not None:
                    ax.axvline(true_params[param], color='k', label='true')
                ax.hist(prior_samples[param], density=True, stacked=True, alpha=0.3, rwidth=1.0, edgecolor='r',  facecolor='r', label='prior')
                ax.hist(param_distributions[param], density=True, stacked=True, alpha=0.5, edgecolor='blue', facecolor='blue', label='posterior\nacross %d runs' % (num_samples))
                ax.set_title(param)
                if 'Die' in param:
                    ax.set_xlim((0, 0.1))
                else:
                    ax.set_xlim((0, 1))
                ax.legend()

        if i == 8:
            i += 2
        else:
            i += 1

    if not plot_disjointly:
        plt.tight_layout()
        if filename_to_save:
            plt.savefig(filename_to_save, bbox_inches='tight', pad_inches=0)
        plt.show()


#################################################

def compute_true_summary_statistics(csv_df, expected_columns, smooth_terminal_counts=True):
    new_dict = {}

    for column in expected_columns:
        if column == 'timestep' or column == 'date':
            continue
        elif column == 'n_occupied_beds':
            new_dict[column] = csv_df['n_InGeneralWard'] + csv_df['n_OffVentInICU'] + csv_df['n_OnVentInICU']
        elif column == 'n_InICU':
            new_dict[column] = csv_df['n_OffVentInICU'] + csv_df['n_OnVentInICU']
        elif column == 'n_TERMINAL' and smooth_terminal_counts:
            try:
                new_dict[column] = csv_df['n_TERMINAL_5daysSmoothed']
            except:
                new_dict[column] = csv_df[column]
        else:
            new_dict[column] = csv_df[column]

    return pd.DataFrame(new_dict)

def plot_forecasts(forecasts_template_path, config_filepath, true_counts_filepath, figure_template_path=None, smooth_terminal_counts=True,
                    expected_columns=['n_InGeneralWard', 'n_OffVentInICU', 'n_OnVentInICU', 'n_InICU', 'n_TERMINAL']):

    title_map = {'n_discharged_InGeneralWard': 'Number of Discharged Patients', 'n_occupied_beds': 'Number of Occupied Beds', 'n_InGeneralWard': 'Number of Patients in General Ward', 'n_OffVentInICU': 'Number of Patients in ICU, not on the Ventilator', 'n_OnVentInICU': 'Number of Patients on the Ventilator in the ICU', 'n_InICU': 'Number of Patients in ICU', 'n_TERMINAL': 'Number of Terminal Patients'}

    true_df = pd.read_csv(true_counts_filepath)

    pred_df = pd.read_csv(forecasts_template_path + '_mean.csv')
    pred_lower_df = pd.read_csv(forecasts_template_path + '_percentile=002.50.csv')
    pred_upper_df = pd.read_csv(forecasts_template_path + '_percentile=097.50.csv')

    timesteps = true_df['timestep']

    with open(config_filepath, 'r') as f:
        config = json.load(f)
        num_training_timesteps = config['num_training_timesteps']

    sns.set_context("notebook", font_scale=1.25)        

    for column in expected_columns:
        plt.figure(figsize=(16, 4))
        plt.axvline(num_training_timesteps, ls='--', color='grey')
        
        if column == 'n_TERMINAL' and smooth_terminal_counts:
            plt.plot(timesteps, true_df['n_TERMINAL_5daysSmoothed'], label='true-smoothed', marker='d', color='k')
            plt.plot(timesteps[num_training_timesteps:], true_df['n_TERMINAL'][num_training_timesteps:], label='true', marker='o', color='brown')
        else:
            plt.plot(timesteps, true_df[column], label='true', marker='d', color='k')
        
        plt.plot(timesteps, compute_true_summary_statistics(pred_df, expected_columns, smooth_terminal_counts)[column], color='blue', label='ABC')
        plt.fill_between(timesteps, compute_true_summary_statistics(pred_lower_df, expected_columns, smooth_terminal_counts)[column], compute_true_summary_statistics(pred_upper_df, expected_columns, smooth_terminal_counts)[column], color='blue', alpha=0.15)
        
        plt.xticks(np.arange(timesteps.shape[0]//7)*7)    
        plt.xlabel('Days since Start of Training Period')
        plt.ylabel('Census Counts')
        plt.title(title_map[column])
        plt.legend(loc='upper left')
        if figure_template_path:
            plt.savefig(figure_template_path + '_%s.pdf' % (column), bbox_inches='tight', pad_inches=0)
        plt.show()

#################################################

if __name__ == '__main__':

    ## Plot statistics across iterations of ABC
    ## works!!!
    # plot_stat('results/US/MA-20201111-20210111-20210211/PRETRAINED_abc_training_stats.csv', 'accepted_distances')

    ## Plot learned posterior and prior
    ## works!!!
    params = gather_params('results/US/CA-20201111-20210111-20210211/PRETRAINED_config_after_abc.json')
    plot_params(params)

    ## Plot forecasts for counts of interest
    ## works!!!
    plot_forecasts('results/US/CA-20201111-20210111-20210211/PRETRAINED_summary_after_abc',
                   'results/US/CA-20201111-20210111-20210211/PRETRAINED_config_after_abc.json',
                   'datasets/US/CA-20201111-20210111-20210211/daily_counts.csv')