from collections import defaultdict
import copy
from enum import Enum

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense

import tensorflow_probability as tfp
from scipy.stats import beta, truncnorm


class Comp(Enum):
    """"""
    A = 0
    M = 1
    G = 2
    GR = 3
    I = 4
    IR = 5
    D = 6


class Vax(Enum):
    total = -1
    no = 0
    yes = 1


class CovidModel(tf.keras.Model):

    def __init__(self,
                 vax_statuses, compartments,
                 transition_window, config,
                 posterior_samples=1000, debug_disable_theta=False,
                 fix_variance=False):
        """Covid Model 1.5

        Args:
            transition_window (int): J in our notation, the number of days to consider a
                possible transition to a more severe state
            model parameters (dict): Dictionary with key for 'prior' and 'posterior_init'.
                Prior values are in the variable's natural (constrained) domain.
                Posterior initializations are given in the modeling (unconstrained) domain
            posterior_samples (int): How many samples to take
            debug_disable_theta (bool): Optional, will disable prior losses if True
            fix_variance (bool): Optional, will not learn variance parameters if True
        """
        super(CovidModel, self).__init__()

        self.transition_window = transition_window
        self.vax_statuses = vax_statuses
        self.compartments = compartments
        self.posterior_samples = posterior_samples
        
        self.config = config

        self.scale_transform = tfp.bijectors.Softplus()

        # create dictionaries to store model parameters / prior distributions
        self._initialize_parameters(config, debug_disable_theta, fix_variance)

        self._initialize_priors(config)

    def call(self, r_t, debug_disable_prior=False, return_all=False):
        """Run covid model 1.5

        Args:
            r_t (tf.Tensor): A tensor of our input data
                rt should be size (1, days_to_forecast)
            debug_disable_prior (bool): If True, will disable adding the prior to the loss. Used to debug gradients
        Returns:
            tf.Tensor: A tensor size (1, days_to_forecast) of incident hospital admissions
        """

        self._constrain_parameters()
        self._sample_and_reparameterize()

        # get rid of batch dimension
        r_t = tf.squeeze(r_t)

        forecast_days = r_t.shape[-1]

        # It's a little weird to iteratively write to tensors one day at a time
        # To do this, we'll use TensorArray's, arrays of tensors where each element is a tensor representing
        # one compartment/vaccine status/ day
        # This helper function creates a nested dictionary keyed on:
        #  compartment->
        #      vaccinestatus->
        #           TensorArray with one tensor per day from: forecast_start to forecast_end for
        forecasted_fluxes = self._initialize_flux_arrays(forecast_days)
        forecasted_counts = self._initialize_count_arrays(forecast_days)
        prev_count_G = {}
        prev_count_I = {}
        for day in range(forecast_days):
            for vax_status in [status.value for status in self.vax_statuses]:
                if day-1 < 0:
                    yesterday_asymp_no = self.warmup_A_samples_constrained[Vax.no.value][day-1]
                    yesterday_asymp_yes = self.warmup_A_samples_constrained[Vax.yes.value][day - 1]
                else:
                    yesterday_asymp_no = forecasted_fluxes[Comp.A.value][Vax.no.value].read(day-1)
                    yesterday_asymp_yes = forecasted_fluxes[Comp.A.value][Vax.yes.value].read(day - 1)

                if vax_status == Vax.yes.value:
                    # I was getting lots of nan gradients for epsilon/delta/T_serial
                    # I worried that the end of the forecast was too sensitive to changes here
                    # So I added 1e-6 to T_serial
                    today_asymp = (yesterday_asymp_no +
                                   self.epsilon_samples_constrained[Vax.total.value]*yesterday_asymp_yes) * \
                    self.delta_samples_constrained[Vax.yes.value]*r_t[day] ** (1/(self.T_serial_samples_constrained[Vax.total.value] + 1e-6))
                else:
                    today_asymp = (yesterday_asymp_no +
                                   self.epsilon_samples_constrained[Vax.total.value] * yesterday_asymp_yes) * \
                                   r_t[day] ** (1 / self.T_serial_samples_constrained[Vax.total.value])

                forecasted_fluxes[Comp.A.value][vax_status] = \
                    forecasted_fluxes[Comp.A.value][vax_status].write(day, today_asymp)

                # We treat our initial count as the count at day -1, not day -J as the writeup expresses.
                # This lets us ignore a situation where it is difficult to understand the count at day 0 as it
                #   would be a function of the warmup. It would also require extra warmup for
                if day == 0:
                    prev_count_G[vax_status] = self.init_count_G_samples_constrained[vax_status]
                    prev_count_I[vax_status] = self.init_count_I_samples_constrained[vax_status]
                else:
                    prev_count_G[vax_status] = forecasted_counts[Comp.G.value][vax_status].read(day-1)
                    prev_count_I[vax_status] = forecasted_counts[Comp.I.value][vax_status].read(day-1)

                previously_asymptomatic_tensor, previously_mild_tensor, previously_gen_tensor, previously_icu_tensor = \
                    self._get_prev_tensors(forecasted_fluxes, vax_status, day)

                # Today's A->M = sum of last J day of A * rho * pi
                forecasted_fluxes[Comp.M.value][vax_status] = \
                    forecasted_fluxes[Comp.M.value][vax_status].write(day,
                                                                     tf.reduce_sum(
                                                                         previously_asymptomatic_tensor *
                                                                         self.rho_M_samples_constrained[vax_status] * self.pi_M_samples[vax_status],
                                                                         axis=0)
                                                                     )

                current_G_in = tf.reduce_sum(previously_mild_tensor *
                                               self.rho_G_samples_constrained[vax_status] * self.pi_G_samples[vax_status],
                                               axis=0)
                current_I_in = tf.reduce_sum(previously_gen_tensor *
                                             self.rho_I_samples_constrained[vax_status] * self.pi_I_samples[vax_status],
                                             axis=0)
                current_D_in = tf.reduce_sum(previously_icu_tensor *
                                             self.rho_D_samples_constrained[vax_status] * self.pi_D_samples[vax_status],
                                             axis=0)


                forecasted_fluxes[Comp.G.value][vax_status] = \
                    forecasted_fluxes[Comp.G.value][vax_status].write(day, current_G_in )
                
                # People who recover from G = people who used to be in G * (1-prob moving to I) * I recovery duration
                current_GR = tf.reduce_sum(previously_gen_tensor *
                                              (1-self.rho_I_samples_constrained[vax_status]) *
                                              self.pi_I_bar_samples[vax_status],
                                              axis=0)

                # Make sure GR doesn't go negative. I don't think that first maximum(0, is necessary
                current_GR = tf.math.maximum(0,
                                             tf.math.minimum(current_GR,
                                                             prev_count_G[vax_status]+current_G_in-current_I_in)
                                             )
                    
                
                forecasted_fluxes[Comp.GR.value][vax_status] = \
                    forecasted_fluxes[Comp.GR.value][vax_status].write(day, current_GR)

                forecasted_counts[Comp.G.value][vax_status] = \
                    forecasted_counts[Comp.G.value][vax_status].write(day,tf.math.maximum(0,
                                                                      prev_count_G[vax_status] + \
                                                                      current_G_in - \
                                                                      current_GR - current_I_in))


                forecasted_fluxes[Comp.I.value][vax_status] = \
                    forecasted_fluxes[Comp.I.value][vax_status].write(day, current_I_in)

                current_IR = tf.reduce_sum(previously_icu_tensor *
                                           (1 - self.rho_D_samples_constrained[vax_status]) *
                                           self.pi_D_bar_samples[vax_status],
                                           axis=0)

                current_IR = tf.math.maximum(0,tf.math.minimum(current_IR, prev_count_I[vax_status]+current_I_in-current_D_in))

                forecasted_fluxes[Comp.IR.value][vax_status] = \
                    forecasted_fluxes[Comp.IR.value][vax_status].write(day, current_IR)

                forecasted_counts[Comp.I.value][vax_status] = \
                    forecasted_counts[Comp.I.value][vax_status].write(day,tf.math.maximum(0,
                                                                      prev_count_I[vax_status] + \
                                                                      current_I_in - \
                                                                      current_IR - current_D_in))

                forecasted_fluxes[Comp.D.value][vax_status] = \
                    forecasted_fluxes[Comp.D.value][vax_status].write(day, current_D_in)


        if not debug_disable_prior:
            self._callable_losses.clear()
            self._add_prior_loss()




        # Re-combine vaccinated and unvaxxed for our output
        if return_all:
            result = forecasted_fluxes, forecasted_counts
        else:
            result = {}
            result['G_count'] = forecasted_counts[Comp.G.value][Vax.yes.value].stack() + forecasted_counts[Comp.G.value][Vax.no.value].stack()
            result['G_in'] = forecasted_fluxes[Comp.G.value][Vax.yes.value].stack() + forecasted_fluxes[Comp.G.value][Vax.no.value].stack()
            result['I_count'] = forecasted_counts[Comp.I.value][Vax.yes.value].stack() + forecasted_counts[Comp.I.value][Vax.no.value].stack()
            result['D_in'] = forecasted_fluxes[Comp.D.value][Vax.yes.value].stack() + forecasted_fluxes[Comp.D.value][Vax.no.value].stack()
            result = tf.expand_dims((result['G_count'], result['G_in'], result['I_count'], result['D_in']), axis=0)

        # Tensorflow thinks we didn't use every array, so we gotta mark them as used
        # TODO: did i screw up?
        self._mark_arrays_used(forecasted_fluxes)

        return result

    def _initialize_parameters(self, config,
                               debug_disable_theta=False, fix_variance=False):
        """Helper function to hide the book-keeping behind initializing model parameters

        TODO: Replace with better/random initializations
        """

        self.model_params = {}

        self.unconstrained_T_serial = {}

        self.unconstrained_epsilon = {}
        self.unconstrained_delta = {}

        self.unconstrained_rho_M = {}
        self.unconstrained_lambda_M = {}
        self.unconstrained_nu_M = {}
        self.unconstrained_rho_G = {}
        self.unconstrained_lambda_G = {}
        self.unconstrained_nu_G = {}

        self.unconstrained_rho_I = {}
        self.unconstrained_lambda_I = {}
        self.unconstrained_nu_I = {}
        self.unconstrained_lambda_I_bar = {}
        self.unconstrained_nu_I_bar = {}

        self.unconstrained_rho_D = {}
        self.unconstrained_lambda_D = {}
        self.unconstrained_nu_D = {}
        self.unconstrained_lambda_D_bar = {}
        self.unconstrained_nu_D_bar = {}

        self.unconstrained_warmup_A_params = {}
        self.unconstrained_warmup_M_params = {}
        self.unconstrained_warmup_G_params = {}
        self.unconstrained_warmup_GR_params = {}
        self.unconstrained_init_count_G_params = {}
        self.unconstrained_warmup_I_params = {}
        self.unconstrained_warmup_IR_params = {}
        self.unconstrained_init_count_I_params = {}

        self.previously_asymptomatic = {}
        self.previously_mild = {}
        self.previously_gen = {}
        self.previously_icu = {}

        train_theta = not debug_disable_theta
        train_variance = not (debug_disable_theta or fix_variance)

        # T_serial, Delta and epsilon dont vary by vaccination status
        self.unconstrained_T_serial = {}
        self.unconstrained_T_serial['loc'] = \
            tf.Variable(config.T_serial.mean_transform.inverse(config.T_serial.value['loc']), dtype=tf.float32,
                        name=f'T_serial_A_loc', trainable=train_theta)
        self.unconstrained_T_serial['scale'] = \
            tf.Variable(config.T_serial.scale_transform.inverse(config.T_serial.value['scale']), dtype=tf.float32,
                        name=f'T_serial_A_scale', trainable=train_variance)

        self.unconstrained_epsilon = {}
        self.unconstrained_epsilon['loc'] = \
            tf.Variable(config.epsilon.mean_transform.inverse(config.epsilon.value['loc']), dtype=tf.float32,
                        name=f'epsilon_A_loc', trainable=train_theta)
        self.unconstrained_epsilon['scale'] = \
            tf.Variable(config.epsilon.scale_transform.inverse(config.epsilon.value['scale']), dtype=tf.float32,
                        name=f'epsilon_A_scale', trainable=train_variance)

        self.unconstrained_delta = {}
        self.unconstrained_delta['loc'] = \
            tf.Variable(config.delta.mean_transform.inverse(config.delta.value['loc']), dtype=tf.float32,
                        name=f'delta_A_loc', trainable=train_theta)
        self.unconstrained_delta['scale'] = \
            tf.Variable(config.delta.scale_transform.inverse(config.delta.value['scale']), dtype=tf.float32,
                        name=f'delta_A_scale', trainable=train_variance)

        for vax_status in [status.value for status in self.vax_statuses]:

            self.unconstrained_rho_M[vax_status] = {}
            self.unconstrained_rho_M[vax_status]['loc'] = \
                tf.Variable(config.rho_M.mean_transform.inverse(config.rho_M.value[vax_status]['loc']), dtype=tf.float32,
                            name=f'rho_M_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_rho_M[vax_status]['scale'] = \
                tf.Variable(config.rho_M.scale_transform.inverse(config.rho_M.value[vax_status]['scale']), dtype=tf.float32,
                            name=f'rho_M_scale_{vax_status}', trainable=train_variance)

            self.unconstrained_rho_G[vax_status] = {}
            self.unconstrained_rho_G[vax_status]['loc'] = \
                tf.Variable(config.rho_G.mean_transform.inverse(config.rho_G.value[vax_status]['loc']), dtype=tf.float32,
                            name=f'rho_G_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_rho_G[vax_status]['scale'] = \
                tf.Variable(config.rho_G.scale_transform.inverse(config.rho_G.value[vax_status]['scale']), dtype=tf.float32,
                            name=f'rho_G_scale_{vax_status}', trainable=train_variance)

            self.unconstrained_rho_I[vax_status] = {}
            self.unconstrained_rho_I[vax_status]['loc'] = \
                tf.Variable(config.rho_I.mean_transform.inverse(config.rho_I.value[vax_status]['loc']), dtype=tf.float32,
                            name=f'rho_I_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_rho_I[vax_status]['scale'] = \
                tf.Variable(config.rho_I.scale_transform.inverse(config.rho_I.value[vax_status]['scale']), dtype=tf.float32,
                            name=f'rho_I_scale_{vax_status}', trainable=train_variance)

            self.unconstrained_rho_D[vax_status] = {}
            self.unconstrained_rho_D[vax_status]['loc'] = \
                tf.Variable(config.rho_D.mean_transform.inverse(config.rho_D.value[vax_status]['loc']), dtype=tf.float32,
                            name=f'rho_D_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_rho_D[vax_status]['scale'] = \
                tf.Variable(config.rho_D.scale_transform.inverse(config.rho_D.value[vax_status]['scale']), dtype=tf.float32,
                            name=f'rho_D_scale_{vax_status}', trainable=train_variance)

            self.unconstrained_lambda_M[vax_status] = {}
            self.unconstrained_lambda_M[vax_status]['loc'] = \
                tf.Variable(config.lambda_M.mean_transform.inverse(config.lambda_M.value[vax_status]['loc']), dtype=tf.float32,
                            name=f'lambda_M_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_lambda_M[vax_status]['scale'] = \
                tf.Variable(config.lambda_M.scale_transform.inverse(config.lambda_M.value[vax_status]['scale']), dtype=tf.float32,
                            name=f'lambda_M_scale_{vax_status}', trainable=train_variance)

            self.unconstrained_lambda_G[vax_status] = {}
            self.unconstrained_lambda_G[vax_status]['loc'] = \
                tf.Variable(config.lambda_G.mean_transform.inverse(config.lambda_G.value[vax_status]['loc']), dtype=tf.float32,
                            name=f'lambda_G_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_lambda_G[vax_status]['scale'] = \
                tf.Variable(config.lambda_G.scale_transform.inverse(config.lambda_G.value[vax_status]['scale']), dtype=tf.float32,
                            name=f'lambda_G_scale_{vax_status}', trainable=train_variance)

            self.unconstrained_lambda_I[vax_status] = {}
            self.unconstrained_lambda_I[vax_status]['loc'] = \
                tf.Variable(config.lambda_I.mean_transform.inverse(config.lambda_I.value[vax_status]['loc']), dtype=tf.float32,
                            name=f'lambda_I_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_lambda_I[vax_status]['scale'] = \
                tf.Variable(config.lambda_I.scale_transform.inverse(config.lambda_I.value[vax_status]['scale']), dtype=tf.float32,
                            name=f'lambda_I_scale_{vax_status}', trainable=train_variance)

            self.unconstrained_lambda_I_bar[vax_status] = {}
            self.unconstrained_lambda_I_bar[vax_status]['loc'] = \
                tf.Variable(config.lambda_I_bar.mean_transform.inverse(config.lambda_I_bar.value[vax_status]['loc']), dtype=tf.float32,
                            name=f'lambda_I_bar_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_lambda_I_bar[vax_status]['scale'] = \
                tf.Variable(config.lambda_I_bar.scale_transform.inverse(config.lambda_I_bar.value[vax_status]['scale']), dtype=tf.float32,
                            name=f'lambda_I_bar_scale_{vax_status}', trainable=train_variance)

            self.unconstrained_lambda_D[vax_status] = {}
            self.unconstrained_lambda_D[vax_status]['loc'] = \
                tf.Variable(config.lambda_D.mean_transform.inverse(config.lambda_D.value[vax_status]['loc']), dtype=tf.float32,
                            name=f'lambda_D_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_lambda_D[vax_status]['scale'] = \
                tf.Variable(config.lambda_D.scale_transform.inverse(config.lambda_D.value[vax_status]['scale']), dtype=tf.float32,
                            name=f'lambda_D_scale_{vax_status}', trainable=train_variance)

            self.unconstrained_lambda_D_bar[vax_status] = {}
            self.unconstrained_lambda_D_bar[vax_status]['loc'] = \
                tf.Variable(config.lambda_D_bar.mean_transform.inverse(config.lambda_D_bar.value[vax_status]['loc']), dtype=tf.float32,
                            name=f'lambda_D_bar_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_lambda_D_bar[vax_status]['scale'] = \
                tf.Variable(config.lambda_D_bar.scale_transform.inverse(config.lambda_D_bar.value[vax_status]['scale']), dtype=tf.float32,
                            name=f'lambda_D_bar_scale_{vax_status}', trainable=train_variance)

            self.unconstrained_nu_M[vax_status] = {}
            self.unconstrained_nu_M[vax_status]['loc'] = \
                tf.Variable(config.nu_M.mean_transform.inverse(config.nu_M.value[vax_status]['loc']), dtype=tf.float32,
                            name=f'nu_M_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_nu_M[vax_status]['scale'] = \
                tf.Variable(config.nu_M.scale_transform.inverse(config.nu_M.value[vax_status]['scale']), dtype=tf.float32,
                            name=f'nu_M_scale_{vax_status}', trainable=train_variance)

            self.unconstrained_nu_G[vax_status] = {}
            self.unconstrained_nu_G[vax_status]['loc'] = \
                tf.Variable(config.nu_G.mean_transform.inverse(config.nu_G.value[vax_status]['loc']), dtype=tf.float32,
                            name=f'nu_G_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_nu_G[vax_status]['scale'] = \
                tf.Variable(config.nu_G.scale_transform.inverse(config.nu_G.value[vax_status]['scale']), dtype=tf.float32,
                            name=f'nu_G_scale_{vax_status}', trainable=train_variance)

            self.unconstrained_nu_I[vax_status] = {}
            self.unconstrained_nu_I[vax_status]['loc'] = \
                tf.Variable(config.nu_I.mean_transform.inverse(config.nu_I.value[vax_status]['loc']), dtype=tf.float32,
                            name=f'nu_I_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_nu_I[vax_status]['scale'] = \
                tf.Variable(config.nu_I.scale_transform.inverse(config.nu_I.value[vax_status]['scale']), dtype=tf.float32,
                            name=f'nu_I_scale_{vax_status}', trainable=train_variance)

            self.unconstrained_nu_I_bar[vax_status] = {}
            self.unconstrained_nu_I_bar[vax_status]['loc'] = \
                tf.Variable(config.nu_I_bar.mean_transform.inverse(config.nu_I_bar.value[vax_status]['loc']), dtype=tf.float32,
                            name=f'nu_I_bar_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_nu_I_bar[vax_status]['scale'] = \
                tf.Variable(config.nu_I_bar.scale_transform.inverse(config.nu_I_bar.value[vax_status]['scale']), dtype=tf.float32,
                            name=f'nu_I_bar_scale_{vax_status}', trainable=train_variance)

            self.unconstrained_nu_D[vax_status] = {}
            self.unconstrained_nu_D[vax_status]['loc'] = \
                tf.Variable(config.nu_D.mean_transform.inverse(config.nu_D.value[vax_status]['loc']), dtype=tf.float32,
                            name=f'nu_D_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_nu_D[vax_status]['scale'] = \
                tf.Variable(config.nu_D.scale_transform.inverse(config.nu_D.value[vax_status]['scale']), dtype=tf.float32,
                            name=f'nu_D_scale_{vax_status}', trainable=train_variance)

            self.unconstrained_nu_D_bar[vax_status] = {}
            self.unconstrained_nu_D_bar[vax_status]['loc'] = \
                tf.Variable(config.nu_D_bar.mean_transform.inverse(config.nu_D_bar.value[vax_status]['loc']), dtype=tf.float32,
                            name=f'nu_D_bar_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_nu_D_bar[vax_status]['scale'] = \
                tf.Variable(config.nu_D_bar.scale_transform.inverse(config.nu_D_bar.value[vax_status]['scale']), dtype=tf.float32,
                            name=f'nu_D_bar_scale_{vax_status}', trainable=train_variance)

            self.unconstrained_warmup_A_params[vax_status] = {}
            self.unconstrained_warmup_M_params[vax_status] = {}
            self.unconstrained_warmup_G_params[vax_status] = {}
            self.unconstrained_warmup_GR_params[vax_status] = {}
            self.unconstrained_init_count_G_params[vax_status] = {}
            self.unconstrained_warmup_I_params[vax_status] = {}
            self.unconstrained_warmup_IR_params[vax_status] = {}
            self.unconstrained_init_count_I_params[vax_status] = {}

            self.unconstrained_warmup_A_params[vax_status]['slope'] = \
                tf.Variable(tf.cast(config.warmup_A.value[vax_status]['slope'],
                                    dtype=tf.float32), dtype=tf.float32,
                            name=f'warmup_A_slope_{vax_status}')
            self.unconstrained_warmup_A_params[vax_status]['intercept'] = \
                tf.Variable(tf.cast(config.warmup_A.mean_transform.inverse(config.warmup_A.value[vax_status]['intercept']),
                                    dtype=tf.float32), dtype=tf.float32,
                            name=f'warmup_A_intercept_{vax_status}')
            self.unconstrained_warmup_A_params[vax_status]['scale'] = \
                tf.Variable(tf.cast(config.warmup_A.scale_transform.inverse(config.warmup_A.value[vax_status]['scale']),
                                    dtype=tf.float32), dtype=tf.float32,
                            name=f'warmup_A_scale_{vax_status}', trainable=train_variance)

            self.unconstrained_warmup_M_params[vax_status]['slope'] = \
                tf.Variable(tf.cast(config.warmup_M.value[vax_status]['slope'],
                                    dtype=tf.float32), dtype=tf.float32,
                            name=f'warmup_M_slope_{vax_status}')
            self.unconstrained_warmup_M_params[vax_status]['intercept'] = \
                tf.Variable(tf.cast(config.warmup_M.mean_transform.inverse(config.warmup_M.value[vax_status]['intercept']),
                                    dtype=tf.float32), dtype=tf.float32,
                            name=f'warmup_M_intercept_{vax_status}')
            self.unconstrained_warmup_M_params[vax_status]['scale'] = \
                tf.Variable(tf.cast(config.warmup_M.scale_transform.inverse(config.warmup_M.value[vax_status]['scale']),
                                    dtype=tf.float32), dtype=tf.float32,
                            name=f'warmup_M_scale_{vax_status}', trainable=train_variance)

            self.unconstrained_warmup_G_params[vax_status]['slope'] = \
                tf.Variable(tf.cast(config.warmup_G.value[vax_status]['slope'],
                                    dtype=tf.float32), dtype=tf.float32,
                            name=f'warmup_G_slope_{vax_status}')
            self.unconstrained_warmup_G_params[vax_status]['intercept'] = \
                tf.Variable(tf.cast(config.warmup_G.mean_transform.inverse(config.warmup_G.value[vax_status]['intercept']),
                                    dtype=tf.float32), dtype=tf.float32,
                            name=f'warmup_G_intercept_{vax_status}')
            self.unconstrained_warmup_G_params[vax_status]['scale'] = \
                tf.Variable(tf.cast(config.warmup_G.scale_transform.inverse(config.warmup_G.value[vax_status]['scale']),
                                    dtype=tf.float32), dtype=tf.float32,
                            name=f'warmup_G_scale_{vax_status}', trainable=train_variance)

            self.unconstrained_warmup_GR_params[vax_status]['slope'] = \
                tf.Variable(tf.cast(config.warmup_GR.value[vax_status]['slope'],
                                    dtype=tf.float32), dtype=tf.float32,
                            name=f'warmup_GR_slope_{vax_status}')
            self.unconstrained_warmup_GR_params[vax_status]['intercept'] = \
                tf.Variable(tf.cast(config.warmup_GR.mean_transform.inverse(config.warmup_GR.value[vax_status]['intercept']),
                                    dtype=tf.float32), dtype=tf.float32,
                            name=f'warmup_GR_intercept_{vax_status}')
            self.unconstrained_warmup_GR_params[vax_status]['scale'] = \
                tf.Variable(tf.cast(config.warmup_GR.scale_transform.inverse(config.warmup_GR.value[vax_status]['scale']),
                                    dtype=tf.float32), dtype=tf.float32,
                            name=f'warmup_GR_scale_{vax_status}', trainable=train_variance)

            self.unconstrained_warmup_I_params[vax_status]['slope'] = \
                tf.Variable(tf.cast(config.warmup_I.value[vax_status]['slope'],
                                    dtype=tf.float32), dtype=tf.float32,
                            name=f'warmup_I_slope_{vax_status}')
            self.unconstrained_warmup_I_params[vax_status]['intercept'] = \
                tf.Variable(tf.cast(config.warmup_I.mean_transform.inverse(config.warmup_I.value[vax_status]['intercept']),
                                    dtype=tf.float32), dtype=tf.float32,
                            name=f'warmup_I_intercept_{vax_status}')
            self.unconstrained_warmup_I_params[vax_status]['scale'] = \
                tf.Variable(tf.cast(config.warmup_I.scale_transform.inverse(config.warmup_I.value[vax_status]['scale']),
                                    dtype=tf.float32), dtype=tf.float32,
                            name=f'warmup_I_scale_{vax_status}', trainable=train_variance)

            self.unconstrained_warmup_IR_params[vax_status]['slope'] = \
                tf.Variable(tf.cast(config.warmup_IR.value[vax_status]['slope'],
                                    dtype=tf.float32), dtype=tf.float32,
                            name=f'warmup_IR_slope_{vax_status}')
            self.unconstrained_warmup_IR_params[vax_status]['intercept'] = \
                tf.Variable(tf.cast(config.warmup_IR.mean_transform.inverse(config.warmup_IR.value[vax_status]['intercept']),
                                    dtype=tf.float32), dtype=tf.float32,
                            name=f'warmup_IR_intercept_{vax_status}')
            self.unconstrained_warmup_IR_params[vax_status]['scale'] = \
                tf.Variable(tf.cast(config.warmup_IR.mean_transform.inverse(config.warmup_IR.value[vax_status]['scale']),
                                    dtype=tf.float32), dtype=tf.float32,
                            name=f'warmup_IR_scale_{vax_status}', trainable=train_variance)
                
            self.unconstrained_init_count_G_params[vax_status]['loc'] = \
                tf.Variable(tf.cast(config.init_count_G.mean_transform.inverse(config.init_count_G.value[vax_status]['loc']),
                                    dtype=tf.float32), dtype=tf.float32,
                            name=f'init_count_G_loc_{vax_status}')
            self.unconstrained_init_count_G_params[vax_status]['scale'] = \
                tf.Variable(tf.cast(config.init_count_G.scale_transform.inverse(config.init_count_G.value[vax_status]['scale']),
                                    dtype=tf.float32), dtype=tf.float32,
                            name=f'init_count_G_scale_{vax_status}', trainable=train_variance)

            self.unconstrained_init_count_I_params[vax_status]['loc'] = \
                tf.Variable(tf.cast(config.init_count_I.mean_transform.inverse(config.init_count_I.value[vax_status]['loc']),
                                    dtype=tf.float32), dtype=tf.float32,
                            name=f'init_count_I_loc_{vax_status}')
            self.unconstrained_init_count_I_params[vax_status]['scale'] = \
                tf.Variable(tf.cast(config.init_count_I.scale_transform.inverse(config.init_count_I.value[vax_status]['scale']),
                                    dtype=tf.float32), dtype=tf.float32,
                            name=f'init_count_I_scale_{vax_status}', trainable=train_variance)

            self.previously_asymptomatic[vax_status] = tf.TensorArray(tf.float32, size=self.transition_window,
                                                                      clear_after_read=False, name=f'prev_asymp')
            self.previously_mild[vax_status] = tf.TensorArray(tf.float32, size=self.transition_window,
                                                                      clear_after_read=False, name=f'prev_mild')
            self.previously_gen[vax_status] = tf.TensorArray(tf.float32, size=self.transition_window,
                                                              clear_after_read=False, name=f'prev_gen')
            self.previously_icu[vax_status] = tf.TensorArray(tf.float32, size=self.transition_window,
                                                              clear_after_read=False, name=f'prev_icu')

        return

    def _initialize_priors(self, config):
        """Helper function to hide the book-keeping behind initializing model priors"""

        self.prior_distros = {}
        for enum_c in self.compartments:
            compartment = enum_c.value
            self.prior_distros[compartment] = {}
            self.prior_distros[compartment][Vax.total.value] = {}
            for vax_status in [status.value for status in self.vax_statuses]:
                self.prior_distros[compartment][vax_status] = {}

        # T_serial, Epsilon and delta are speical
        # T serial must be positive
        self.prior_distros[Comp.A.value][Vax.total.value]['T_serial'] = tfp.distributions.TransformedDistribution(
            tfp.distributions.TruncatedNormal(
                config.T_serial.prior['loc'],
                config.T_serial.prior['scale'],
                0, np.inf),
            bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
        )

        self.prior_distros[Comp.A.value][Vax.total.value]['epsilon'] = tfp.distributions.TransformedDistribution(
            tfp.distributions.Beta(
                config.epsilon.prior['a'],
                config.epsilon.prior['b']),
            bijector=tfp.bijectors.Invert(tfp.bijectors.Sigmoid())
        )

        self.prior_distros[Comp.A.value][Vax.yes.value]['delta'] = tfp.distributions.TransformedDistribution(
            tfp.distributions.Beta(
                config.delta.prior['a'],
                config.delta.prior['b']),
            bijector=tfp.bijectors.Invert(tfp.bijectors.Sigmoid())
        )

        # create prior distributions
        for vax_status in [status.value for status in self.vax_statuses]:

            self.prior_distros[Comp.M.value][vax_status]['rho_M'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.Beta(
                    config.rho_M.prior[vax_status]['a'],
                    config.rho_M.prior[vax_status]['b']),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Sigmoid())
            )

            self.prior_distros[Comp.G.value][vax_status]['rho_G'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.Beta(
                    config.rho_G.prior[vax_status]['a'],
                    config.rho_G.prior[vax_status]['b']),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Sigmoid())
            )
            self.prior_distros[Comp.I.value][vax_status]['rho_I'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.Beta(
                    config.rho_I.prior[vax_status]['a'],
                    config.rho_I.prior[vax_status]['b']),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Sigmoid())
            )

            self.prior_distros[Comp.D.value][vax_status]['rho_D'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.Beta(
                    config.rho_D.prior[vax_status]['a'],
                    config.rho_D.prior[vax_status]['b']),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Sigmoid())
            )

            #  must be positive
            self.prior_distros[Comp.M.value][vax_status]['lambda_M'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.TruncatedNormal(
                    config.lambda_M.prior[vax_status]['loc'],
                    config.lambda_M.prior[vax_status]['scale'],
                    0, np.inf),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
            )

            self.prior_distros[Comp.G.value][vax_status]['lambda_G'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.TruncatedNormal(
                    config.lambda_G.prior[vax_status]['loc'],
                    config.lambda_G.prior[vax_status]['scale'],
                    0, np.inf),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
            )
            
            self.prior_distros[Comp.I.value][vax_status]['lambda_I'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.TruncatedNormal(
                    config.lambda_I.prior[vax_status]['loc'],
                    config.lambda_I.prior[vax_status]['scale'],
                    0, np.inf),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
            )

            self.prior_distros[Comp.IR.value][vax_status]['lambda_I_bar'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.TruncatedNormal(
                    config.lambda_I_bar.prior[vax_status]['loc'],
                    config.lambda_I_bar.prior[vax_status]['scale'],
                    0, np.inf),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
            )
            
            self.prior_distros[Comp.D.value][vax_status]['lambda_D'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.TruncatedNormal(
                    config.lambda_D.prior[vax_status]['loc'],
                    config.lambda_D.prior[vax_status]['scale'],
                    0, np.inf),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
            )

            self.prior_distros[Comp.D.value][vax_status]['lambda_D_bar'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.TruncatedNormal(
                    config.lambda_D_bar.prior[vax_status]['loc'],
                    config.lambda_D_bar.prior[vax_status]['scale'],
                    0, np.inf),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
            )

            self.prior_distros[Comp.M.value][vax_status]['nu_M'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.TruncatedNormal(
                    config.nu_M.prior[vax_status]['loc'],
                    config.nu_M.prior[vax_status]['scale'],
                    0, np.inf),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
            )
            
            self.prior_distros[Comp.G.value][vax_status]['nu_G'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.TruncatedNormal(
                    config.nu_G.prior[vax_status]['loc'],
                    config.nu_G.prior[vax_status]['scale'],
                    0, np.inf),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
            )
            self.prior_distros[Comp.I.value][vax_status]['nu_I'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.TruncatedNormal(
                    config.nu_I.prior[vax_status]['loc'],
                    config.nu_I.prior[vax_status]['scale'],
                    0, np.inf),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
            )
            self.prior_distros[Comp.IR.value][vax_status]['nu_I_bar'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.TruncatedNormal(
                    config.nu_I_bar.prior[vax_status]['loc'],
                    config.nu_I_bar.prior[vax_status]['scale'],
                    0, np.inf),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
            )
            self.prior_distros[Comp.D.value][vax_status]['nu_D'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.TruncatedNormal(
                    config.nu_D.prior[vax_status]['loc'],
                    config.nu_D.prior[vax_status]['scale'],
                    0, np.inf),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
            )
            self.prior_distros[Comp.D.value][vax_status]['nu_D_bar'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.TruncatedNormal(
                    config.nu_D_bar.prior[vax_status]['loc'],
                    config.nu_D_bar.prior[vax_status]['scale'],
                    0, np.inf),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
            )

            self.prior_distros[Comp.A.value][vax_status]['warmup_A'] = []
            self.prior_distros[Comp.M.value][vax_status]['warmup_M'] = []
            self.prior_distros[Comp.G.value][vax_status]['warmup_G'] = []
            self.prior_distros[Comp.GR.value][vax_status]['warmup_GR'] = []
            self.prior_distros[Comp.I.value][vax_status]['warmup_I'] = []
            self.prior_distros[Comp.IR.value][vax_status]['warmup_IR'] = []
            for day in range(self.transition_window):
                self.prior_distros[Comp.A.value][vax_status]['warmup_A'].append(
                    tfp.distributions.TransformedDistribution(
                        tfp.distributions.TruncatedNormal(
                            tf.cast(config.warmup_A.prior[vax_status]['intercept'],dtype=tf.float32) +
                            tf.cast(day * config.warmup_A.prior[vax_status]['slope'],dtype=tf.float32),
                            tf.cast(config.warmup_A.prior[vax_status]['scale'],dtype=tf.float32),
                            0, tf.float32.max),
                        bijector=tfp.bijectors.Invert(tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]))
                    )
                )
                self.prior_distros[Comp.M.value][vax_status]['warmup_M'].append(
                    tfp.distributions.TransformedDistribution(
                        tfp.distributions.TruncatedNormal(
                            tf.cast(config.warmup_M.prior[vax_status]['intercept'], dtype=tf.float32) +
                            tf.cast(day * config.warmup_M.prior[vax_status]['slope'], dtype=tf.float32),
                            tf.cast(config.warmup_M.prior[vax_status]['scale'], dtype=tf.float32),
                            0, tf.float32.max),
                        bijector=tfp.bijectors.Invert(tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]))
                    )
                )
                self.prior_distros[Comp.G.value][vax_status]['warmup_G'].append(
                    tfp.distributions.TransformedDistribution(
                        tfp.distributions.TruncatedNormal(
                            tf.cast(config.warmup_G.prior[vax_status]['intercept'], dtype=tf.float32) +
                            tf.cast(day * config.warmup_G.prior[vax_status]['slope'], dtype=tf.float32),
                            tf.cast(config.warmup_G.prior[vax_status]['scale'], dtype=tf.float32),
                            0, tf.float32.max),
                        bijector=tfp.bijectors.Invert(
                            tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]))
                    )
                )
                self.prior_distros[Comp.GR.value][vax_status]['warmup_GR'].append(
                    tfp.distributions.TransformedDistribution(
                        tfp.distributions.TruncatedNormal(
                            tf.cast(config.warmup_GR.prior[vax_status]['intercept'], dtype=tf.float32) +
                            tf.cast(day * config.warmup_GR.prior[vax_status]['slope'], dtype=tf.float32),
                            tf.cast(config.warmup_GR.prior[vax_status]['scale'], dtype=tf.float32),
                            0, tf.float32.max),
                        bijector=tfp.bijectors.Invert(
                            tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]))
                    )
                )
                self.prior_distros[Comp.I.value][vax_status]['warmup_I'].append(
                    tfp.distributions.TransformedDistribution(
                        tfp.distributions.TruncatedNormal(
                            tf.cast(config.warmup_I.prior[vax_status]['intercept'], dtype=tf.float32) +
                            tf.cast(day * config.warmup_I.prior[vax_status]['slope'], dtype=tf.float32),
                            tf.cast(config.warmup_I.prior[vax_status]['scale'], dtype=tf.float32),
                            0, tf.float32.max),
                        bijector=tfp.bijectors.Invert(
                            tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]))
                    )
                )
                self.prior_distros[Comp.IR.value][vax_status]['warmup_IR'].append(
                    tfp.distributions.TransformedDistribution(
                        tfp.distributions.TruncatedNormal(
                            tf.cast(config.warmup_IR.prior[vax_status]['intercept'], dtype=tf.float32) +
                            tf.cast(day * config.warmup_IR.prior[vax_status]['slope'], dtype=tf.float32),
                            tf.cast(config.warmup_IR.prior[vax_status]['scale'], dtype=tf.float32),
                            0, tf.float32.max),
                        bijector=tfp.bijectors.Invert(
                            tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]))
                    )
                )
            self.prior_distros[Comp.G.value][vax_status]['init_count_G'] = (
                tfp.distributions.TransformedDistribution(
                    tfp.distributions.TruncatedNormal(
                        tf.cast(config.init_count_G.prior[vax_status]['loc'], dtype=tf.float32),
                        tf.cast(config.init_count_G.prior[vax_status]['scale'], dtype=tf.float32),
                        0, tf.float32.max),
                    bijector=tfp.bijectors.Invert(
                        tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]))
                )
            )
            self.prior_distros[Comp.I.value][vax_status]['init_count_I'] = (
                tfp.distributions.TransformedDistribution(
                    tfp.distributions.TruncatedNormal(
                        tf.cast(config.init_count_I.prior[vax_status]['loc'], dtype=tf.float32),
                        tf.cast(config.init_count_I.prior[vax_status]['scale'], dtype=tf.float32),
                        0, tf.float32.max),
                    bijector=tfp.bijectors.Invert(
                        tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]))
                )
            )

        return

    def  _constrain_parameters(self):
        """Helper function to make sure all of our posterior variance parameters are positive

        Note: We don't constrain the means here, the means are still real numbers
        """

        self.T_serial_params = {}
        self.epsilon_params = {}
        self.delta_params = {}
        self.rho_M_params = {}
        self.lambda_M_params = {}
        self.nu_M_params = {}
        self.rho_G_params = {}
        self.lambda_G_params = {}
        self.nu_G_params = {}
        self.rho_I_params = {}
        self.lambda_I_params = {}
        self.nu_I_params = {}
        self.lambda_I_bar_params = {}
        self.nu_I_bar_params = {}
        self.rho_D_params = {}
        self.lambda_D_params = {}
        self.lambda_D_bar_params = {}
        self.nu_D_params = {}
        self.nu_D_bar_params = {}
        self.warmup_A_params = {}
        self.warmup_M_params = {}
        self.warmup_G_params = {}
        self.warmup_GR_params = {}
        self.init_count_G_params = {}
        self.warmup_I_params = {}
        self.warmup_IR_params = {}
        self.init_count_I_params = {}

        self.T_serial_params[Vax.total.value] = {}
        self.epsilon_params[Vax.total.value] = {}
        self.delta_params[Vax.yes.value] = {}

        self.T_serial_params[Vax.total.value]['loc'] = self.unconstrained_T_serial['loc']
        self.T_serial_params[Vax.total.value]['scale'] = self.scale_transform.forward(self.unconstrained_T_serial['scale'])

        self.epsilon_params[Vax.total.value]['loc'] = self.unconstrained_epsilon['loc']
        self.epsilon_params[Vax.total.value]['scale'] = self.scale_transform.forward(self.unconstrained_epsilon['scale'])

        self.delta_params[Vax.yes.value]['loc'] = self.unconstrained_delta['loc']
        self.delta_params[Vax.yes.value]['scale'] = self.scale_transform.forward(self.unconstrained_delta['scale'])

        for vax_status in [status.value for status in self.vax_statuses]:

            self.rho_M_params[vax_status] = {}
            self.lambda_M_params[vax_status] = {}
            self.nu_M_params[vax_status] = {}
            self.rho_G_params[vax_status] = {}
            self.lambda_G_params[vax_status] = {}
            self.nu_G_params[vax_status] = {}
            self.rho_I_params[vax_status] = {}
            self.lambda_I_params[vax_status] = {}
            self.nu_I_params[vax_status] = {}
            self.lambda_I_bar_params[vax_status] = {}
            self.nu_I_bar_params[vax_status] = {}
            self.rho_D_params[vax_status] = {}
            self.lambda_D_params[vax_status] = {}
            self.nu_D_params[vax_status] = {}
            self.lambda_D_bar_params[vax_status] = {}
            self.nu_D_bar_params[vax_status] = {}
            self.warmup_A_params[vax_status] = {}
            self.warmup_M_params[vax_status] = {}
            self.warmup_G_params[vax_status] = {}
            self.warmup_GR_params[vax_status] = {}
            self.init_count_G_params[vax_status] = {}
            self.warmup_I_params[vax_status] = {}
            self.warmup_IR_params[vax_status] = {}
            self.init_count_I_params[vax_status] = {}


            self.rho_M_params[vax_status]['loc'] = self.unconstrained_rho_M[vax_status]['loc']
            self.rho_M_params[vax_status]['scale'] = self.scale_transform.forward(self.unconstrained_rho_M[vax_status]['scale'])

            self.lambda_M_params[vax_status]['loc'] = self.unconstrained_lambda_M[vax_status]['loc']
            self.lambda_M_params[vax_status]['scale'] = self.scale_transform.forward(self.unconstrained_lambda_M[vax_status]['scale'])

            self.nu_M_params[vax_status]['loc'] = self.unconstrained_nu_M[vax_status]['loc']
            self.nu_M_params[vax_status]['scale'] = self.scale_transform.forward(self.unconstrained_nu_M[vax_status]['scale'])

            self.rho_G_params[vax_status]['loc'] = self.unconstrained_rho_G[vax_status]['loc']
            self.rho_G_params[vax_status]['scale'] = self.scale_transform.forward(self.unconstrained_rho_G[vax_status]['scale'])

            self.lambda_G_params[vax_status]['loc'] = self.unconstrained_lambda_G[vax_status]['loc']
            self.lambda_G_params[vax_status]['scale'] = self.scale_transform.forward(self.unconstrained_lambda_G[vax_status]['scale'])

            self.nu_G_params[vax_status]['loc'] = self.unconstrained_nu_G[vax_status]['loc']
            self.nu_G_params[vax_status]['scale'] = self.scale_transform.forward(self.unconstrained_nu_G[vax_status]['scale'])

            self.rho_I_params[vax_status]['loc'] = self.unconstrained_rho_I[vax_status]['loc']
            self.rho_I_params[vax_status]['scale'] = self.scale_transform.forward(self.unconstrained_rho_I[vax_status]['scale'])

            self.lambda_I_params[vax_status]['loc'] = self.unconstrained_lambda_I[vax_status]['loc']
            self.lambda_I_params[vax_status]['scale'] = self.scale_transform.forward(
                self.unconstrained_lambda_I[vax_status]['scale'])

            self.lambda_I_bar_params[vax_status]['loc'] = self.unconstrained_lambda_I_bar[vax_status]['loc']
            self.lambda_I_bar_params[vax_status]['scale'] = self.scale_transform.forward(
                self.unconstrained_lambda_I_bar[vax_status]['scale'])

            self.nu_I_params[vax_status]['loc'] = self.unconstrained_nu_I[vax_status]['loc']
            self.nu_I_params[vax_status]['scale'] = self.scale_transform.forward(self.unconstrained_nu_I[vax_status]['scale'])

            self.nu_I_bar_params[vax_status]['loc'] = self.unconstrained_nu_I_bar[vax_status]['loc']
            self.nu_I_bar_params[vax_status]['scale'] = self.scale_transform.forward(self.unconstrained_nu_I_bar[vax_status]['scale'])

            self.rho_D_params[vax_status]['loc'] = self.unconstrained_rho_D[vax_status]['loc']
            self.rho_D_params[vax_status]['scale'] = self.scale_transform.forward(self.unconstrained_rho_D[vax_status]['scale'])

            self.lambda_D_params[vax_status]['loc'] = self.unconstrained_lambda_D[vax_status]['loc']
            self.lambda_D_params[vax_status]['scale'] = self.scale_transform.forward(
                self.unconstrained_lambda_D[vax_status]['scale'])

            self.lambda_D_bar_params[vax_status]['loc'] = self.unconstrained_lambda_D_bar[vax_status]['loc']
            self.lambda_D_bar_params[vax_status]['scale'] = self.scale_transform.forward(
                self.unconstrained_lambda_D_bar[vax_status]['scale'])

            self.nu_D_params[vax_status]['loc'] = self.unconstrained_nu_D[vax_status]['loc']
            self.nu_D_params[vax_status]['scale'] = self.scale_transform.forward(self.unconstrained_nu_D[vax_status]['scale'])

            self.nu_D_bar_params[vax_status]['loc'] = self.unconstrained_nu_D_bar[vax_status]['loc']
            self.nu_D_bar_params[vax_status]['scale'] = self.scale_transform.forward(self.unconstrained_nu_D_bar[vax_status]['scale'])

            self.warmup_A_params[vax_status]['slope'] = \
                self.unconstrained_warmup_A_params[vax_status]['slope']
            self.warmup_A_params[vax_status]['intercept'] = \
                self.unconstrained_warmup_A_params[vax_status]['intercept']
            self.warmup_A_params[vax_status]['scale'] = \
                self.scale_transform.forward(self.unconstrained_warmup_A_params[vax_status]['scale'])

            self.warmup_M_params[vax_status]['slope'] = \
                self.unconstrained_warmup_M_params[vax_status]['slope']
            self.warmup_M_params[vax_status]['intercept'] = \
                self.unconstrained_warmup_M_params[vax_status]['intercept']
            self.warmup_M_params[vax_status]['scale'] = \
                self.scale_transform.forward(self.unconstrained_warmup_M_params[vax_status]['scale'])

            self.warmup_G_params[vax_status]['slope'] = \
                self.unconstrained_warmup_G_params[vax_status]['slope']
            self.warmup_G_params[vax_status]['intercept'] = \
                self.unconstrained_warmup_G_params[vax_status]['intercept']
            self.warmup_G_params[vax_status]['scale'] = \
                self.scale_transform.forward(self.unconstrained_warmup_G_params[vax_status]['scale'])

            self.warmup_GR_params[vax_status]['slope'] = \
                self.unconstrained_warmup_GR_params[vax_status]['slope']
            self.warmup_GR_params[vax_status]['intercept'] = \
                self.unconstrained_warmup_GR_params[vax_status]['intercept']
            self.warmup_GR_params[vax_status]['scale'] = \
                self.scale_transform.forward(self.unconstrained_warmup_GR_params[vax_status]['scale'])

            self.warmup_I_params[vax_status]['slope'] = \
                self.unconstrained_warmup_I_params[vax_status]['slope']
            self.warmup_I_params[vax_status]['intercept'] = \
                self.unconstrained_warmup_I_params[vax_status]['intercept']
            self.warmup_I_params[vax_status]['scale'] = \
                self.scale_transform.forward(self.unconstrained_warmup_I_params[vax_status]['scale'])

            self.warmup_IR_params[vax_status]['slope'] = \
                self.unconstrained_warmup_IR_params[vax_status]['slope']
            self.warmup_IR_params[vax_status]['intercept'] = \
                self.unconstrained_warmup_IR_params[vax_status]['intercept']
            self.warmup_IR_params[vax_status]['scale'] = \
                self.scale_transform.forward(self.unconstrained_warmup_IR_params[vax_status]['scale'])

            self.init_count_G_params[vax_status]['loc'] = \
                self.unconstrained_init_count_G_params[vax_status]['loc']
            self.init_count_G_params[vax_status]['scale'] = \
                self.scale_transform.forward(self.unconstrained_init_count_G_params[vax_status]['scale'])

            self.init_count_I_params[vax_status]['loc'] = \
                self.unconstrained_init_count_I_params[vax_status]['loc']
            self.init_count_I_params[vax_status]['scale'] = \
                self.scale_transform.forward(self.unconstrained_init_count_I_params[vax_status]['scale'])

        return

    def _sample_and_reparameterize(self):
        """Here we again constrain, and our prior distribution will fix it"""
        
        self._create_sample_tensor_dicts()


        (self.T_serial_samples[Vax.total.value],
         self.T_serial_samples_constrained[Vax.total.value],
         self.T_serial_probs[Vax.total.value])  = \
            self._sample_reparam_single(self.T_serial_params[Vax.total.value], tfp.bijectors.Softplus())

        (self.epsilon_samples[Vax.total.value],
         self.epsilon_samples_constrained[Vax.total.value],
         self.epsilon_probs[Vax.total.value]) = \
            self._sample_reparam_single(self.epsilon_params[Vax.total.value], tfp.bijectors.Sigmoid())

        (self.delta_samples[Vax.yes.value],
         self.delta_samples_constrained[Vax.yes.value],
         self.delta_probs[Vax.yes.value]) = \
            self._sample_reparam_single(self.delta_params[Vax.yes.value], tfp.bijectors.Sigmoid())

        for vax_status in [status.value for status in self.vax_statuses]:

            (self.rho_M_samples[vax_status],
             self.rho_M_samples_constrained[vax_status],
             self.rho_M_probs[vax_status]) =self._sample_reparam_single(self.rho_M_params[vax_status],
                                                                        tfp.bijectors.Sigmoid())

            (self.rho_G_samples[vax_status],
             self.rho_G_samples_constrained[vax_status],
             self.rho_G_probs[vax_status]) = self._sample_reparam_single(self.rho_G_params[vax_status],
                                                                         tfp.bijectors.Sigmoid())

            (self.rho_I_samples[vax_status],
             self.rho_I_samples_constrained[vax_status],
             self.rho_I_probs[vax_status]) = self._sample_reparam_single(self.rho_I_params[vax_status],
                                                                         tfp.bijectors.Sigmoid())

            (self.rho_D_samples[vax_status],
             self.rho_D_samples_constrained[vax_status],
             self.rho_D_probs[vax_status]) = self._sample_reparam_single(self.rho_D_params[vax_status],
                                                                         tfp.bijectors.Sigmoid())

            (self.lambda_M_samples[vax_status],
             self.lambda_M_samples_constrained[vax_status],
             self.lambda_M_probs[vax_status]) = self._sample_reparam_single(self.lambda_M_params[vax_status],
                                                                            tfp.bijectors.Softplus())

            (self.lambda_G_samples[vax_status],
             self.lambda_G_samples_constrained[vax_status],
             self.lambda_G_probs[vax_status]) = self._sample_reparam_single(self.lambda_G_params[vax_status],
                                                                            tfp.bijectors.Softplus())

            (self.lambda_I_samples[vax_status],
             self.lambda_I_samples_constrained[vax_status],
             self.lambda_I_probs[vax_status]) = self._sample_reparam_single(self.lambda_I_params[vax_status],
                                                                            tfp.bijectors.Softplus())

            (self.lambda_I_bar_samples[vax_status],
             self.lambda_I_bar_samples_constrained[vax_status],
             self.lambda_I_bar_probs[vax_status]) = self._sample_reparam_single(self.lambda_I_bar_params[vax_status],
                                                                                tfp.bijectors.Softplus())

            (self.lambda_D_samples[vax_status],
             self.lambda_D_samples_constrained[vax_status],
             self.lambda_D_probs[vax_status]) = self._sample_reparam_single(self.lambda_D_params[vax_status],
                                                                            tfp.bijectors.Softplus())

            (self.lambda_D_bar_samples[vax_status],
             self.lambda_D_bar_samples_constrained[vax_status],
             self.lambda_D_bar_probs[vax_status]) = self._sample_reparam_single(self.lambda_D_bar_params[vax_status],
                                                                                tfp.bijectors.Softplus())

            (self.nu_M_samples[vax_status],
             self.nu_M_samples_constrained[vax_status],
             self.nu_M_probs[vax_status]) = self._sample_reparam_single(self.nu_M_params[vax_status],
                                                                        tfp.bijectors.Softplus())

            (self.nu_G_samples[vax_status],
             self.nu_G_samples_constrained[vax_status],
             self.nu_G_probs[vax_status]) = self._sample_reparam_single(self.nu_G_params[vax_status],
                                                                        tfp.bijectors.Softplus())

            (self.nu_I_samples[vax_status],
             self.nu_I_samples_constrained[vax_status],
             self.nu_I_probs[vax_status]) = self._sample_reparam_single(self.nu_I_params[vax_status],
                                                                        tfp.bijectors.Softplus())

            (self.nu_I_bar_samples[vax_status],
             self.nu_I_bar_samples_constrained[vax_status],
             self.nu_I_bar_probs[vax_status]) = self._sample_reparam_single(self.nu_I_bar_params[vax_status],
                                                                            tfp.bijectors.Softplus())

            (self.nu_D_samples[vax_status],
             self.nu_D_samples_constrained[vax_status],
             self.nu_D_probs[vax_status]) = self._sample_reparam_single(self.nu_D_params[vax_status],
                                                                        tfp.bijectors.Softplus())

            (self.nu_D_bar_samples[vax_status],
             self.nu_D_bar_samples_constrained[vax_status],
             self.nu_D_bar_probs[vax_status]) = self._sample_reparam_single(self.nu_D_bar_params[vax_status],
                                                                            tfp.bijectors.Softplus())
            
            
            for day in range(self.transition_window):
                
                (samples,
                 samples_constrained,
                 probs) = \
                    self._sample_reparam_single(self.warmup_A_params[vax_status], 
                                                tfp.bijectors.Chain([tfp.bijectors.Scale(100),
                                                                     tfp.bijectors.Softplus()]),
                                                warmup=True, day=day)
                self.warmup_A_samples[vax_status].append(samples)
                self.warmup_A_samples_constrained[vax_status].append(samples_constrained)
                self.warmup_A_probs[vax_status].append(probs)

                (samples,
                 samples_constrained,
                 probs) = \
                    self._sample_reparam_single(self.warmup_M_params[vax_status],
                                                tfp.bijectors.Chain([tfp.bijectors.Scale(100),
                                                                     tfp.bijectors.Softplus()]),
                                                warmup=True, day=day)
                self.warmup_M_samples[vax_status].append(samples)
                self.warmup_M_samples_constrained[vax_status].append(samples_constrained)
                self.warmup_M_probs[vax_status].append(probs)

                (samples,
                 samples_constrained,
                 probs) = \
                    self._sample_reparam_single(self.warmup_G_params[vax_status],
                                                tfp.bijectors.Chain([tfp.bijectors.Scale(100),
                                                                     tfp.bijectors.Softplus()]),
                                                warmup=True, day=day)
                self.warmup_G_samples[vax_status].append(samples)
                self.warmup_G_samples_constrained[vax_status].append(samples_constrained)
                self.warmup_G_probs[vax_status].append(probs)

                (samples,
                 samples_constrained,
                 probs) = \
                    self._sample_reparam_single(self.warmup_GR_params[vax_status],
                                                tfp.bijectors.Chain([tfp.bijectors.Scale(100),
                                                                     tfp.bijectors.Softplus()]),
                                                warmup=True, day=day)
                self.warmup_GR_samples[vax_status].append(samples)
                self.warmup_GR_samples_constrained[vax_status].append(samples_constrained)
                self.warmup_GR_probs[vax_status].append(probs)

                (samples,
                 samples_constrained,
                 probs) = \
                    self._sample_reparam_single(self.warmup_I_params[vax_status],
                                                tfp.bijectors.Chain([tfp.bijectors.Scale(100),
                                                                     tfp.bijectors.Softplus()]),
                                                warmup=True, day=day)
                self.warmup_I_samples[vax_status].append(samples)
                self.warmup_I_samples_constrained[vax_status].append(samples_constrained)
                self.warmup_I_probs[vax_status].append(probs)

                (samples,
                 samples_constrained,
                 probs) = \
                    self._sample_reparam_single(self.warmup_IR_params[vax_status],
                                                tfp.bijectors.Chain([tfp.bijectors.Scale(100),
                                                                     tfp.bijectors.Softplus()]),
                                                warmup=True, day=day)
                self.warmup_IR_samples[vax_status].append(samples)
                self.warmup_IR_samples_constrained[vax_status].append(samples_constrained)
                self.warmup_IR_probs[vax_status].append(probs)

            (self.init_count_G_samples[vax_status],
             self.init_count_G_samples_constrained[vax_status],
             self.init_count_G_probs[vax_status]) = \
                self._sample_reparam_single(self.init_count_G_params[vax_status],
                                            tfp.bijectors.Chain([tfp.bijectors.Scale(100),
                                                                 tfp.bijectors.Softplus()]))

            (self.init_count_I_samples[vax_status],
             self.init_count_I_samples_constrained[vax_status],
             self.init_count_I_probs[vax_status]) = \
                self._sample_reparam_single(self.init_count_I_params[vax_status],
                                            tfp.bijectors.Chain([tfp.bijectors.Scale(100),
                                                                 tfp.bijectors.Softplus()]))


            poisson_M_dist_samples = [tfp.distributions.Poisson(rate=lambda_M)
                                      for lambda_M in self.lambda_M_samples_constrained[vax_status]]
    
            poisson_G_dist_samples = [tfp.distributions.Poisson(rate=lambda_G)
                                      for lambda_G in self.lambda_G_samples_constrained[vax_status]]
            poisson_I_dist_samples = [tfp.distributions.Poisson(rate=lambda_I)
                                      for lambda_I in self.lambda_I_samples_constrained[vax_status]]
            poisson_I_bar_dist_samples = [tfp.distributions.Poisson(rate=lambda_I_bar)
                                      for lambda_I_bar in self.lambda_I_bar_samples_constrained[vax_status]]
            poisson_D_dist_samples = [tfp.distributions.Poisson(rate=lambda_D)
                                      for lambda_D in self.lambda_D_samples_constrained[vax_status]]
            poisson_D_bar_dist_samples = [tfp.distributions.Poisson(rate=lambda_D_bar)
                                      for lambda_D_bar in self.lambda_D_bar_samples_constrained[vax_status]]
    
    
            self.pi_M_samples[vax_status] = tf.TensorArray(tf.float32, size=self.transition_window, clear_after_read=False,
                                               name='pi_M_samples')
    
            self.pi_G_samples[vax_status] = tf.TensorArray(tf.float32, size=self.transition_window, clear_after_read=False,
                                               name='pi_G_samples')
            self.pi_I_samples[vax_status] = tf.TensorArray(tf.float32, size=self.transition_window,
                                                           clear_after_read=False,
                                                           name='pi_I_samples')
            self.pi_I_bar_samples[vax_status] = tf.TensorArray(tf.float32, size=self.transition_window,
                                                           clear_after_read=False,
                                                           name='pi_I_bar_samples')
            self.pi_D_samples[vax_status] = tf.TensorArray(tf.float32, size=self.transition_window,
                                                           clear_after_read=False,
                                                           name='pi_D_samples')
            self.pi_D_bar_samples[vax_status] = tf.TensorArray(tf.float32, size=self.transition_window,
                                                           clear_after_read=False,
                                                           name='pi_D_bar_samples')
    
            for j in range(self.transition_window):
                self.pi_M_samples[vax_status] = self.pi_M_samples[vax_status].write(j, np.array([dist.log_prob(j + 1) for dist in poisson_M_dist_samples]) /
                                                               self.nu_M_samples_constrained[vax_status])
    
                self.pi_G_samples[vax_status] = self.pi_G_samples[vax_status].write(j, np.array(
                    [dist.log_prob(j + 1) for dist in poisson_G_dist_samples]) /
                                                            self.nu_G_samples_constrained[vax_status])
                self.pi_I_samples[vax_status] = self.pi_I_samples[vax_status].write(j, np.array(
                    [dist.log_prob(j + 1) for dist in poisson_I_dist_samples]) /
                                                                                    self.nu_I_samples_constrained[
                                                                                        vax_status])
                self.pi_I_bar_samples[vax_status] = self.pi_I_bar_samples[vax_status].write(j, np.array(
                    [dist.log_prob(j + 1) for dist in poisson_I_bar_dist_samples]) /
                                                                                    self.nu_I_bar_samples_constrained[
                                                                                        vax_status])
                self.pi_D_samples[vax_status] = self.pi_D_samples[vax_status].write(j, np.array(
                    [dist.log_prob(j + 1) for dist in poisson_D_dist_samples]) /
                                                                                    self.nu_D_samples_constrained[
                                                                                        vax_status])
                self.pi_D_bar_samples[vax_status] = self.pi_D_bar_samples[vax_status].write(j, np.array(
                    [dist.log_prob(j + 1) for dist in poisson_D_bar_dist_samples]) /
                                                                                    self.nu_D_bar_samples_constrained[
                                                                                        vax_status])
    
            self.pi_M_samples[vax_status] = self.pi_M_samples[vax_status].stack()
            self.pi_G_samples[vax_status] = self.pi_G_samples[vax_status].stack()
            self.pi_I_samples[vax_status] = self.pi_I_samples[vax_status].stack()
            self.pi_I_bar_samples[vax_status] = self.pi_I_bar_samples[vax_status].stack()
            self.pi_D_samples[vax_status] = self.pi_D_samples[vax_status].stack()
            self.pi_D_bar_samples[vax_status] = self.pi_D_bar_samples[vax_status].stack()
            # Softmax so it sums to 1
            self.pi_M_samples[vax_status] = tf.nn.softmax(self.pi_M_samples[vax_status], axis=0)
            self.pi_G_samples[vax_status] = tf.nn.softmax(self.pi_G_samples[vax_status], axis=0)
            self.pi_I_samples[vax_status] = tf.nn.softmax(self.pi_I_samples[vax_status], axis=0)
            self.pi_I_bar_samples[vax_status] = tf.nn.softmax(self.pi_I_bar_samples[vax_status], axis=0)
            self.pi_D_samples[vax_status] = tf.nn.softmax(self.pi_D_samples[vax_status], axis=0)
            self.pi_D_bar_samples[vax_status] = tf.nn.softmax(self.pi_D_bar_samples[vax_status], axis=0)

        return

    def _initialize_flux_arrays(self, forecast_days):

        forecasted_fluxes = {}

        for compartment in [comp.value for comp in self.compartments]:

            forecasted_fluxes[compartment] = {}
            for vax_status in [status.value for status in self.vax_statuses]:
                forecasted_fluxes[compartment][vax_status] = \
                    tf.TensorArray(tf.float32, size=forecast_days, clear_after_read=False,
                                   name=f'{compartment}_{vax_status}')

        return forecasted_fluxes

    def _initialize_count_arrays(self, forecast_days):
        forecasted_counts = {}
        for compartment in [comp.value for comp in [Comp.G, Comp.I]]:
            forecasted_counts[compartment] = {}
            for vax_status in [status.value for status in self.vax_statuses]:
                forecasted_counts[compartment][vax_status] = \
                    tf.TensorArray(tf.float32, size=forecast_days, clear_after_read=False,
                                   name=f'count_{compartment}_{vax_status}')

        return forecasted_counts

    def _add_prior_loss(self, debug=False):
        """Helper function for adding loss from model prior"""

        # Flip the signs from our elbo equation because tensorflow minimizes
        T_serial_prior_probs = [self.prior_distros[Comp.A.value][Vax.total.value]['T_serial'].log_prob(self.T_serial_samples_constrained[Vax.total.value]) + \
            tfp.bijectors.Softplus().forward_log_det_jacobian(self.T_serial_samples[Vax.total.value]) for status in self.vax_statuses]
        T_serial_posterior_probs = self.T_serial_probs[Vax.total.value]
        self.add_loss(lambda:  -tf.reduce_sum(tf.reduce_mean(T_serial_prior_probs- T_serial_posterior_probs ,axis=-1)))

        delta_prior_prob = self.prior_distros[Comp.A.value][Vax.yes.value]['delta'].log_prob(
            self.delta_samples_constrained[Vax.yes.value]) + \
                           tfp.bijectors.Sigmoid().forward_log_det_jacobian(self.delta_samples[Vax.yes.value])
        delta_posterior_prob = self.delta_probs[Vax.yes.value]
        self.add_loss(lambda: -tf.reduce_sum(tf.reduce_mean(delta_prior_prob - delta_posterior_prob, axis=-1)))

        epsilon_prior_prob = self.prior_distros[Comp.A.value][Vax.total.value]['epsilon'].log_prob(
            self.epsilon_samples_constrained[Vax.total.value]) + \
                           tfp.bijectors.Sigmoid().forward_log_det_jacobian(self.epsilon_samples[Vax.total.value])
        epsilon_posterior_prob = self.epsilon_probs[Vax.total.value]
        self.add_loss(lambda: -tf.reduce_sum(tf.reduce_mean(epsilon_prior_prob - epsilon_posterior_prob, axis=-1)))

        rho_M_prior_probs = [
            self.prior_distros[Comp.M.value][status.value]['rho_M'].log_prob(self.rho_M_samples_constrained[status.value]) + \
            tfp.bijectors.Softplus().forward_log_det_jacobian(self.rho_M_samples[status.value]) for status in self.vax_statuses]
        rho_M_posterior_probs = self.rho_M_probs
        self.add_loss(lambda: tf.reduce_sum([-tf.reduce_sum(
            tf.reduce_mean(rho_M_prior_probs[status.value] - rho_M_posterior_probs[status.value], axis=-1))
                                             for status in self.vax_statuses]))

        rho_G_prior_probs = [
            self.prior_distros[Comp.G.value][status.value]['rho_G'].log_prob(self.rho_G_samples_constrained[status.value]) + \
            tfp.bijectors.Softplus().forward_log_det_jacobian(self.rho_G_samples[status.value]) for status in self.vax_statuses]
        rho_G_posterior_probs = self.rho_G_probs
        self.add_loss(lambda: tf.reduce_sum([-tf.reduce_sum(
            tf.reduce_mean(rho_G_prior_probs[status.value] - rho_G_posterior_probs[status.value], axis=-1))
                                             for status in self.vax_statuses]))

        rho_I_prior_probs = [
            self.prior_distros[Comp.I.value][status.value]['rho_I'].log_prob(
                self.rho_I_samples_constrained[status.value]) + \
            tfp.bijectors.Softplus().forward_log_det_jacobian(self.rho_I_samples[status.value]) for status in
            self.vax_statuses]
        rho_I_posterior_probs = self.rho_I_probs
        self.add_loss(lambda: tf.reduce_sum([-tf.reduce_sum(
            tf.reduce_mean(rho_I_prior_probs[status.value] - rho_I_posterior_probs[status.value], axis=-1))
                                             for status in self.vax_statuses]))
        rho_D_prior_probs = [
            self.prior_distros[Comp.D.value][status.value]['rho_D'].log_prob(
                self.rho_D_samples_constrained[status.value]) + \
            tfp.bijectors.Softplus().forward_log_det_jacobian(self.rho_D_samples[status.value]) for status in
            self.vax_statuses]
        rho_D_posterior_probs = self.rho_D_probs
        self.add_loss(lambda: tf.reduce_sum([-tf.reduce_sum(
            tf.reduce_mean(rho_D_prior_probs[status.value] - rho_D_posterior_probs[status.value], axis=-1))
                                             for status in self.vax_statuses]))

        lambda_M_prior_probs = [
            self.prior_distros[Comp.M.value][status.value]['lambda_M'].log_prob(self.lambda_M_samples_constrained[status.value]) + \
            tfp.bijectors.Softplus().forward_log_det_jacobian(self.lambda_M_samples[status.value]) for status in self.vax_statuses]
        lambda_M_posterior_probs = self.lambda_M_probs
        self.add_loss(lambda: tf.reduce_sum([-tf.reduce_sum(
            tf.reduce_mean(lambda_M_prior_probs[status.value] - lambda_M_posterior_probs[status.value], axis=-1))
                                             for status in self.vax_statuses]))

        lambda_G_prior_probs = [
            self.prior_distros[Comp.G.value][status.value]['lambda_G'].log_prob(self.lambda_G_samples_constrained[status.value]) + \
            tfp.bijectors.Softplus().forward_log_det_jacobian(self.lambda_G_samples[status.value]) for status in self.vax_statuses]
        lambda_G_posterior_probs = self.lambda_G_probs
        self.add_loss(lambda: tf.reduce_sum([-tf.reduce_sum(
            tf.reduce_mean(lambda_G_prior_probs[status.value] - lambda_G_posterior_probs[status.value], axis=-1))
                                             for status in self.vax_statuses]))
        lambda_I_prior_probs = [
            self.prior_distros[Comp.I.value][status.value]['lambda_I'].log_prob(
                self.lambda_I_samples_constrained[status.value]) + \
            tfp.bijectors.Softplus().forward_log_det_jacobian(self.lambda_I_samples[status.value]) for status in
            self.vax_statuses]
        lambda_I_posterior_probs = self.lambda_I_probs
        self.add_loss(lambda: tf.reduce_sum([-tf.reduce_sum(
            tf.reduce_mean(lambda_I_prior_probs[status.value] - lambda_I_posterior_probs[status.value], axis=-1))
                                             for status in self.vax_statuses]))
        lambda_I_bar_prior_probs = [
            self.prior_distros[Comp.IR.value][status.value]['lambda_I_bar'].log_prob(
                self.lambda_I_bar_samples_constrained[status.value]) + \
            tfp.bijectors.Softplus().forward_log_det_jacobian(self.lambda_I_bar_samples[status.value]) for status in
            self.vax_statuses]
        lambda_I_bar_posterior_probs = self.lambda_I_bar_probs
        self.add_loss(lambda: tf.reduce_sum([-tf.reduce_sum(
            tf.reduce_mean(lambda_I_bar_prior_probs[status.value] - lambda_I_bar_posterior_probs[status.value], axis=-1))
                                             for status in self.vax_statuses]))

        lambda_D_prior_probs = [
            self.prior_distros[Comp.D.value][status.value]['lambda_D'].log_prob(
                self.lambda_D_samples_constrained[status.value]) + \
            tfp.bijectors.Softplus().forward_log_det_jacobian(self.lambda_D_samples[status.value]) for status in
            self.vax_statuses]
        lambda_D_posterior_probs = self.lambda_D_probs
        self.add_loss(lambda: tf.reduce_sum([-tf.reduce_sum(
            tf.reduce_mean(lambda_D_prior_probs[status.value] - lambda_D_posterior_probs[status.value], axis=-1))
                                             for status in self.vax_statuses]))
        lambda_D_bar_prior_probs = [
            self.prior_distros[Comp.D.value][status.value]['lambda_D_bar'].log_prob(
                self.lambda_D_bar_samples_constrained[status.value]) + \
            tfp.bijectors.Softplus().forward_log_det_jacobian(self.lambda_D_bar_samples[status.value]) for status in
            self.vax_statuses]
        lambda_D_bar_posterior_probs = self.lambda_D_bar_probs
        self.add_loss(lambda: tf.reduce_sum([-tf.reduce_sum(
            tf.reduce_mean(lambda_D_bar_prior_probs[status.value] - lambda_D_bar_posterior_probs[status.value], axis=-1))
                                             for status in self.vax_statuses]))

        nu_M_prior_probs = [
            self.prior_distros[Comp.M.value][status.value]['nu_M'].log_prob(self.nu_M_samples_constrained[status.value]) + \
            tfp.bijectors.Softplus().forward_log_det_jacobian(self.nu_M_samples[status.value]) for status in self.vax_statuses]
        nu_M_posterior_probs = self.nu_M_probs
        self.add_loss(lambda: tf.reduce_sum([-tf.reduce_sum(
            tf.reduce_mean(nu_M_prior_probs[status.value] - nu_M_posterior_probs[status.value], axis=-1))
                                             for status in self.vax_statuses]))

        nu_G_prior_probs = [
            self.prior_distros[Comp.G.value][status.value]['nu_G'].log_prob(self.nu_G_samples_constrained[status.value]) + \
            tfp.bijectors.Softplus().forward_log_det_jacobian(self.nu_G_samples[status.value]) for status in self.vax_statuses]
        nu_G_posterior_probs = self.nu_G_probs
        self.add_loss(lambda: tf.reduce_sum([-tf.reduce_sum(
            tf.reduce_mean(nu_G_prior_probs[status.value] - nu_G_posterior_probs[status.value], axis=-1))
                                             for status in self.vax_statuses]))
        nu_I_prior_probs = [
            self.prior_distros[Comp.I.value][status.value]['nu_I'].log_prob(
                self.nu_I_samples_constrained[status.value]) + \
            tfp.bijectors.Softplus().forward_log_det_jacobian(self.nu_I_samples[status.value]) for status in
            self.vax_statuses]
        nu_I_posterior_probs = self.nu_I_probs
        self.add_loss(lambda: tf.reduce_sum([-tf.reduce_sum(
            tf.reduce_mean(nu_I_prior_probs[status.value] - nu_I_posterior_probs[status.value], axis=-1))
                                             for status in self.vax_statuses]))
        nu_I_bar_prior_probs = [
            self.prior_distros[Comp.IR.value][status.value]['nu_I_bar'].log_prob(
                self.nu_I_bar_samples_constrained[status.value]) + \
            tfp.bijectors.Softplus().forward_log_det_jacobian(self.nu_I_bar_samples[status.value]) for status in
            self.vax_statuses]
        nu_I_bar_posterior_probs = self.nu_I_bar_probs
        self.add_loss(lambda: tf.reduce_sum([-tf.reduce_sum(
            tf.reduce_mean(nu_I_bar_prior_probs[status.value] - nu_I_bar_posterior_probs[status.value], axis=-1))
                                             for status in self.vax_statuses]))
        nu_D_prior_probs = [
            self.prior_distros[Comp.D.value][status.value]['nu_D'].log_prob(
                self.nu_D_samples_constrained[status.value]) + \
            tfp.bijectors.Softplus().forward_log_det_jacobian(self.nu_D_samples[status.value]) for status in
            self.vax_statuses]
        nu_D_posterior_probs = self.nu_D_probs
        self.add_loss(lambda: tf.reduce_sum([-tf.reduce_sum(
            tf.reduce_mean(nu_D_prior_probs[status.value] - nu_D_posterior_probs[status.value], axis=-1))
                                             for status in self.vax_statuses]))
        nu_D_bar_prior_probs = [
            self.prior_distros[Comp.D.value][status.value]['nu_D_bar'].log_prob(
                self.nu_D_bar_samples_constrained[status.value]) + \
            tfp.bijectors.Softplus().forward_log_det_jacobian(self.nu_D_bar_samples[status.value]) for status in
            self.vax_statuses]
        nu_D_bar_posterior_probs = self.nu_D_bar_probs
        self.add_loss(lambda: tf.reduce_sum([-tf.reduce_sum(
            tf.reduce_mean(nu_D_bar_prior_probs[status.value] - nu_D_bar_posterior_probs[status.value], axis=-1))
                                             for status in self.vax_statuses]))

        init_count_G_prior_probs = [
            self.prior_distros[Comp.G.value][status.value]['init_count_G'].log_prob(
                self.init_count_G_samples[status.value]) + \
            tfp.bijectors.Chain([tfp.bijectors.Scale(100),
                                 tfp.bijectors.Softplus()]).forward_log_det_jacobian(
                self.init_count_G_samples[status.value]) for status in
            self.vax_statuses]
        init_count_G_posterior_probs = self.init_count_G_probs
        self.add_loss(lambda: tf.reduce_sum([-tf.reduce_sum(
            tf.reduce_mean(init_count_G_prior_probs[status.value] - init_count_G_posterior_probs[status.value], axis=-1))
                                             for status in self.vax_statuses]))

        init_count_I_prior_probs = [
            self.prior_distros[Comp.I.value][status.value]['init_count_I'].log_prob(
                self.init_count_I_samples[status.value]) + \
            tfp.bijectors.Chain([tfp.bijectors.Scale(100),
                                 tfp.bijectors.Softplus()]).forward_log_det_jacobian(
                self.init_count_I_samples[status.value]) for status in
            self.vax_statuses]
        init_count_I_posterior_probs = self.init_count_I_probs
        self.add_loss(lambda: tf.reduce_sum([-tf.reduce_sum(
            tf.reduce_mean(init_count_I_prior_probs[status.value] - init_count_I_posterior_probs[status.value],
                           axis=-1))
                                             for status in self.vax_statuses]))

        # open bug about adding loss inisde a for loop: https://github.com/tensorflow/tensorflow/issues/44590
        # sum over status, mean over days, sum over nothing, mean over draws
        self.add_loss(lambda:  tf.reduce_sum([tf.reduce_mean([-tf.reduce_sum(tf.reduce_mean(
                self.prior_distros[Comp.A.value][status.value]['warmup_A'][day].log_prob(
                    self.warmup_A_samples[status.value][day]) + \
                tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]).forward_log_det_jacobian(self.warmup_A_samples[status.value][day])
                - self.warmup_A_probs[status.value][day],axis=-1)) for day in range(self.transition_window)])for status in self.vax_statuses]))

        self.add_loss(lambda: tf.reduce_sum([tf.reduce_mean([-tf.reduce_sum(tf.reduce_mean(
            self.prior_distros[Comp.M.value][status.value]['warmup_M'][day].log_prob(
                self.warmup_M_samples[status.value][day]) + \
            tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]).forward_log_det_jacobian(self.warmup_M_samples[status.value][day])
            - self.warmup_M_probs[status.value][day], axis=-1)) for day in range(self.transition_window)]) for status in
                                             self.vax_statuses]))
        self.add_loss(lambda: tf.reduce_sum([tf.reduce_mean([-tf.reduce_sum(tf.reduce_mean(
            self.prior_distros[Comp.G.value][status.value]['warmup_G'][day].log_prob(
                self.warmup_G_samples[status.value][day]) + \
            tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]).forward_log_det_jacobian(
                self.warmup_G_samples[status.value][day])
            - self.warmup_G_probs[status.value][day], axis=-1)) for day in range(self.transition_window)]) for status in
                                             self.vax_statuses]))
        self.add_loss(lambda: tf.reduce_sum([tf.reduce_mean([-tf.reduce_sum(tf.reduce_mean(
            self.prior_distros[Comp.GR.value][status.value]['warmup_GR'][day].log_prob(
                self.warmup_GR_samples[status.value][day]) + \
            tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]).forward_log_det_jacobian(
                self.warmup_GR_samples[status.value][day])
            - self.warmup_GR_probs[status.value][day], axis=-1)) for day in range(self.transition_window)]) for status in
                                             self.vax_statuses]))
        self.add_loss(lambda: tf.reduce_sum([tf.reduce_mean([-tf.reduce_sum(tf.reduce_mean(
            self.prior_distros[Comp.I.value][status.value]['warmup_I'][day].log_prob(
                self.warmup_I_samples[status.value][day]) + \
            tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]).forward_log_det_jacobian(
                self.warmup_I_samples[status.value][day])
            - self.warmup_I_probs[status.value][day], axis=-1)) for day in range(self.transition_window)]) for status in
                                             self.vax_statuses]))
        self.add_loss(lambda: tf.reduce_sum([tf.reduce_mean([-tf.reduce_sum(tf.reduce_mean(
            self.prior_distros[Comp.IR.value][status.value]['warmup_IR'][day].log_prob(
                self.warmup_IR_samples[status.value][day]) + \
            tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]).forward_log_det_jacobian(
                self.warmup_IR_samples[status.value][day])
            - self.warmup_IR_probs[status.value][day], axis=-1)) for day in range(self.transition_window)]) for status in
                                             self.vax_statuses]))


        if debug:
            print(f'Rho M loss {vax_status}: {rho_M_loss}')
            print(f'Rho X loss {vax_status}: {rho_X_loss}')
            print(f'Rho G loss {vax_status}: {rho_G_loss}')
            print(f'lam M loss {vax_status}: {lambda_M_loss}')
            print(f'lam X loss {vax_status}: {lambda_X_loss}')
            print(f'lam G loss {vax_status}: {lambda_G_loss}')
            print(f'nu M loss {vax_status}: {nu_M_loss}')
            print(f'nu X loss {vax_status}: {nu_X_loss}')
            print(f'nu G loss {vax_status}: {nu_G_loss}')


    def _mark_arrays_used(self, forecasted_fluxes):
        """Helper function that supresses noisy error about not using all arrays"""
        for vax_status in [status.value for status in self.vax_statuses]:
            forecasted_fluxes[Comp.A.value][vax_status].mark_used()
            forecasted_fluxes[Comp.M.value][vax_status].mark_used()
            forecasted_fluxes[Comp.G.value][vax_status].mark_used()
            forecasted_fluxes[Comp.GR.value][vax_status].mark_used()
            forecasted_fluxes[Comp.I.value][vax_status].mark_used()
            forecasted_fluxes[Comp.IR.value][vax_status].mark_used()
            forecasted_fluxes[Comp.D.value][vax_status].mark_used()
            self.previously_asymptomatic[vax_status].mark_used()
            self.previously_mild[vax_status].mark_used()
            self.previously_gen[vax_status].mark_used()
            self.previously_icu[vax_status].mark_used()

        return


    def _get_prev_tensors(self, forecasted_fluxes, vax_status, day):
        """Build a tensor with the last j days of each compartment we can multiply this by pi

        previously_[compartment]_tensor is (j, 1000), and the days are in reverse order:
            previously_mild_tensor[0,:] = 1000 samples of the mild influx at day t-1
            previously_mild_tensor[4,:] = 1000 samples of the mild influx at day t-5

        This way, pi[0] can be the probability of a duration 1 transition and the two tensors
            can be elementwise multiplied

        Returns:
            (asmyptomatic, mild, gen, icu)
        """
        for j in range(self.transition_window):

            if day - j - 1 < 0:
                j_ago_asymp = self.warmup_A_samples_constrained[vax_status][day - j - 1]
                j_ago_mild = self.warmup_M_samples_constrained[vax_status][day - j - 1]
                j_ago_gen = self.warmup_G_samples_constrained[vax_status][day - j - 1]
                j_ago_genrec = self.warmup_G_samples_constrained[vax_status][day - j - 1]
                j_ago_icu = self.warmup_I_samples_constrained[vax_status][day - j - 1]
                j_ago_icurec = self.warmup_I_samples_constrained[vax_status][day - j - 1]
            else:
                j_ago_asymp = forecasted_fluxes[Comp.A.value][vax_status].read(day - j - 1)
                j_ago_mild = forecasted_fluxes[Comp.M.value][vax_status].read(day - j - 1)
                j_ago_gen = forecasted_fluxes[Comp.G.value][vax_status].read(day - j - 1)
                j_ago_genrec = forecasted_fluxes[Comp.GR.value][vax_status].read(day - j - 1)
                j_ago_icu = forecasted_fluxes[Comp.I.value][vax_status].read(day - j - 1)
                j_ago_icurec = forecasted_fluxes[Comp.IR.value][vax_status].read(day - j - 1)

            self.previously_asymptomatic[vax_status] = \
                self.previously_asymptomatic[vax_status].write(j, j_ago_asymp)
            self.previously_mild[vax_status] = \
                self.previously_mild[vax_status].write(j, j_ago_mild)
            self.previously_gen[vax_status] = \
                self.previously_gen[vax_status].write(j, j_ago_gen)
            self.previously_icu[vax_status] = \
                self.previously_icu[vax_status].write(j, j_ago_icu)

        previously_asymptomatic_tensor = self.previously_asymptomatic[vax_status].stack()
        previously_mild_tensor = self.previously_mild[vax_status].stack()
        previously_gen_tensor = self.previously_gen[vax_status].stack()
        previously_icu_tensor = self.previously_icu[vax_status].stack()

        return (previously_asymptomatic_tensor, previously_mild_tensor, previously_gen_tensor, previously_icu_tensor)


    def _create_sample_tensor_dicts(self):
        self.T_serial_samples = defaultdict(int)
        self.T_serial_samples_constrained = defaultdict(int)
        self.T_serial_probs = defaultdict(int)

        self.epsilon_samples = defaultdict(int)
        self.epsilon_samples_constrained = defaultdict(int)
        self.epsilon_probs = defaultdict(int)

        self.delta_samples = defaultdict(int)
        self.delta_samples_constrained = defaultdict(int)
        self.delta_probs = defaultdict(int)

        self.rho_M_samples = defaultdict(int)
        self.rho_M_samples_constrained = defaultdict(int)
        self.rho_M_probs = defaultdict(int)

        self.rho_G_samples = defaultdict(int)
        self.rho_G_samples_constrained = defaultdict(int)
        self.rho_G_probs = defaultdict(int)

        self.rho_I_samples = defaultdict(int)
        self.rho_I_samples_constrained = defaultdict(int)
        self.rho_I_probs = defaultdict(int)

        self.rho_D_samples = defaultdict(int)
        self.rho_D_samples_constrained = defaultdict(int)
        self.rho_D_probs = defaultdict(int)

        self.lambda_M_samples = defaultdict(int)
        self.lambda_M_samples_constrained = defaultdict(int)
        self.lambda_M_probs = defaultdict(int)

        self.lambda_G_samples = defaultdict(int)
        self.lambda_G_samples_constrained = defaultdict(int)
        self.lambda_G_probs = defaultdict(int)
        self.lambda_I_samples = defaultdict(int)
        self.lambda_I_samples_constrained = defaultdict(int)
        self.lambda_I_probs = defaultdict(int)
        self.lambda_I_bar_samples = defaultdict(int)
        self.lambda_I_bar_samples_constrained = defaultdict(int)
        self.lambda_I_bar_probs = defaultdict(int)

        self.lambda_D_samples = defaultdict(int)
        self.lambda_D_samples_constrained = defaultdict(int)
        self.lambda_D_probs = defaultdict(int)
        self.lambda_D_bar_samples = defaultdict(int)
        self.lambda_D_bar_samples_constrained = defaultdict(int)
        self.lambda_D_bar_probs = defaultdict(int)

        self.nu_M_samples = defaultdict(int)
        self.nu_M_samples_constrained = defaultdict(int)
        self.nu_M_probs = defaultdict(int)

        self.nu_G_samples = defaultdict(int)
        self.nu_G_samples_constrained = defaultdict(int)
        self.nu_G_probs = defaultdict(int)
        self.nu_I_samples = defaultdict(int)
        self.nu_I_samples_constrained = defaultdict(int)
        self.nu_I_probs = defaultdict(int)
        self.nu_I_bar_samples = defaultdict(int)
        self.nu_I_bar_samples_constrained = defaultdict(int)
        self.nu_I_bar_probs = defaultdict(int)
        self.nu_D_samples = defaultdict(int)
        self.nu_D_samples_constrained = defaultdict(int)
        self.nu_D_probs = defaultdict(int)
        self.nu_D_bar_samples = defaultdict(int)
        self.nu_D_bar_samples_constrained = defaultdict(int)
        self.nu_D_bar_probs = defaultdict(int)

        self.warmup_A_samples = defaultdict(int)
        self.warmup_A_samples_constrained = defaultdict(int)
        self.warmup_A_probs = defaultdict(int)

        self.warmup_M_samples = defaultdict(int)
        self.warmup_M_samples_constrained = defaultdict(int)
        self.warmup_M_probs = defaultdict(int)
        self.warmup_G_samples = defaultdict(int)
        self.warmup_G_samples_constrained = defaultdict(int)
        self.warmup_G_probs = defaultdict(int)
        self.warmup_GR_samples = defaultdict(int)
        self.warmup_GR_samples_constrained = defaultdict(int)
        self.warmup_GR_probs = defaultdict(int)
        self.warmup_I_samples = defaultdict(int)
        self.warmup_I_samples_constrained = defaultdict(int)
        self.warmup_I_probs = defaultdict(int)
        self.warmup_IR_samples = defaultdict(int)
        self.warmup_IR_samples_constrained = defaultdict(int)
        self.warmup_IR_probs = defaultdict(int)

        self.init_count_G_samples = defaultdict(int)
        self.init_count_G_samples_constrained = defaultdict(int)
        self.init_count_G_probs = defaultdict(int)
        self.init_count_I_samples = defaultdict(int)
        self.init_count_I_samples_constrained = defaultdict(int)
        self.init_count_I_probs = defaultdict(int)

        self.pi_M_samples = defaultdict(int)
        self.pi_G_samples = defaultdict(int)
        self.pi_I_samples = defaultdict(int)
        self.pi_I_bar_samples = defaultdict(int)
        self.pi_D_samples = defaultdict(int)
        self.pi_D_bar_samples = defaultdict(int)

        for vax_status in [status.value for status in self.vax_statuses]:
            self.warmup_A_samples[vax_status] = []
            self.warmup_A_samples_constrained[vax_status] = []
            self.warmup_A_probs[vax_status] = []
            self.warmup_M_samples[vax_status] = []
            self.warmup_M_samples_constrained[vax_status] = []
            self.warmup_M_probs[vax_status] = []
            self.warmup_G_samples[vax_status] = []
            self.warmup_G_samples_constrained[vax_status] = []
            self.warmup_G_probs[vax_status] = []
            self.warmup_GR_samples[vax_status] = []
            self.warmup_GR_samples_constrained[vax_status] = []
            self.warmup_GR_probs[vax_status] = []
            self.warmup_I_samples[vax_status] = []
            self.warmup_I_samples_constrained[vax_status] = []
            self.warmup_I_probs[vax_status] = []
            self.warmup_IR_samples[vax_status] = []
            self.warmup_IR_samples_constrained[vax_status] = []
            self.warmup_IR_probs[vax_status] = []

        return


    def _sample_reparam_single(self, params, bijector, warmup=False, day=None):
        """Create samples and reparameterize a single variable.

        Args:
            params (dict): dictionary containing variational 'loc' and 'scale params
            bijector (tfp.bijectors): Transformation real numbers -> space we care about
            warmup (bool): Optional, if True, will calculate the mean as intercept + slope * day rather than loc
        Returns:
            samples (tf.Tensor): Tensor (or list of tensors) for storing samples on the real line
            samples_constrained (tf.Tensor): Tensor (or list) for storing samples in data space
            probs (tf.Tensor): Tensor (or list) for storing posterior probabilities
        """

        noise = tf.random.normal((self.posterior_samples, ))

        if warmup:
            assert(day is not None)
            mean = params['slope']*day + params['intercept']
        else:
            mean = params['loc']

        # Create samples, mean is still on real numbers
        samples = mean + params['scale']*noise

        # get samples in interpretable space
        samples_constrained = bijector.forward(samples)

        # faster to store this as a model object rather than make it
        # every loop?
        variational_posterior = tfp.distributions.Normal(mean,
                                                         params['scale'])

        probs = variational_posterior.log_prob(samples)

        return samples, samples_constrained, probs

# Custom LogPoisson Probability Loss function
def calc_poisson(inputs):
    predicted_rate, true_rate = inputs
    poisson = tfp.distributions.Poisson(rate=predicted_rate+1e-6)
    return poisson.log_prob(true_rate)


class LogPoissonProb(tf.keras.losses.Loss):

    def call(self, y_true, y_pred, debug=False):
        if len(y_true.shape)==3:
            # we got a batch
            y_true = y_true[0]
            y_pred = y_pred[0]


        y_t = {}
        y_t['G_count'] = y_true[0]
        y_t['G_in'] = y_true[1]
        y_t['I_count'] = y_true[2]
        y_t['D_in'] = y_true[3]

        y_p = {}
        y_p['G_count'] = y_pred[0]
        y_p['G_in'] = y_pred[1]
        y_p['I_count'] = y_pred[2]
        y_p['D_in'] = y_pred[3]

        log_probs_G_count = tf.map_fn(calc_poisson, (y_p['G_count'], y_t['G_count']),
                                      fn_output_signature=tf.float32)
        log_probs_G_in = tf.map_fn(calc_poisson, (tf.squeeze(y_p['G_in']), y_t['G_in']),
                                      fn_output_signature=tf.float32)
        log_probs_I_count = tf.map_fn(calc_poisson, (y_p['I_count'], y_t['I_count']),
                                      fn_output_signature=tf.float32)
        log_probs_D_in = tf.map_fn(calc_poisson, (tf.squeeze(y_p['D_in']), y_t['D_in']),
                                   fn_output_signature=tf.float32)

        # mean over days, mean over draws
        G_count_log_likelihood = tf.reduce_mean(tf.reduce_mean(log_probs_G_count,axis=1))
        G_in_log_likelihood = tf.reduce_mean(tf.reduce_mean(log_probs_G_in, axis=1))
        I_count_log_likelihood = tf.reduce_mean(tf.reduce_mean(log_probs_I_count, axis=1))
        D_in_log_likelihood = tf.reduce_mean(tf.reduce_mean(log_probs_D_in, axis=1))

        if True:
            print(f'G count: {G_count_log_likelihood}')
            print(f'G in: {G_in_log_likelihood}')
            print(f'I count: {I_count_log_likelihood}')
            print(f'D in: {D_in_log_likelihood}')

        # return negative log likielihood
        return -G_count_log_likelihood + \
               -G_in_log_likelihood + \
               -I_count_log_likelihood + \
               -D_in_log_likelihood

class ConfigCallback(tf.keras.callbacks.Callback):

    def __init__(self, config_outpath, every_nth_epoch=10):
        self.every_nth_epoch = every_nth_epoch
        self.config_outpath = config_outpath

    def on_epoch_end(self, epoch, logs):

        if epoch % self.every_nth_epoch != 0:
            return

        self.model.config = self.model.config.update_from_model(self.model)
        self.model.config.to_json(self.config_outpath)

class VarLogCallback(tf.keras.callbacks.Callback):
    """Logs all our model parameters"""

    def __init__(self, every_nth_epoch=1):
        self.every_nth_epoch = every_nth_epoch

    def on_epoch_end(self, epoch, logs):

        if epoch % self.every_nth_epoch != 0:
            return

        tf.summary.scalar(f'T_serial_mean',
                          data=tf.squeeze(tf.math.softplus(self.model.unconstrained_T_serial['loc'])),
                          step=epoch)
        tf.summary.scalar(f'T_serial_scale',
                          data=tf.squeeze(tf.math.softplus(self.model.unconstrained_T_serial['scale'])),
                          step=epoch)

        tf.summary.scalar(f'epsilon_mean',
                          data=tf.squeeze(tf.math.softplus(self.model.unconstrained_epsilon['loc'])),
                          step=epoch)
        tf.summary.scalar(f'epsilon_scale',
                          data=tf.squeeze(
                              tf.math.softplus(self.model.unconstrained_epsilon['scale'])),
                          step=epoch)

        tf.summary.scalar(f'delta_mean',
                          data=tf.squeeze(tf.math.softplus(self.model.unconstrained_delta['loc'])),
                          step=epoch)
        tf.summary.scalar(f'delta_scale',
                          data=tf.squeeze(
                              tf.math.softplus(self.model.unconstrained_delta['scale'])),
                          step=epoch)

        for vax_status in [status.value for status in self.model.vax_statuses]:

            
            tf.summary.scalar(f'rho_M_mean_{vax_status}',
                              data=tf.squeeze(tf.math.sigmoid(self.model.unconstrained_rho_M[vax_status]['loc'])),
                              step=epoch)
            tf.summary.scalar(f'rho_M_scale_{vax_status}',
                              data=tf.squeeze(tf.math.softplus(self.model.unconstrained_rho_M[vax_status]['scale'])),
                              step=epoch)
            tf.summary.scalar(f'lambda_M_mean_{vax_status}',
                              data=tf.squeeze(tf.math.softplus(self.model.unconstrained_lambda_M[vax_status]['loc'])),
                              step=epoch)
            tf.summary.scalar(f'lambda_M_scale_{vax_status}',
                              data=tf.squeeze(tf.math.softplus(self.model.unconstrained_lambda_M[vax_status]['scale'])),
                              step=epoch)
            tf.summary.scalar(f'nu_M_mean_{vax_status}',
                              data=tf.squeeze(tf.math.softplus(self.model.unconstrained_nu_M[vax_status]['loc'])),
                              step=epoch)
            tf.summary.scalar(f'nu_M_scale_{vax_status}',
                              data=tf.squeeze(tf.math.softplus(self.model.unconstrained_nu_M[vax_status]['scale'])),
                              step=epoch)
            tf.summary.scalar(f'rho_G_mean_{vax_status}',
                              data=tf.squeeze(tf.math.sigmoid(self.model.unconstrained_rho_G[vax_status]['loc'])),
                              step=epoch)
            tf.summary.scalar(f'rho_G_scale_{vax_status}',
                              data=tf.squeeze(tf.math.softplus(self.model.unconstrained_rho_G[vax_status]['scale'])),
                              step=epoch)
            tf.summary.scalar(f'lambda_G_mean_{vax_status}',
                              data=tf.squeeze(tf.math.softplus(self.model.unconstrained_lambda_G[vax_status]['loc'])),
                              step=epoch)
            tf.summary.scalar(f'lambda_G_scale_{vax_status}',
                              data=tf.squeeze(tf.math.softplus(self.model.unconstrained_lambda_G[vax_status]['scale'])),
                              step=epoch)
            tf.summary.scalar(f'nu_G_mean_{vax_status}',
                              data=tf.squeeze(tf.math.softplus(self.model.unconstrained_nu_G[vax_status]['loc'])),
                              step=epoch)
            tf.summary.scalar(f'nu_G_scale_{vax_status}',
                              data=tf.squeeze(tf.math.softplus(self.model.unconstrained_nu_G[vax_status]['scale'])),
                              step=epoch)
            tf.summary.scalar(f'lambda_I_bar_mean_{vax_status}',
                              data=tf.squeeze(tf.math.softplus(self.model.unconstrained_lambda_I_bar[vax_status]['loc'])),
                              step=epoch)
            tf.summary.scalar(f'lambda_I_bar_scale_{vax_status}',
                              data=tf.squeeze(tf.math.softplus(self.model.unconstrained_lambda_I_bar[vax_status]['scale'])),
                              step=epoch)
            tf.summary.scalar(f'nu_I_bar_mean_{vax_status}',
                              data=tf.squeeze(tf.math.softplus(self.model.unconstrained_nu_I_bar[vax_status]['loc'])),
                              step=epoch)
            tf.summary.scalar(f'nu_I_bar_scale_{vax_status}',
                              data=tf.squeeze(tf.math.softplus(self.model.unconstrained_nu_I_bar[vax_status]['scale'])),
                              step=epoch)
            tf.summary.scalar(f'rho_I_mean_{vax_status}',
                              data=tf.squeeze(tf.math.sigmoid(self.model.unconstrained_rho_I[vax_status]['loc'])),
                              step=epoch)
            tf.summary.scalar(f'rho_I_scale_{vax_status}',
                              data=tf.squeeze(tf.math.softplus(self.model.unconstrained_rho_I[vax_status]['scale'])),
                              step=epoch)
            tf.summary.scalar(f'lambda_I_mean_{vax_status}',
                              data=tf.squeeze(tf.math.softplus(self.model.unconstrained_lambda_I[vax_status]['loc'])),
                              step=epoch)
            tf.summary.scalar(f'lambda_I_scale_{vax_status}',
                              data=tf.squeeze(tf.math.softplus(self.model.unconstrained_lambda_I[vax_status]['scale'])),
                              step=epoch)
            tf.summary.scalar(f'nu_I_mean_{vax_status}',
                              data=tf.squeeze(tf.math.softplus(self.model.unconstrained_nu_I[vax_status]['loc'])),
                              step=epoch)
            tf.summary.scalar(f'nu_I_scale_{vax_status}',
                              data=tf.squeeze(tf.math.softplus(self.model.unconstrained_nu_I[vax_status]['scale'])),
                              step=epoch)
            tf.summary.scalar(f'rho_D_mean_{vax_status}',
                              data=tf.squeeze(tf.math.sigmoid(self.model.unconstrained_rho_D[vax_status]['loc'])),
                              step=epoch)
            tf.summary.scalar(f'rho_D_scale_{vax_status}',
                              data=tf.squeeze(tf.math.softplus(self.model.unconstrained_rho_D[vax_status]['scale'])),
                              step=epoch)
            tf.summary.scalar(f'lambda_D_mean_{vax_status}',
                              data=tf.squeeze(tf.math.softplus(self.model.unconstrained_lambda_D[vax_status]['loc'])),
                              step=epoch)
            tf.summary.scalar(f'lambda_D_scale_{vax_status}',
                              data=tf.squeeze(tf.math.softplus(self.model.unconstrained_lambda_D[vax_status]['scale'])),
                              step=epoch)
            tf.summary.scalar(f'nu_D_mean_{vax_status}',
                              data=tf.squeeze(tf.math.softplus(self.model.unconstrained_nu_D[vax_status]['loc'])),
                              step=epoch)
            tf.summary.scalar(f'nu_D_scale_{vax_status}',
                              data=tf.squeeze(tf.math.softplus(self.model.unconstrained_nu_D[vax_status]['scale'])),
                              step=epoch)
            tf.summary.scalar(f'lambda_D_bar_mean_{vax_status}',
                              data=tf.squeeze(
                                  tf.math.softplus(self.model.unconstrained_lambda_D_bar[vax_status]['loc'])),
                              step=epoch)
            tf.summary.scalar(f'lambda_D_bar_scale_{vax_status}',
                              data=tf.squeeze(
                                  tf.math.softplus(self.model.unconstrained_lambda_D_bar[vax_status]['scale'])),
                              step=epoch)
            tf.summary.scalar(f'nu_D_bar_mean_{vax_status}',
                              data=tf.squeeze(tf.math.softplus(self.model.unconstrained_nu_D_bar[vax_status]['loc'])),
                              step=epoch)
            tf.summary.scalar(f'nu_D_bar_scale_{vax_status}',
                              data=tf.squeeze(tf.math.softplus(self.model.unconstrained_nu_D_bar[vax_status]['scale'])),
                              step=epoch)

            tf.summary.scalar(f'warmup_A_mean_{vax_status}', data=tf.squeeze(tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]).forward(self.model.unconstrained_warmup_A_params[vax_status]['intercept'])), step=epoch)
            tf.summary.scalar(f'warmup_A_scale_{vax_status}',
                              data=tf.squeeze(tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]).forward(self.model.unconstrained_warmup_A_params[vax_status]['scale'])), step=epoch)
            tf.summary.scalar(f'warmup_M_mean_{vax_status}', data=tf.squeeze(
                tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]).forward(
                    self.model.unconstrained_warmup_M_params[vax_status]['intercept'])), step=epoch)
            tf.summary.scalar(f'warmup_M_scale_{vax_status}',
                              data=tf.squeeze(
                                  tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]).forward(
                                      self.model.unconstrained_warmup_M_params[vax_status]['scale'])), step=epoch)
            tf.summary.scalar(f'warmup_G_mean_{vax_status}', data=tf.squeeze(
                tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]).forward(
                    self.model.unconstrained_warmup_G_params[vax_status]['intercept'])), step=epoch)
            tf.summary.scalar(f'warmup_G_scale_{vax_status}',
                              data=tf.squeeze(
                                  tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]).forward(
                                      self.model.unconstrained_warmup_G_params[vax_status]['scale'])), step=epoch)
            tf.summary.scalar(f'warmup_GR_mean_{vax_status}', data=tf.squeeze(
                tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]).forward(
                    self.model.unconstrained_warmup_GR_params[vax_status]['intercept'])), step=epoch)
            tf.summary.scalar(f'warmup_GR_scale_{vax_status}',
                              data=tf.squeeze(
                                  tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]).forward(
                                      self.model.unconstrained_warmup_GR_params[vax_status]['scale'])), step=epoch)
            tf.summary.scalar(f'warmup_I_mean_{vax_status}', data=tf.squeeze(
                tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]).forward(
                    self.model.unconstrained_warmup_I_params[vax_status]['intercept'])), step=epoch)
            tf.summary.scalar(f'warmup_I_scale_{vax_status}',
                              data=tf.squeeze(
                                  tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]).forward(
                                      self.model.unconstrained_warmup_I_params[vax_status]['scale'])), step=epoch)
            tf.summary.scalar(f'warmup_IR_mean_{vax_status}', data=tf.squeeze(
                tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]).forward(
                    self.model.unconstrained_warmup_IR_params[vax_status]['intercept'])), step=epoch)
            tf.summary.scalar(f'warmup_IR_scale_{vax_status}',
                              data=tf.squeeze(
                                  tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]).forward(
                                      self.model.unconstrained_warmup_IR_params[vax_status]['scale'])), step=epoch)

            tf.summary.scalar(f'init_count_G_mean_{vax_status}',
                              data=tf.squeeze(
                tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]).forward(
                    self.model.unconstrained_init_count_G_params[vax_status]['loc'])),
                              step=epoch)
            tf.summary.scalar(f'init_count_G_scale_{vax_status}',
                              data=tf.squeeze(
                tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]).forward(
                    self.model.unconstrained_init_count_G_params[vax_status]['scale'])),
                              step=epoch)
            tf.summary.scalar(f'init_count_I_mean_{vax_status}',
                              data=tf.squeeze(
                tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]).forward(
                    self.model.unconstrained_init_count_I_params[vax_status]['loc'])),
                              step=epoch)
            tf.summary.scalar(f'init_count_I_scale_{vax_status}',
                              data=tf.squeeze(
                tfp.bijectors.Chain([tfp.bijectors.Scale(100), tfp.bijectors.Softplus()]).forward(
                    self.model.unconstrained_init_count_I_params[vax_status]['scale'])),
                              step=epoch)

        return

def get_logging_callbacks(log_dir):
    """Get tensorflow callbacks to write tensorboard logs to given log_dir"""
    file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    file_writer.set_as_default()
    logging_callback = VarLogCallback()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    config_callback = ConfigCallback(log_dir + "/saved_config.json")
    return [logging_callback, tensorboard_callback, config_callback]

