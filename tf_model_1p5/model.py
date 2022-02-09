import copy
from enum import Enum

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense

import tensorflow_probability as tfp
from scipy.stats import beta, truncnorm


class Comp(Enum):
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
                 transition_window, T_serial, epsilon, delta,
                 rho_M, lambda_M, nu_M,
                 rho_G, lambda_G, nu_G,
                 rho_I, lambda_I, nu_I,
                 lambda_I_bar, nu_I_bar,
                 rho_D, lambda_D, nu_D,
                 lambda_D_bar, nu_D_bar,
                 warmup_A_params,
                 warmup_M_params,
                 warmup_G_params, warmup_GR_params, init_count_G,
                 warmup_I_params, warmup_IR_params, init_count_I,
                 posterior_samples=1000, debug_disable_theta=False):
        """Covid Model 1.5

        Args:
            transition_window (int): J in our notation, the number of days to consider a
                possible transition to a more severe state
        """
        super(CovidModel, self).__init__()

        self.transition_window = transition_window
        self.vax_statuses = vax_statuses
        self.compartments = compartments
        self.posterior_samples = posterior_samples

        # create dictionaries to store model parameters / prior distributions
        self._initialize_parameters(T_serial, epsilon, delta,
                                    rho_M, lambda_M, nu_M,
                                    rho_G, lambda_G, nu_G,
                                    rho_I, lambda_I, nu_I,
                                    lambda_I_bar, nu_I_bar,
                                    rho_D, lambda_D, nu_D,
                                    lambda_D_bar, nu_D_bar,
                                    warmup_A_params,
                                    warmup_M_params,
                                    warmup_G_params, warmup_GR_params, init_count_G,
                                    warmup_I_params, warmup_IR_params, init_count_I,
                                    debug_disable_theta)

        self._initialize_priors(T_serial, epsilon, delta,
                                rho_M, lambda_M, nu_M,
                                rho_G, lambda_G, nu_G,
                                rho_I, lambda_I, nu_I,
                                lambda_I_bar, nu_I_bar,
                                rho_D, lambda_D, nu_D,
                                lambda_D_bar, nu_D_bar,
                                warmup_A_params,
                                warmup_M_params,
                                warmup_G_params, warmup_GR_params, init_count_G,
                                warmup_I_params, warmup_IR_params, init_count_I,)

    def call(self, r_t, debug_disable_prior=False, return_all=False):
        """Run covid model 1.5

        Args:
            r_t (tf.Tensor): A tuple of all input tensors we need. It should be, in order:
                rt should be size (1, days_to_forecast)
            debug_disable_prior (bool): If True, will disable adding the prior to the loss. Used to debug gradients
        Returns:
            tf.Tensor: A tensor size (1, days_to_forecast) of incident hospital admissions
        """

        self._constrain_parameters()
        self._sample_and_reparameterize()

        r_t = tf.squeeze(r_t)
        forecast_days = r_t.shape[-1]

        # It's a little weird to iteratively write to tensors one day at a time
        # To do this, we'll use TensorArray's, arrays of tensors where each element is a tensor representing
        # one compartment/vaccine status/ day
        # This helper function creates a nested dictionary keyed on:
        #  compartment->
        #      vaccinestatus->
        #           TensorArray with one tensor per day from:
        #               warmup_start to forecast_end for any quantities with warmup data
        #               forecast_start to forecast_end for the outcome, which does not have warmup data
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
                    today_asymp = (yesterday_asymp_no +
                                   self.epsilon_samples_constrained[Vax.total.value]*yesterday_asymp_yes) * \
                    self.delta_samples_constrained[Vax.yes.value]*r_t[day] ** (1/(self.T_serial_samples_constrained[Vax.total.value] + 1e-6))
                else:
                    today_asymp = (yesterday_asymp_no +
                                   self.epsilon_samples_constrained[Vax.total.value] * yesterday_asymp_yes) * \
                                   r_t[day] ** (1 / self.T_serial_samples_constrained[Vax.total.value])

                forecasted_fluxes[Comp.A.value][vax_status] = \
                    forecasted_fluxes[Comp.A.value][vax_status].write(day, today_asymp)

                if day == 0:
                    prev_count_G[vax_status] = self.init_count_G_samples_constrained[vax_status]
                    prev_count_I[vax_status] = self.init_count_I_samples_constrained[vax_status]

                for j in range(self.transition_window):

                    if day == 0 and j>0:
                        prev_count_G[vax_status] = prev_count_G[vax_status] + \
                                                   self.warmup_G_samples_constrained[vax_status][j] - \
                                                   self.warmup_GR_samples_constrained[vax_status][j]
                        prev_count_I[vax_status] = prev_count_I[vax_status] + \
                                                   self.warmup_I_samples_constrained[vax_status][j] - \
                                                   self.warmup_IR_samples_constrained[vax_status][j]

                    if day - j - 1 < 0:
                        j_ago_asymp = self.warmup_A_samples_constrained[vax_status][day-j-1]
                        j_ago_mild = self.warmup_M_samples_constrained[vax_status][day - j - 1]
                        j_ago_gen = self.warmup_G_samples_constrained[vax_status][day - j - 1]
                        j_ago_genrec = self.warmup_G_samples_constrained[vax_status][day - j - 1]
                        j_ago_icu = self.warmup_I_samples_constrained[vax_status][day - j - 1]
                        j_ago_icurec = self.warmup_I_samples_constrained[vax_status][day - j - 1]
                    else:
                        j_ago_asymp = forecasted_fluxes[Comp.A.value][vax_status].read(day-j-1)
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

                # Today's MG = sum of last J * rho * pi
                forecasted_fluxes[Comp.M.value][vax_status] = \
                    forecasted_fluxes[Comp.M.value][vax_status].write(day,
                                                                     tf.reduce_sum(
                                                                         previously_asymptomatic_tensor *
                                                                         self.rho_M_samples_constrained[vax_status] * self.pi_M_samples[vax_status],
                                                                         axis=0)
                                                                     )

                forecasted_fluxes[Comp.G.value][vax_status] = \
                    forecasted_fluxes[Comp.G.value][vax_status].write(day,
                                                                           tf.reduce_sum(
                                                                               previously_mild_tensor *
                                                                               self.rho_G_samples_constrained[vax_status] * self.pi_G_samples[vax_status],
                                                                               axis=0)
                                                                           )
                forecasted_fluxes[Comp.GR.value][vax_status] = \
                    forecasted_fluxes[Comp.GR.value][vax_status].write(day,
                                                                      tf.reduce_sum(
                                                                          previously_gen_tensor *
                                                                          (1-self.rho_I_samples_constrained[vax_status]) *
                                                                          self.pi_I_bar_samples[vax_status],
                                                                          axis=0)
                                                                      )

                forecasted_fluxes[Comp.I.value][vax_status] = \
                    forecasted_fluxes[Comp.I.value][vax_status].write(day,
                                                                      tf.reduce_sum(
                                                                          previously_gen_tensor *
                                                                          self.rho_I_samples_constrained[vax_status] *
                                                                          self.pi_I_samples[vax_status],
                                                                          axis=0)
                                                                      )

                forecasted_fluxes[Comp.IR.value][vax_status] = \
                    forecasted_fluxes[Comp.IR.value][vax_status].write(day,
                                                                      tf.reduce_sum(
                                                                          previously_icu_tensor *
                                                                          (1-self.rho_D_samples_constrained[vax_status]) *
                                                                          self.pi_D_bar_samples[vax_status],
                                                                          axis=0)
                                                                      )


                forecasted_fluxes[Comp.D.value][vax_status] = \
                    forecasted_fluxes[Comp.D.value][vax_status].write(day,
                                                                      tf.reduce_sum(
                                                                          previously_icu_tensor *
                                                                          self.rho_D_samples_constrained[vax_status] *
                                                                          self.pi_D_samples[vax_status],
                                                                          axis=0)
                                                                      )

                forecasted_counts[Comp.G.value][vax_status] =  \
                    forecasted_counts[Comp.G.value][vax_status].write(day,
                                                                      prev_count_G[vax_status] + \
                                                                      forecasted_fluxes[Comp.G.value][vax_status].read(day) - \
                                                                      forecasted_fluxes[Comp.GR.value][vax_status].read(day))

                forecasted_counts[Comp.I.value][vax_status] = \
                    forecasted_counts[Comp.I.value][vax_status].write(day,
                                                                      prev_count_I[vax_status] + \
                                                                      forecasted_fluxes[Comp.I.value][vax_status].read(
                                                                          day) - \
                                                                      forecasted_fluxes[Comp.IR.value][vax_status].read(
                                                                          day))

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

        # Tensorflow thinks we didn't use every array, so we gotta mark them as used
        # TODO: did i screw up?
        self._mark_arrays_used(forecasted_fluxes)

        return tf.expand_dims((result['G_count'], result['G_in'], result['I_count'], result['D_in']), axis=0)

    def _initialize_parameters(self, T_serial, epsilon, delta,
                                    rho_M, lambda_M, nu_M,
                                    rho_G, lambda_G, nu_G,
                                    rho_I, lambda_I, nu_I,
                                    lambda_I_bar, nu_I_bar,
                                    rho_D, lambda_D, nu_D,
                                    lambda_D_bar, nu_D_bar,
                                    warmup_A_params,
                                    warmup_M_params,
                               warmup_G_params, warmup_GR_params, init_count_G,
                               warmup_I_params, warmup_IR_params, init_count_I,
                               debug_disable_theta=False):
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

        # T_serial, Delta and epsilon dont vary by vaccination status
        self.unconstrained_T_serial = {}
        self.unconstrained_T_serial['loc'] = \
            tf.Variable(T_serial['posterior_init']['loc'], dtype=tf.float32,
                        name=f'T_serial_A_loc', trainable=train_theta)
        self.unconstrained_T_serial['scale'] = \
            tf.Variable(T_serial['posterior_init']['scale'], dtype=tf.float32,
                        name=f'T_serial_A_scale', trainable=train_theta)

        self.unconstrained_epsilon = {}
        self.unconstrained_epsilon['loc'] = \
            tf.Variable(epsilon['posterior_init']['loc'], dtype=tf.float32,
                        name=f'epsilon_A_loc', trainable=train_theta)
        self.unconstrained_epsilon['scale'] = \
            tf.Variable(epsilon['posterior_init']['scale'], dtype=tf.float32,
                        name=f'epsilon_A_scale', trainable=train_theta)

        self.unconstrained_delta = {}
        self.unconstrained_delta['loc'] = \
            tf.Variable(delta['posterior_init']['loc'], dtype=tf.float32,
                        name=f'delta_A_loc', trainable=train_theta)
        self.unconstrained_delta['scale'] = \
            tf.Variable(delta['posterior_init']['scale'], dtype=tf.float32,
                        name=f'delta_A_scale', trainable=train_theta)

        for vax_status in [status.value for status in self.vax_statuses]:

            self.unconstrained_rho_M[vax_status] = {}
            self.unconstrained_rho_M[vax_status]['loc'] = \
                tf.Variable(rho_M[vax_status]['posterior_init']['loc'], dtype=tf.float32,
                            name=f'rho_M_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_rho_M[vax_status]['scale'] = \
                tf.Variable(rho_M[vax_status]['posterior_init']['scale'], dtype=tf.float32,
                            name=f'rho_M_scale_{vax_status}', trainable=train_theta)

            self.unconstrained_rho_G[vax_status] = {}
            self.unconstrained_rho_G[vax_status]['loc'] = \
                tf.Variable(rho_G[vax_status]['posterior_init']['loc'], dtype=tf.float32,
                            name=f'rho_G_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_rho_G[vax_status]['scale'] = \
                tf.Variable(rho_G[vax_status]['posterior_init']['scale'], dtype=tf.float32,
                            name=f'rho_G_scale_{vax_status}', trainable=train_theta)

            self.unconstrained_rho_I[vax_status] = {}
            self.unconstrained_rho_I[vax_status]['loc'] = \
                tf.Variable(rho_I[vax_status]['posterior_init']['loc'], dtype=tf.float32,
                            name=f'rho_I_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_rho_I[vax_status]['scale'] = \
                tf.Variable(rho_I[vax_status]['posterior_init']['scale'], dtype=tf.float32,
                            name=f'rho_I_scale_{vax_status}', trainable=train_theta)

            self.unconstrained_rho_D[vax_status] = {}
            self.unconstrained_rho_D[vax_status]['loc'] = \
                tf.Variable(rho_D[vax_status]['posterior_init']['loc'], dtype=tf.float32,
                            name=f'rho_D_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_rho_D[vax_status]['scale'] = \
                tf.Variable(rho_D[vax_status]['posterior_init']['scale'], dtype=tf.float32,
                            name=f'rho_D_scale_{vax_status}', trainable=train_theta)

            self.unconstrained_lambda_M[vax_status] = {}
            self.unconstrained_lambda_M[vax_status]['loc'] = \
                tf.Variable(lambda_M[vax_status]['posterior_init']['loc'], dtype=tf.float32,
                            name=f'lambda_M_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_lambda_M[vax_status]['scale'] = \
                tf.Variable(lambda_M[vax_status]['posterior_init']['scale'], dtype=tf.float32,
                            name=f'lambda_M_scale_{vax_status}', trainable=train_theta)

            self.unconstrained_lambda_G[vax_status] = {}
            self.unconstrained_lambda_G[vax_status]['loc'] = \
                tf.Variable(lambda_G[vax_status]['posterior_init']['loc'], dtype=tf.float32,
                            name=f'lambda_G_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_lambda_G[vax_status]['scale'] = \
                tf.Variable(lambda_G[vax_status]['posterior_init']['scale'], dtype=tf.float32,
                            name=f'lambda_G_scale_{vax_status}', trainable=train_theta)

            self.unconstrained_lambda_I[vax_status] = {}
            self.unconstrained_lambda_I[vax_status]['loc'] = \
                tf.Variable(lambda_I[vax_status]['posterior_init']['loc'], dtype=tf.float32,
                            name=f'lambda_I_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_lambda_I[vax_status]['scale'] = \
                tf.Variable(lambda_I[vax_status]['posterior_init']['scale'], dtype=tf.float32,
                            name=f'lambda_I_scale_{vax_status}', trainable=train_theta)

            self.unconstrained_lambda_I_bar[vax_status] = {}
            self.unconstrained_lambda_I_bar[vax_status]['loc'] = \
                tf.Variable(lambda_I_bar[vax_status]['posterior_init']['loc'], dtype=tf.float32,
                            name=f'lambda_I_bar_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_lambda_I_bar[vax_status]['scale'] = \
                tf.Variable(lambda_I_bar[vax_status]['posterior_init']['scale'], dtype=tf.float32,
                            name=f'lambda_I_bar_scale_{vax_status}', trainable=train_theta)

            self.unconstrained_lambda_D[vax_status] = {}
            self.unconstrained_lambda_D[vax_status]['loc'] = \
                tf.Variable(lambda_D[vax_status]['posterior_init']['loc'], dtype=tf.float32,
                            name=f'lambda_D_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_lambda_D[vax_status]['scale'] = \
                tf.Variable(lambda_D[vax_status]['posterior_init']['scale'], dtype=tf.float32,
                            name=f'lambda_D_scale_{vax_status}', trainable=train_theta)

            self.unconstrained_lambda_D_bar[vax_status] = {}
            self.unconstrained_lambda_D_bar[vax_status]['loc'] = \
                tf.Variable(lambda_D_bar[vax_status]['posterior_init']['loc'], dtype=tf.float32,
                            name=f'lambda_D_bar_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_lambda_D_bar[vax_status]['scale'] = \
                tf.Variable(lambda_D_bar[vax_status]['posterior_init']['scale'], dtype=tf.float32,
                            name=f'lambda_D_bar_scale_{vax_status}', trainable=train_theta)

            self.unconstrained_nu_M[vax_status] = {}
            self.unconstrained_nu_M[vax_status]['loc'] = \
                tf.Variable(nu_M[vax_status]['posterior_init']['loc'], dtype=tf.float32,
                            name=f'nu_M_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_nu_M[vax_status]['scale'] = \
                tf.Variable(nu_M[vax_status]['posterior_init']['scale'], dtype=tf.float32,
                            name=f'nu_M_scale_{vax_status}', trainable=train_theta)

            self.unconstrained_nu_G[vax_status] = {}
            self.unconstrained_nu_G[vax_status]['loc'] = \
                tf.Variable(nu_G[vax_status]['posterior_init']['loc'], dtype=tf.float32,
                            name=f'nu_G_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_nu_G[vax_status]['scale'] = \
                tf.Variable(nu_G[vax_status]['posterior_init']['scale'], dtype=tf.float32,
                            name=f'nu_G_scale_{vax_status}', trainable=train_theta)

            self.unconstrained_nu_I[vax_status] = {}
            self.unconstrained_nu_I[vax_status]['loc'] = \
                tf.Variable(nu_I[vax_status]['posterior_init']['loc'], dtype=tf.float32,
                            name=f'nu_I_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_nu_I[vax_status]['scale'] = \
                tf.Variable(nu_I[vax_status]['posterior_init']['scale'], dtype=tf.float32,
                            name=f'nu_I_scale_{vax_status}', trainable=train_theta)

            self.unconstrained_nu_I_bar[vax_status] = {}
            self.unconstrained_nu_I_bar[vax_status]['loc'] = \
                tf.Variable(nu_I_bar[vax_status]['posterior_init']['loc'], dtype=tf.float32,
                            name=f'nu_I_bar_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_nu_I_bar[vax_status]['scale'] = \
                tf.Variable(nu_I_bar[vax_status]['posterior_init']['scale'], dtype=tf.float32,
                            name=f'nu_I_bar_scale_{vax_status}', trainable=train_theta)

            self.unconstrained_nu_D[vax_status] = {}
            self.unconstrained_nu_D[vax_status]['loc'] = \
                tf.Variable(nu_D[vax_status]['posterior_init']['loc'], dtype=tf.float32,
                            name=f'nu_D_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_nu_D[vax_status]['scale'] = \
                tf.Variable(nu_D[vax_status]['posterior_init']['scale'], dtype=tf.float32,
                            name=f'nu_D_scale_{vax_status}', trainable=train_theta)

            self.unconstrained_nu_D_bar[vax_status] = {}
            self.unconstrained_nu_D_bar[vax_status]['loc'] = \
                tf.Variable(nu_D_bar[vax_status]['posterior_init']['loc'], dtype=tf.float32,
                            name=f'nu_D_bar_loc_{vax_status}', trainable=train_theta)
            self.unconstrained_nu_D_bar[vax_status]['scale'] = \
                tf.Variable(nu_D_bar[vax_status]['posterior_init']['scale'], dtype=tf.float32,
                            name=f'nu_D_bar_scale_{vax_status}', trainable=train_theta)

            self.unconstrained_warmup_A_params[vax_status] = []
            self.unconstrained_warmup_M_params[vax_status] = []
            self.unconstrained_warmup_G_params[vax_status] = []
            self.unconstrained_warmup_GR_params[vax_status] = []
            self.unconstrained_init_count_G_params[vax_status] = {}
            self.unconstrained_warmup_I_params[vax_status] = []
            self.unconstrained_warmup_IR_params[vax_status] = []
            self.unconstrained_init_count_I_params[vax_status] = {}
            
            for day in range(self.transition_window):
                self.unconstrained_warmup_A_params[vax_status].append({})
                self.unconstrained_warmup_A_params[vax_status][day]['loc'] = \
                    tf.Variable(tf.cast(warmup_A_params[vax_status]['posterior_init'][day]['loc'],
                                        dtype=tf.float32), dtype=tf.float32,
                                name=f'warmup_A_loc_{day}_{vax_status}')
                self.unconstrained_warmup_A_params[vax_status][day]['scale'] = \
                    tf.Variable(tf.cast(warmup_A_params[vax_status]['posterior_init'][day]['scale'],
                                        dtype=tf.float32), dtype=tf.float32,
                                name=f'warmup_A_scale_{day}_{vax_status}')

                self.unconstrained_warmup_M_params[vax_status].append({})
                self.unconstrained_warmup_M_params[vax_status][day]['loc'] = \
                    tf.Variable(tf.cast(warmup_M_params[vax_status]['posterior_init'][day]['loc'],
                                        dtype=tf.float32), dtype=tf.float32,
                                name=f'warmup_M_loc_{day}_{vax_status}')
                self.unconstrained_warmup_M_params[vax_status][day]['scale'] = \
                    tf.Variable(tf.cast(warmup_M_params[vax_status]['posterior_init'][day]['scale'],
                                        dtype=tf.float32), dtype=tf.float32,
                                name=f'warmup_M_scale_{day}_{vax_status}')

                self.unconstrained_warmup_G_params[vax_status].append({})
                self.unconstrained_warmup_G_params[vax_status][day]['loc'] = \
                    tf.Variable(tf.cast(warmup_G_params[vax_status]['posterior_init'][day]['loc'],
                                        dtype=tf.float32), dtype=tf.float32,
                                name=f'warmup_G_loc_{day}_{vax_status}')
                self.unconstrained_warmup_G_params[vax_status][day]['scale'] = \
                    tf.Variable(tf.cast(warmup_G_params[vax_status]['posterior_init'][day]['scale'],
                                        dtype=tf.float32), dtype=tf.float32,
                                name=f'warmup_G_scale_{day}_{vax_status}')

                self.unconstrained_warmup_GR_params[vax_status].append({})
                self.unconstrained_warmup_GR_params[vax_status][day]['loc'] = \
                    tf.Variable(tf.cast(warmup_GR_params[vax_status]['posterior_init'][day]['loc'],
                                        dtype=tf.float32), dtype=tf.float32,
                                name=f'warmup_GR_loc_{day}_{vax_status}')
                self.unconstrained_warmup_GR_params[vax_status][day]['scale'] = \
                    tf.Variable(tf.cast(warmup_GR_params[vax_status]['posterior_init'][day]['scale'],
                                        dtype=tf.float32), dtype=tf.float32,
                                name=f'warmup_GR_scale_{day}_{vax_status}')

                self.unconstrained_warmup_I_params[vax_status].append({})
                self.unconstrained_warmup_I_params[vax_status][day]['loc'] = \
                    tf.Variable(tf.cast(warmup_I_params[vax_status]['posterior_init'][day]['loc'],
                                        dtype=tf.float32), dtype=tf.float32,
                                name=f'warmup_I_loc_{day}_{vax_status}')
                self.unconstrained_warmup_I_params[vax_status][day]['scale'] = \
                    tf.Variable(tf.cast(warmup_I_params[vax_status]['posterior_init'][day]['scale'],
                                        dtype=tf.float32), dtype=tf.float32,
                                name=f'warmup_I_scale_{day}_{vax_status}')

                self.unconstrained_warmup_IR_params[vax_status].append({})
                self.unconstrained_warmup_IR_params[vax_status][day]['loc'] = \
                    tf.Variable(tf.cast(warmup_IR_params[vax_status]['posterior_init'][day]['loc'],
                                        dtype=tf.float32), dtype=tf.float32,
                                name=f'warmup_IR_loc_{day}_{vax_status}')
                self.unconstrained_warmup_IR_params[vax_status][day]['scale'] = \
                    tf.Variable(tf.cast(warmup_IR_params[vax_status]['posterior_init'][day]['scale'],
                                        dtype=tf.float32), dtype=tf.float32,
                                name=f'warmup_IR_scale_{day}_{vax_status}')
                
            self.unconstrained_init_count_G_params[vax_status]['loc'] = \
                tf.Variable(tf.cast(init_count_G[vax_status]['posterior_init']['loc'],
                                    dtype=tf.float32), dtype=tf.float32,
                            name=f'init_count_G_loc_{vax_status}')
            self.unconstrained_init_count_G_params[vax_status]['scale'] = \
                tf.Variable(tf.cast(init_count_G[vax_status]['posterior_init']['scale'],
                                    dtype=tf.float32), dtype=tf.float32,
                            name=f'init_count_G_scale_{vax_status}')

            self.unconstrained_init_count_I_params[vax_status]['loc'] = \
                tf.Variable(tf.cast(init_count_I[vax_status]['posterior_init']['loc'],
                                    dtype=tf.float32), dtype=tf.float32,
                            name=f'init_count_I_loc_{vax_status}')
            self.unconstrained_init_count_I_params[vax_status]['scale'] = \
                tf.Variable(tf.cast(init_count_I[vax_status]['posterior_init']['scale'],
                                    dtype=tf.float32), dtype=tf.float32,
                            name=f'init_count_I_scale_{vax_status}')

            self.previously_asymptomatic[vax_status] = tf.TensorArray(tf.float32, size=self.transition_window,
                                                                      clear_after_read=False, name=f'prev_asymp')
            self.previously_mild[vax_status] = tf.TensorArray(tf.float32, size=self.transition_window,
                                                                      clear_after_read=False, name=f'prev_mild')
            self.previously_gen[vax_status] = tf.TensorArray(tf.float32, size=self.transition_window,
                                                              clear_after_read=False, name=f'prev_gen')
            self.previously_icu[vax_status] = tf.TensorArray(tf.float32, size=self.transition_window,
                                                              clear_after_read=False, name=f'prev_icu')

        return

    def _initialize_priors(self, T_serial, epsilon, delta,
                                rho_M, lambda_M, nu_M,
                                rho_G, lambda_G, nu_G,
                                rho_I, lambda_I, nu_I,
                                lambda_I_bar, nu_I_bar,
                                rho_D, lambda_D, nu_D,
                                lambda_D_bar, nu_D_bar,
                                warmup_A_params,
                                warmup_M_params,
                                warmup_G_params, warmup_GR_params, init_count_G,
                                warmup_I_params, warmup_IR_params, init_count_I):
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
                T_serial['prior']['loc'],
                T_serial['prior']['scale'],
                0, np.inf),
            bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
        )

        self.prior_distros[Comp.A.value][Vax.total.value]['epsilon'] = tfp.distributions.TransformedDistribution(
            tfp.distributions.Beta(
                epsilon['prior']['a'],
                epsilon['prior']['b']),
            bijector=tfp.bijectors.Invert(tfp.bijectors.Sigmoid())
        )

        self.prior_distros[Comp.A.value][Vax.yes.value]['delta'] = tfp.distributions.TransformedDistribution(
            tfp.distributions.Beta(
                delta['prior']['a'],
                delta['prior']['b']),
            bijector=tfp.bijectors.Invert(tfp.bijectors.Sigmoid())
        )

        # create prior distributions
        for vax_status in [status.value for status in self.vax_statuses]:

            self.prior_distros[Comp.M.value][vax_status]['rho_M'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.Beta(
                    rho_M[vax_status]['prior']['a'],
                    rho_M[vax_status]['prior']['b']),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Sigmoid())
            )

            self.prior_distros[Comp.G.value][vax_status]['rho_G'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.Beta(
                    rho_G[vax_status]['prior']['a'],
                    rho_G[vax_status]['prior']['b']),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Sigmoid())
            )
            self.prior_distros[Comp.I.value][vax_status]['rho_I'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.Beta(
                    rho_I[vax_status]['prior']['a'],
                    rho_I[vax_status]['prior']['b']),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Sigmoid())
            )

            self.prior_distros[Comp.D.value][vax_status]['rho_D'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.Beta(
                    rho_D[vax_status]['prior']['a'],
                    rho_D[vax_status]['prior']['b']),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Sigmoid())
            )

            #  must be positive
            self.prior_distros[Comp.M.value][vax_status]['lambda_M'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.TruncatedNormal(
                    lambda_M[vax_status]['prior']['loc'],
                    lambda_M[vax_status]['prior']['scale'],
                    0, np.inf),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
            )

            self.prior_distros[Comp.G.value][vax_status]['lambda_G'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.TruncatedNormal(
                    lambda_G[vax_status]['prior']['loc'],
                    lambda_G[vax_status]['prior']['scale'],
                    0, np.inf),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
            )
            
            self.prior_distros[Comp.I.value][vax_status]['lambda_I'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.TruncatedNormal(
                    lambda_I[vax_status]['prior']['loc'],
                    lambda_I[vax_status]['prior']['scale'],
                    0, np.inf),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
            )

            self.prior_distros[Comp.IR.value][vax_status]['lambda_I_bar'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.TruncatedNormal(
                    lambda_I_bar[vax_status]['prior']['loc'],
                    lambda_I_bar[vax_status]['prior']['scale'],
                    0, np.inf),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
            )
            
            self.prior_distros[Comp.D.value][vax_status]['lambda_D'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.TruncatedNormal(
                    lambda_D[vax_status]['prior']['loc'],
                    lambda_D[vax_status]['prior']['scale'],
                    0, np.inf),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
            )

            self.prior_distros[Comp.D.value][vax_status]['lambda_D_bar'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.TruncatedNormal(
                    lambda_D_bar[vax_status]['prior']['loc'],
                    lambda_D_bar[vax_status]['prior']['scale'],
                    0, np.inf),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
            )

            self.prior_distros[Comp.M.value][vax_status]['nu_M'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.TruncatedNormal(
                    nu_M[vax_status]['prior']['loc'],
                    nu_M[vax_status]['prior']['scale'],
                    0, np.inf),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
            )
            
            self.prior_distros[Comp.G.value][vax_status]['nu_G'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.TruncatedNormal(
                    nu_G[vax_status]['prior']['loc'],
                    nu_G[vax_status]['prior']['scale'],
                    0, np.inf),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
            )
            self.prior_distros[Comp.I.value][vax_status]['nu_I'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.TruncatedNormal(
                    nu_I[vax_status]['prior']['loc'],
                    nu_I[vax_status]['prior']['scale'],
                    0, np.inf),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
            )
            self.prior_distros[Comp.IR.value][vax_status]['nu_I_bar'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.TruncatedNormal(
                    nu_I_bar[vax_status]['prior']['loc'],
                    nu_I_bar[vax_status]['prior']['scale'],
                    0, np.inf),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
            )
            self.prior_distros[Comp.D.value][vax_status]['nu_D'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.TruncatedNormal(
                    nu_D[vax_status]['prior']['loc'],
                    nu_D[vax_status]['prior']['scale'],
                    0, np.inf),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
            )
            self.prior_distros[Comp.D.value][vax_status]['nu_D_bar'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.TruncatedNormal(
                    nu_D_bar[vax_status]['prior']['loc'],
                    nu_D_bar[vax_status]['prior']['scale'],
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
                            tf.cast(warmup_A_params[vax_status]['prior'][day]['loc'],dtype=tf.float32),
                            tf.cast(warmup_A_params[vax_status]['prior'][day]['scale'],dtype=tf.float32),
                            0, tf.float32.max),
                        bijector=tfp.bijectors.Invert(tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Scale(100)]))
                    )
                )
                self.prior_distros[Comp.M.value][vax_status]['warmup_M'].append(
                    tfp.distributions.TransformedDistribution(
                        tfp.distributions.TruncatedNormal(
                            tf.cast(warmup_M_params[vax_status]['prior'][day]['loc'], dtype=tf.float32),
                            tf.cast(warmup_M_params[vax_status]['prior'][day]['scale'], dtype=tf.float32),
                            0, tf.float32.max),
                        bijector=tfp.bijectors.Invert(tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Scale(100)]))
                    )
                )
                self.prior_distros[Comp.G.value][vax_status]['warmup_G'].append(
                    tfp.distributions.TransformedDistribution(
                        tfp.distributions.TruncatedNormal(
                            tf.cast(warmup_G_params[vax_status]['prior'][day]['loc'], dtype=tf.float32),
                            tf.cast(warmup_G_params[vax_status]['prior'][day]['scale'], dtype=tf.float32),
                            0, tf.float32.max),
                        bijector=tfp.bijectors.Invert(
                            tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Scale(100)]))
                    )
                )
                self.prior_distros[Comp.GR.value][vax_status]['warmup_GR'].append(
                    tfp.distributions.TransformedDistribution(
                        tfp.distributions.TruncatedNormal(
                            tf.cast(warmup_GR_params[vax_status]['prior'][day]['loc'], dtype=tf.float32),
                            tf.cast(warmup_GR_params[vax_status]['prior'][day]['scale'], dtype=tf.float32),
                            0, tf.float32.max),
                        bijector=tfp.bijectors.Invert(
                            tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Scale(100)]))
                    )
                )
                self.prior_distros[Comp.I.value][vax_status]['warmup_I'].append(
                    tfp.distributions.TransformedDistribution(
                        tfp.distributions.TruncatedNormal(
                            tf.cast(warmup_I_params[vax_status]['prior'][day]['loc'], dtype=tf.float32),
                            tf.cast(warmup_I_params[vax_status]['prior'][day]['scale'], dtype=tf.float32),
                            0, tf.float32.max),
                        bijector=tfp.bijectors.Invert(
                            tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Scale(100)]))
                    )
                )
                self.prior_distros[Comp.IR.value][vax_status]['warmup_IR'].append(
                    tfp.distributions.TransformedDistribution(
                        tfp.distributions.TruncatedNormal(
                            tf.cast(warmup_IR_params[vax_status]['prior'][day]['loc'], dtype=tf.float32),
                            tf.cast(warmup_IR_params[vax_status]['prior'][day]['scale'], dtype=tf.float32),
                            0, tf.float32.max),
                        bijector=tfp.bijectors.Invert(
                            tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Scale(100)]))
                    )
                )
            self.prior_distros[Comp.G.value][vax_status]['init_count_G'] = (
                tfp.distributions.TransformedDistribution(
                    tfp.distributions.TruncatedNormal(
                        tf.cast(init_count_G[vax_status]['prior']['loc'], dtype=tf.float32),
                        tf.cast(init_count_G[vax_status]['prior']['scale'], dtype=tf.float32),
                        0, tf.float32.max),
                    bijector=tfp.bijectors.Invert(
                        tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Scale(100)]))
                )
            )
            self.prior_distros[Comp.I.value][vax_status]['init_count_I'] = (
                tfp.distributions.TransformedDistribution(
                    tfp.distributions.TruncatedNormal(
                        tf.cast(init_count_I[vax_status]['prior']['loc'], dtype=tf.float32),
                        tf.cast(init_count_I[vax_status]['prior']['scale'], dtype=tf.float32),
                        0, tf.float32.max),
                    bijector=tfp.bijectors.Invert(
                        tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Scale(100)]))
                )
            )

        return

    def  _constrain_parameters(self):
        """Helper function to make sure all of our posterior variance parameters are positive"""

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
        self.T_serial_params[Vax.total.value]['scale'] = tf.math.softplus(self.unconstrained_T_serial['scale'])

        self.epsilon_params[Vax.total.value]['loc'] = self.unconstrained_epsilon['loc']
        self.epsilon_params[Vax.total.value]['scale'] = tf.math.softplus(self.unconstrained_epsilon['scale'])

        self.delta_params[Vax.yes.value]['loc'] = self.unconstrained_delta['loc']
        self.delta_params[Vax.yes.value]['scale'] = tf.math.softplus(self.unconstrained_delta['scale'])

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
            self.warmup_A_params[vax_status] = []
            self.warmup_M_params[vax_status] = []
            self.warmup_G_params[vax_status] = []
            self.warmup_GR_params[vax_status] = []
            self.init_count_G_params[vax_status] = {}
            self.warmup_I_params[vax_status] = []
            self.warmup_IR_params[vax_status] = []
            self.init_count_I_params[vax_status] = {}
            for day in range(self.transition_window):
                self.warmup_A_params[vax_status].append({})
                self.warmup_M_params[vax_status].append({})
                self.warmup_G_params[vax_status].append({})
                self.warmup_GR_params[vax_status].append({})
                self.warmup_I_params[vax_status].append({})
                self.warmup_IR_params[vax_status].append({})

            self.rho_M_params[vax_status]['loc'] = self.unconstrained_rho_M[vax_status]['loc']
            self.rho_M_params[vax_status]['scale'] = tf.math.softplus(self.unconstrained_rho_M[vax_status]['scale'])

            self.lambda_M_params[vax_status]['loc'] = self.unconstrained_lambda_M[vax_status]['loc']
            self.lambda_M_params[vax_status]['scale'] = tf.math.softplus(self.unconstrained_lambda_M[vax_status]['scale'])

            self.nu_M_params[vax_status]['loc'] = self.unconstrained_nu_M[vax_status]['loc']
            self.nu_M_params[vax_status]['scale'] = tf.math.softplus(self.unconstrained_nu_M[vax_status]['scale'])

            self.rho_G_params[vax_status]['loc'] = self.unconstrained_rho_G[vax_status]['loc']
            self.rho_G_params[vax_status]['scale'] = tf.math.softplus(self.unconstrained_rho_G[vax_status]['scale'])

            self.lambda_G_params[vax_status]['loc'] = self.unconstrained_lambda_G[vax_status]['loc']
            self.lambda_G_params[vax_status]['scale'] = tf.math.softplus(self.unconstrained_lambda_G[vax_status]['scale'])

            self.nu_G_params[vax_status]['loc'] = self.unconstrained_nu_G[vax_status]['loc']
            self.nu_G_params[vax_status]['scale'] = tf.math.softplus(self.unconstrained_nu_G[vax_status]['scale'])

            self.rho_I_params[vax_status]['loc'] = self.unconstrained_rho_I[vax_status]['loc']
            self.rho_I_params[vax_status]['scale'] = tf.math.softplus(self.unconstrained_rho_I[vax_status]['scale'])

            self.lambda_I_params[vax_status]['loc'] = self.unconstrained_lambda_I[vax_status]['loc']
            self.lambda_I_params[vax_status]['scale'] = tf.math.softplus(
                self.unconstrained_lambda_I[vax_status]['scale'])

            self.lambda_I_bar_params[vax_status]['loc'] = self.unconstrained_lambda_I_bar[vax_status]['loc']
            self.lambda_I_bar_params[vax_status]['scale'] = tf.math.softplus(
                self.unconstrained_lambda_I_bar[vax_status]['scale'])

            self.nu_I_params[vax_status]['loc'] = self.unconstrained_nu_I[vax_status]['loc']
            self.nu_I_params[vax_status]['scale'] = tf.math.softplus(self.unconstrained_nu_I[vax_status]['scale'])

            self.nu_I_bar_params[vax_status]['loc'] = self.unconstrained_nu_I_bar[vax_status]['loc']
            self.nu_I_bar_params[vax_status]['scale'] = tf.math.softplus(self.unconstrained_nu_I_bar[vax_status]['scale'])

            self.rho_D_params[vax_status]['loc'] = self.unconstrained_rho_D[vax_status]['loc']
            self.rho_D_params[vax_status]['scale'] = tf.math.softplus(self.unconstrained_rho_D[vax_status]['scale'])

            self.lambda_D_params[vax_status]['loc'] = self.unconstrained_lambda_D[vax_status]['loc']
            self.lambda_D_params[vax_status]['scale'] = tf.math.softplus(
                self.unconstrained_lambda_D[vax_status]['scale'])

            self.lambda_D_bar_params[vax_status]['loc'] = self.unconstrained_lambda_D_bar[vax_status]['loc']
            self.lambda_D_bar_params[vax_status]['scale'] = tf.math.softplus(
                self.unconstrained_lambda_D_bar[vax_status]['scale'])

            self.nu_D_params[vax_status]['loc'] = self.unconstrained_nu_D[vax_status]['loc']
            self.nu_D_params[vax_status]['scale'] = tf.math.softplus(self.unconstrained_nu_D[vax_status]['scale'])

            self.nu_D_bar_params[vax_status]['loc'] = self.unconstrained_nu_D_bar[vax_status]['loc']
            self.nu_D_bar_params[vax_status]['scale'] = tf.math.softplus(self.unconstrained_nu_D_bar[vax_status]['scale'])

            for day in range(self.transition_window):
                self.warmup_A_params[vax_status][day]['loc'] = \
                    self.unconstrained_warmup_A_params[vax_status][day]['loc']
                self.warmup_A_params[vax_status][day]['scale'] = \
                    tf.math.softplus(self.unconstrained_warmup_A_params[vax_status][day]['scale'])

                self.warmup_M_params[vax_status][day]['loc'] = \
                    self.unconstrained_warmup_M_params[vax_status][day]['loc']
                self.warmup_M_params[vax_status][day]['scale'] = \
                    tf.math.softplus(self.unconstrained_warmup_M_params[vax_status][day]['scale'])

                self.warmup_G_params[vax_status][day]['loc'] = \
                    self.unconstrained_warmup_G_params[vax_status][day]['loc']
                self.warmup_G_params[vax_status][day]['scale'] = \
                    tf.math.softplus(self.unconstrained_warmup_G_params[vax_status][day]['scale'])

                self.warmup_GR_params[vax_status][day]['loc'] = \
                    self.unconstrained_warmup_GR_params[vax_status][day]['loc']
                self.warmup_GR_params[vax_status][day]['scale'] = \
                    tf.math.softplus(self.unconstrained_warmup_GR_params[vax_status][day]['scale'])

                self.warmup_I_params[vax_status][day]['loc'] = \
                    self.unconstrained_warmup_I_params[vax_status][day]['loc']
                self.warmup_I_params[vax_status][day]['scale'] = \
                    tf.math.softplus(self.unconstrained_warmup_I_params[vax_status][day]['scale'])

                self.warmup_IR_params[vax_status][day]['loc'] = \
                    self.unconstrained_warmup_IR_params[vax_status][day]['loc']
                self.warmup_IR_params[vax_status][day]['scale'] = \
                    tf.math.softplus(self.unconstrained_warmup_IR_params[vax_status][day]['scale'])

            self.init_count_G_params[vax_status]['loc'] = \
                self.unconstrained_init_count_G_params[vax_status]['loc']
            self.init_count_G_params[vax_status]['scale'] = \
                tf.math.softplus(self.unconstrained_init_count_G_params[vax_status]['scale'])

            self.init_count_I_params[vax_status]['loc'] = \
                self.unconstrained_init_count_I_params[vax_status]['loc']
            self.init_count_I_params[vax_status]['scale'] = \
                tf.math.softplus(self.unconstrained_init_count_I_params[vax_status]['scale'])

        return

    def _sample_and_reparameterize(self):
        """Here we again constrain, and our prior distribution will fix it"""
        
        self.T_serial_samples = {}
        self.T_serial_samples_constrained = {}
        self.T_serial_probs = {}

        self.epsilon_samples = {}
        self.epsilon_samples_constrained = {}
        self.epsilon_probs = {}

        self.delta_samples = {}
        self.delta_samples_constrained = {}
        self.delta_probs = {}

        self.rho_M_samples = {}
        self.rho_M_samples_constrained = {}
        self.rho_M_probs = {}

        self.rho_G_samples = {}
        self.rho_G_samples_constrained = {}
        self.rho_G_probs = {}

        self.rho_I_samples = {}
        self.rho_I_samples_constrained = {}
        self.rho_I_probs = {}

        self.rho_D_samples = {}
        self.rho_D_samples_constrained = {}
        self.rho_D_probs = {}

        self.lambda_M_samples = {}
        self.lambda_M_samples_constrained = {}
        self.lambda_M_probs = {}

        self.lambda_G_samples = {}
        self.lambda_G_samples_constrained = {}
        self.lambda_G_probs = {}
        self.lambda_I_samples = {}
        self.lambda_I_samples_constrained = {}
        self.lambda_I_probs = {}
        self.lambda_I_bar_samples = {}
        self.lambda_I_bar_samples_constrained = {}
        self.lambda_I_bar_probs = {}

        self.lambda_D_samples = {}
        self.lambda_D_samples_constrained = {}
        self.lambda_D_probs = {}
        self.lambda_D_bar_samples = {}
        self.lambda_D_bar_samples_constrained = {}
        self.lambda_D_bar_probs = {}

        self.nu_M_samples = {}
        self.nu_M_samples_constrained = {}
        self.nu_M_probs = {}

        self.nu_G_samples = {}
        self.nu_G_samples_constrained = {}
        self.nu_G_probs = {}
        self.nu_I_samples = {}
        self.nu_I_samples_constrained = {}
        self.nu_I_probs = {}
        self.nu_I_bar_samples = {}
        self.nu_I_bar_samples_constrained = {}
        self.nu_I_bar_probs = {}
        self.nu_D_samples = {}
        self.nu_D_samples_constrained = {}
        self.nu_D_probs = {}
        self.nu_D_bar_samples = {}
        self.nu_D_bar_samples_constrained = {}
        self.nu_D_bar_probs = {}

        self.warmup_A_samples = {}
        self.warmup_A_samples_constrained = {}
        self.warmup_A_probs = {}

        self.warmup_M_samples = {}
        self.warmup_M_samples_constrained = {}
        self.warmup_M_probs = {}
        self.warmup_G_samples = {}
        self.warmup_G_samples_constrained = {}
        self.warmup_G_probs = {}
        self.warmup_GR_samples = {}
        self.warmup_GR_samples_constrained = {}
        self.warmup_GR_probs = {}
        self.warmup_I_samples = {}
        self.warmup_I_samples_constrained = {}
        self.warmup_I_probs = {}
        self.warmup_IR_samples = {}
        self.warmup_IR_samples_constrained = {}
        self.warmup_IR_probs = {}

        self.init_count_G_samples = {}
        self.init_count_G_samples_constrained = {}
        self.init_count_G_probs = {}
        self.init_count_I_samples = {}
        self.init_count_I_samples_constrained = {}
        self.init_count_I_probs = {}
        
        self.pi_M_samples = {}
        self.pi_G_samples = {}
        self.pi_I_samples = {}
        self.pi_I_bar_samples = {}
        self.pi_D_samples = {}
        self.pi_D_bar_samples = {}

        T_serial_noise = tf.random.normal((self.posterior_samples,))
        self.T_serial_samples[Vax.total.value] = self.T_serial_params[Vax.total.value]['loc'] + \
                                                self.T_serial_params[Vax.total.value]['scale'] * T_serial_noise

        # Constrain samples with softplus
        self.T_serial_samples_constrained[Vax.total.value] = tfp.bijectors.Softplus().forward(
            self.T_serial_samples[Vax.total.value])

        # Calulate variational posterior probability of un constrained samples
        T_serial_variational_posterior = tfp.distributions.Normal(self.T_serial_params[Vax.total.value]['loc'],
                                                                 self.T_serial_params[Vax.total.value]['scale'])

        self.T_serial_probs[Vax.total.value] = T_serial_variational_posterior.log_prob(
            self.T_serial_samples[Vax.total.value])

        epsilon_noise = tf.random.normal((self.posterior_samples,))
        self.epsilon_samples[Vax.total.value] = self.epsilon_params[Vax.total.value]['loc'] + \
                                            self.epsilon_params[Vax.total.value]['scale'] * epsilon_noise

        # Constrain samples with softplus
        self.epsilon_samples_constrained[Vax.total.value] = tfp.bijectors.Softplus().forward(
            self.epsilon_samples[Vax.total.value])

        # Calulate variational posterior probability of un constrained samples
        epsilon_variational_posterior = tfp.distributions.Normal(self.epsilon_params[Vax.total.value]['loc'],
                                                               self.epsilon_params[Vax.total.value]['scale'])

        self.epsilon_probs[Vax.total.value] = epsilon_variational_posterior.log_prob(self.epsilon_samples[Vax.total.value])

        delta_noise = tf.random.normal((self.posterior_samples,))
        # Use reparameterization trick to get unconstrained samples
        self.delta_samples[Vax.yes.value] = self.delta_params[Vax.yes.value]['loc'] + \
                                self.delta_params[Vax.yes.value]['scale'] * delta_noise

        # Constrain samples with softplus
        self.delta_samples_constrained[Vax.yes.value] = tfp.bijectors.Softplus().forward(self.delta_samples[Vax.yes.value])

        # Calulate variational posterior probability of un constrained samples
        delta_variational_posterior = tfp.distributions.Normal(self.delta_params[Vax.yes.value]['loc'],
                                                                  self.delta_params[Vax.yes.value]['scale'])

        self.delta_probs[Vax.yes.value] = delta_variational_posterior.log_prob(self.delta_samples[Vax.yes.value])


        for vax_status in [status.value for status in self.vax_statuses]:
    
            rho_M_noise = tf.random.normal((self.posterior_samples,))
            self.rho_M_samples[vax_status] = self.rho_M_params[vax_status]['loc'] + \
                                             self.rho_M_params[vax_status]['scale'] * rho_M_noise
            self.rho_M_samples_constrained[vax_status] = tfp.bijectors.Sigmoid().forward(self.rho_M_samples[vax_status])
    
            rho_M_variational_posterior = tfp.distributions.Normal(self.rho_M_params[vax_status]['loc'],
                                                                   self.rho_M_params[vax_status]['scale'])
    
            self.rho_M_probs[vax_status] = rho_M_variational_posterior.log_prob(self.rho_M_samples[vax_status])

            rho_G_noise = tf.random.normal((self.posterior_samples,))
            self.rho_G_samples[vax_status] = self.rho_G_params[vax_status]['loc'] + \
                                             self.rho_G_params[vax_status]['scale'] * rho_G_noise
            self.rho_G_samples_constrained[vax_status] = tfp.bijectors.Sigmoid().forward(self.rho_G_samples[vax_status])

            rho_G_variational_posterior = tfp.distributions.Normal(self.rho_G_params[vax_status]['loc'],
                                                                   self.rho_G_params[vax_status]['scale'])

            self.rho_G_probs[vax_status] = rho_G_variational_posterior.log_prob(self.rho_G_samples[vax_status])

            rho_I_noise = tf.random.normal((self.posterior_samples,))
            self.rho_I_samples[vax_status] = self.rho_I_params[vax_status]['loc'] + \
                                             self.rho_I_params[vax_status]['scale'] * rho_I_noise
            self.rho_I_samples_constrained[vax_status] = tfp.bijectors.Sigmoid().forward(self.rho_I_samples[vax_status])

            rho_I_variational_posterior = tfp.distributions.Normal(self.rho_I_params[vax_status]['loc'],
                                                                   self.rho_I_params[vax_status]['scale'])

            self.rho_I_probs[vax_status] = rho_I_variational_posterior.log_prob(self.rho_I_samples[vax_status])

            rho_D_noise = tf.random.normal((self.posterior_samples,))
            self.rho_D_samples[vax_status] = self.rho_D_params[vax_status]['loc'] + \
                                             self.rho_D_params[vax_status]['scale'] * rho_D_noise
            self.rho_D_samples_constrained[vax_status] = tfp.bijectors.Sigmoid().forward(self.rho_D_samples[vax_status])

            rho_D_variational_posterior = tfp.distributions.Normal(self.rho_D_params[vax_status]['loc'],
                                                                   self.rho_D_params[vax_status]['scale'])

            self.rho_D_probs[vax_status] = rho_D_variational_posterior.log_prob(self.rho_D_samples[vax_status])

            lambda_M_noise = tf.random.normal((self.posterior_samples,))
            self.lambda_M_samples[vax_status] = self.lambda_M_params[vax_status]['loc'] + \
                                             self.lambda_M_params[vax_status]['scale'] * lambda_M_noise
            self.lambda_M_samples_constrained[vax_status] = tfp.bijectors.Softplus().forward(self.lambda_M_samples[vax_status])

            lambda_M_variational_posterior = tfp.distributions.Normal(self.lambda_M_params[vax_status]['loc'],
                                                                   self.lambda_M_params[vax_status]['scale'])

            self.lambda_M_probs[vax_status] = lambda_M_variational_posterior.log_prob(self.lambda_M_samples[vax_status])

            lambda_G_noise = tf.random.normal((self.posterior_samples,))
            self.lambda_G_samples[vax_status] = self.lambda_G_params[vax_status]['loc'] + \
                                             self.lambda_G_params[vax_status]['scale'] * lambda_G_noise
            self.lambda_G_samples_constrained[vax_status] = tfp.bijectors.Softplus().forward(self.lambda_G_samples[vax_status])

            lambda_G_variational_posterior = tfp.distributions.Normal(self.lambda_G_params[vax_status]['loc'],
                                                                   self.lambda_G_params[vax_status]['scale'])

            self.lambda_G_probs[vax_status] = lambda_G_variational_posterior.log_prob(self.lambda_G_samples[vax_status])

            lambda_I_noise = tf.random.normal((self.posterior_samples,))
            self.lambda_I_samples[vax_status] = self.lambda_I_params[vax_status]['loc'] + \
                                                self.lambda_I_params[vax_status]['scale'] * lambda_I_noise
            self.lambda_I_samples_constrained[vax_status] = tfp.bijectors.Softplus().forward(
                self.lambda_I_samples[vax_status])

            lambda_I_variational_posterior = tfp.distributions.Normal(self.lambda_I_params[vax_status]['loc'],
                                                                      self.lambda_I_params[vax_status]['scale'])

            self.lambda_I_probs[vax_status] = lambda_I_variational_posterior.log_prob(self.lambda_I_samples[vax_status])

            lambda_I_bar_noise = tf.random.normal((self.posterior_samples,))
            self.lambda_I_bar_samples[vax_status] = self.lambda_I_bar_params[vax_status]['loc'] + \
                                                self.lambda_I_bar_params[vax_status]['scale'] * lambda_I_bar_noise
            self.lambda_I_bar_samples_constrained[vax_status] = tfp.bijectors.Softplus().forward(
                self.lambda_I_bar_samples[vax_status])

            lambda_I_bar_variational_posterior = tfp.distributions.Normal(self.lambda_I_bar_params[vax_status]['loc'],
                                                                      self.lambda_I_bar_params[vax_status]['scale'])

            self.lambda_I_bar_probs[vax_status] = lambda_I_bar_variational_posterior.log_prob(self.lambda_I_bar_samples[vax_status])

            lambda_D_noise = tf.random.normal((self.posterior_samples,))
            self.lambda_D_samples[vax_status] = self.lambda_D_params[vax_status]['loc'] + \
                                                self.lambda_D_params[vax_status]['scale'] * lambda_D_noise
            self.lambda_D_samples_constrained[vax_status] = tfp.bijectors.Softplus().forward(
                self.lambda_D_samples[vax_status])

            lambda_D_variational_posterior = tfp.distributions.Normal(self.lambda_D_params[vax_status]['loc'],
                                                                      self.lambda_D_params[vax_status]['scale'])

            self.lambda_D_probs[vax_status] = lambda_D_variational_posterior.log_prob(self.lambda_D_samples[vax_status])

            lambda_D_bar_noise = tf.random.normal((self.posterior_samples,))
            self.lambda_D_bar_samples[vax_status] = self.lambda_D_bar_params[vax_status]['loc'] + \
                                                self.lambda_D_bar_params[vax_status]['scale'] * lambda_D_bar_noise
            self.lambda_D_bar_samples_constrained[vax_status] = tfp.bijectors.Softplus().forward(
                self.lambda_D_bar_samples[vax_status])

            lambda_D_bar_variational_posterior = tfp.distributions.Normal(self.lambda_D_bar_params[vax_status]['loc'],
                                                                      self.lambda_D_bar_params[vax_status]['scale'])

            self.lambda_D_bar_probs[vax_status] = lambda_D_bar_variational_posterior.log_prob(self.lambda_D_bar_samples[vax_status])

            nu_M_noise = tf.random.normal((self.posterior_samples,))
            self.nu_M_samples[vax_status] = self.nu_M_params[vax_status]['loc'] + \
                                             self.nu_M_params[vax_status]['scale'] * nu_M_noise
            self.nu_M_samples_constrained[vax_status] = tfp.bijectors.Softplus().forward(self.nu_M_samples[vax_status])

            nu_M_variational_posterior = tfp.distributions.Normal(self.nu_M_params[vax_status]['loc'],
                                                                   self.nu_M_params[vax_status]['scale'])

            self.nu_M_probs[vax_status] = nu_M_variational_posterior.log_prob(self.nu_M_samples[vax_status])

            nu_G_noise = tf.random.normal((self.posterior_samples,))
            self.nu_G_samples[vax_status] = self.nu_G_params[vax_status]['loc'] + \
                                             self.nu_G_params[vax_status]['scale'] * nu_G_noise
            self.nu_G_samples_constrained[vax_status] = tfp.bijectors.Softplus().forward(self.nu_G_samples[vax_status])

            nu_G_variational_posterior = tfp.distributions.Normal(self.nu_G_params[vax_status]['loc'],
                                                                   self.nu_G_params[vax_status]['scale'])

            self.nu_G_probs[vax_status] = nu_G_variational_posterior.log_prob(self.nu_G_samples[vax_status])

            nu_I_noise = tf.random.normal((self.posterior_samples,))
            self.nu_I_samples[vax_status] = self.nu_I_params[vax_status]['loc'] + \
                                            self.nu_I_params[vax_status]['scale'] * nu_I_noise
            self.nu_I_samples_constrained[vax_status] = tfp.bijectors.Softplus().forward(self.nu_I_samples[vax_status])

            nu_I_variational_posterior = tfp.distributions.Normal(self.nu_I_params[vax_status]['loc'],
                                                                  self.nu_I_params[vax_status]['scale'])

            self.nu_I_probs[vax_status] = nu_I_variational_posterior.log_prob(self.nu_I_samples[vax_status])

            nu_I_bar_noise = tf.random.normal((self.posterior_samples,))
            self.nu_I_bar_samples[vax_status] = self.nu_I_bar_params[vax_status]['loc'] + \
                                            self.nu_I_bar_params[vax_status]['scale'] * nu_I_bar_noise
            self.nu_I_bar_samples_constrained[vax_status] = tfp.bijectors.Softplus().forward(self.nu_I_bar_samples[vax_status])

            nu_I_bar_variational_posterior = tfp.distributions.Normal(self.nu_I_bar_params[vax_status]['loc'],
                                                                  self.nu_I_bar_params[vax_status]['scale'])

            self.nu_I_bar_probs[vax_status] = nu_I_bar_variational_posterior.log_prob(self.nu_I_bar_samples[vax_status])

            nu_D_noise = tf.random.normal((self.posterior_samples,))
            self.nu_D_samples[vax_status] = self.nu_D_params[vax_status]['loc'] + \
                                            self.nu_D_params[vax_status]['scale'] * nu_D_noise
            self.nu_D_samples_constrained[vax_status] = tfp.bijectors.Softplus().forward(self.nu_D_samples[vax_status])

            nu_D_variational_posterior = tfp.distributions.Normal(self.nu_D_params[vax_status]['loc'],
                                                                  self.nu_D_params[vax_status]['scale'])

            self.nu_D_probs[vax_status] = nu_D_variational_posterior.log_prob(self.nu_D_samples[vax_status])

            nu_D_bar_noise = tf.random.normal((self.posterior_samples,))
            self.nu_D_bar_samples[vax_status] = self.nu_D_bar_params[vax_status]['loc'] + \
                                            self.nu_D_bar_params[vax_status]['scale'] * nu_D_bar_noise
            self.nu_D_bar_samples_constrained[vax_status] = tfp.bijectors.Softplus().forward(self.nu_D_bar_samples[vax_status])

            nu_D_bar_variational_posterior = tfp.distributions.Normal(self.nu_D_bar_params[vax_status]['loc'],
                                                                  self.nu_D_bar_params[vax_status]['scale'])

            self.nu_D_bar_probs[vax_status] = nu_D_bar_variational_posterior.log_prob(self.nu_D_bar_samples[vax_status])

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
            
            for day in range(self.transition_window):
                warmup_A_noise = tf.random.normal((self.posterior_samples,))
                self.warmup_A_samples[vax_status].append(self.warmup_A_params[vax_status][day]['loc'] +
                                             self.warmup_A_params[vax_status][day]['scale'] *
                                             warmup_A_noise)
                self.warmup_A_samples_constrained[vax_status].append(tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Scale(100)]).forward(self.warmup_A_samples[vax_status][-1]))
    
                warmup_A_variational_posterior = tfp.distributions.Normal(self.warmup_A_params[vax_status][day]['loc'],
                                                                          self.warmup_A_params[vax_status][day]['scale'])

                self.warmup_A_probs[vax_status].append(warmup_A_variational_posterior.log_prob(self.warmup_A_samples[vax_status][-1]))
    
                warmup_M_noise = tf.random.normal((self.posterior_samples,))
                self.warmup_M_samples[vax_status].append(self.warmup_M_params[vax_status][day]['loc'] +
                                             self.warmup_M_params[vax_status][day]['scale'] *
                                             warmup_M_noise)
                self.warmup_M_samples_constrained[vax_status].append(tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Scale(100)]).forward(self.warmup_M_samples[vax_status][-1]))
    
                warmup_M_variational_posterior = tfp.distributions.Normal(self.warmup_M_params[vax_status][day]['loc'],
                                                                          self.warmup_M_params[vax_status][day][
                                                                              'scale'])
    
                self.warmup_M_probs[vax_status].append(warmup_M_variational_posterior.log_prob(self.warmup_M_samples[vax_status][-1]))

                warmup_G_noise = tf.random.normal((self.posterior_samples,))
                self.warmup_G_samples[vax_status].append(self.warmup_G_params[vax_status][day]['loc'] +
                                                         self.warmup_G_params[vax_status][day]['scale'] *
                                                         warmup_G_noise)
                self.warmup_G_samples_constrained[vax_status].append(
                    tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Scale(100)]).forward(
                        self.warmup_G_samples[vax_status][-1]))

                warmup_G_variational_posterior = tfp.distributions.Normal(self.warmup_G_params[vax_status][day]['loc'],
                                                                          self.warmup_G_params[vax_status][day][
                                                                              'scale'])

                self.warmup_G_probs[vax_status].append(
                    warmup_G_variational_posterior.log_prob(self.warmup_G_samples[vax_status][-1]))

                warmup_GR_noise = tf.random.normal((self.posterior_samples,))
                self.warmup_GR_samples[vax_status].append(self.warmup_GR_params[vax_status][day]['loc'] +
                                                         self.warmup_GR_params[vax_status][day]['scale'] *
                                                         warmup_GR_noise)
                self.warmup_GR_samples_constrained[vax_status].append(
                    tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Scale(100)]).forward(
                        self.warmup_GR_samples[vax_status][-1]))

                warmup_GR_variational_posterior = tfp.distributions.Normal(self.warmup_GR_params[vax_status][day]['loc'],
                                                                          self.warmup_GR_params[vax_status][day][
                                                                              'scale'])

                self.warmup_GR_probs[vax_status].append(
                    warmup_GR_variational_posterior.log_prob(self.warmup_GR_samples[vax_status][-1]))

                warmup_I_noise = tf.random.normal((self.posterior_samples,))
                self.warmup_I_samples[vax_status].append(self.warmup_I_params[vax_status][day]['loc'] +
                                                         self.warmup_I_params[vax_status][day]['scale'] *
                                                         warmup_I_noise)
                self.warmup_I_samples_constrained[vax_status].append(
                    tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Scale(100)]).forward(
                        self.warmup_I_samples[vax_status][-1]))

                warmup_I_variational_posterior = tfp.distributions.Normal(self.warmup_I_params[vax_status][day]['loc'],
                                                                          self.warmup_I_params[vax_status][day][
                                                                              'scale'])

                self.warmup_I_probs[vax_status].append(
                    warmup_I_variational_posterior.log_prob(self.warmup_I_samples[vax_status][-1]))

                warmup_IR_noise = tf.random.normal((self.posterior_samples,))
                self.warmup_IR_samples[vax_status].append(self.warmup_IR_params[vax_status][day]['loc'] +
                                                         self.warmup_IR_params[vax_status][day]['scale'] *
                                                         warmup_IR_noise)
                self.warmup_IR_samples_constrained[vax_status].append(
                    tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Scale(100)]).forward(
                        self.warmup_IR_samples[vax_status][-1]))

                warmup_IR_variational_posterior = tfp.distributions.Normal(self.warmup_IR_params[vax_status][day]['loc'],
                                                                          self.warmup_IR_params[vax_status][day][
                                                                              'scale'])

                self.warmup_IR_probs[vax_status].append(
                    warmup_IR_variational_posterior.log_prob(self.warmup_IR_samples[vax_status][-1]))

            init_count_G_noise = tf.random.normal((self.posterior_samples,))
            self.init_count_G_samples[vax_status] = (self.init_count_G_params[vax_status]['loc'] +
                                                     self.init_count_G_params[vax_status]['scale'] *
                                                     init_count_G_noise)
            self.init_count_G_samples_constrained[vax_status] = (
                tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Scale(100)]).forward(
                    self.init_count_G_samples[vax_status][-1]))

            init_count_G_variational_posterior = tfp.distributions.Normal(self.init_count_G_params[vax_status]['loc'],
                                                                      self.init_count_G_params[vax_status][
                                                                          'scale'])

            self.init_count_G_probs[vax_status] = (
                init_count_G_variational_posterior.log_prob(self.init_count_G_samples[vax_status][-1]))

            init_count_I_noise = tf.random.normal((self.posterior_samples,))
            self.init_count_I_samples[vax_status] = (self.init_count_I_params[vax_status]['loc'] +
                                                     self.init_count_I_params[vax_status]['scale'] *
                                                     init_count_I_noise)
            self.init_count_I_samples_constrained[vax_status] = (
                tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Scale(100)]).forward(
                    self.init_count_I_samples[vax_status][-1]))

            init_count_I_variational_posterior = tfp.distributions.Normal(self.init_count_I_params[vax_status]['loc'],
                                                                      self.init_count_I_params[vax_status][
                                                                          'scale'])

            self.init_count_I_probs[vax_status] = (
                init_count_I_variational_posterior.log_prob(self.init_count_I_samples[vax_status][-1]))


    
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
                self.init_count_G_samples_constrained[status.value]) + \
            tfp.bijectors.Chain([tfp.bijectors.Softplus(), 
                                 tfp.bijectors.Scale(100)]).forward_log_det_jacobian(
                self.init_count_G_samples[status.value]) for status in
            self.vax_statuses]
        init_count_G_posterior_probs = self.init_count_G_probs
        self.add_loss(lambda: tf.reduce_sum([-tf.reduce_sum(
            tf.reduce_mean(init_count_G_prior_probs[status.value] - init_count_G_posterior_probs[status.value], axis=-1))
                                             for status in self.vax_statuses]))

        init_count_I_prior_probs = [
            self.prior_distros[Comp.I.value][status.value]['init_count_I'].log_prob(
                self.init_count_I_samples_constrained[status.value]) + \
            tfp.bijectors.Chain([tfp.bijectors.Softplus(),
                                 tfp.bijectors.Scale(100)]).forward_log_det_jacobian(
                self.init_count_I_samples[status.value]) for status in
            self.vax_statuses]
        init_count_I_posterior_probs = self.init_count_I_probs
        self.add_loss(lambda: tf.reduce_sum([-tf.reduce_sum(
            tf.reduce_mean(init_count_I_prior_probs[status.value] - init_count_I_posterior_probs[status.value],
                           axis=-1))
                                             for status in self.vax_statuses]))

        # open bug about adding loss inisde a for loop: https://github.com/tensorflow/tensorflow/issues/44590
        self.add_loss(lambda:  tf.reduce_sum([tf.reduce_sum([-tf.reduce_sum(tf.reduce_mean(
                self.prior_distros[Comp.A.value][status.value]['warmup_A'][day].log_prob(
                    self.warmup_A_samples[status.value][day]) + \
                tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Scale(100)]).forward_log_det_jacobian(self.warmup_A_samples[status.value][day])
                - self.warmup_A_probs[status.value][day],axis=-1)) for day in range(self.transition_window)])for status in self.vax_statuses]))

        self.add_loss(lambda: tf.reduce_sum([tf.reduce_sum([-tf.reduce_sum(tf.reduce_mean(
            self.prior_distros[Comp.M.value][status.value]['warmup_M'][day].log_prob(
                self.warmup_M_samples[status.value][day]) + \
            tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Scale(100)]).forward_log_det_jacobian(self.warmup_M_samples[status.value][day])
            - self.warmup_M_probs[status.value][day], axis=-1)) for day in range(self.transition_window)]) for status in
                                             self.vax_statuses]))
        self.add_loss(lambda: tf.reduce_sum([tf.reduce_sum([-tf.reduce_sum(tf.reduce_mean(
            self.prior_distros[Comp.G.value][status.value]['warmup_G'][day].log_prob(
                self.warmup_G_samples[status.value][day]) + \
            tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Scale(100)]).forward_log_det_jacobian(
                self.warmup_G_samples[status.value][day])
            - self.warmup_G_probs[status.value][day], axis=-1)) for day in range(self.transition_window)]) for status in
                                             self.vax_statuses]))
        self.add_loss(lambda: tf.reduce_sum([tf.reduce_sum([-tf.reduce_sum(tf.reduce_mean(
            self.prior_distros[Comp.GR.value][status.value]['warmup_GR'][day].log_prob(
                self.warmup_GR_samples[status.value][day]) + \
            tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Scale(100)]).forward_log_det_jacobian(
                self.warmup_GR_samples[status.value][day])
            - self.warmup_GR_probs[status.value][day], axis=-1)) for day in range(self.transition_window)]) for status in
                                             self.vax_statuses]))
        self.add_loss(lambda: tf.reduce_sum([tf.reduce_sum([-tf.reduce_sum(tf.reduce_mean(
            self.prior_distros[Comp.I.value][status.value]['warmup_I'][day].log_prob(
                self.warmup_I_samples[status.value][day]) + \
            tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Scale(100)]).forward_log_det_jacobian(
                self.warmup_I_samples[status.value][day])
            - self.warmup_I_probs[status.value][day], axis=-1)) for day in range(self.transition_window)]) for status in
                                             self.vax_statuses]))
        self.add_loss(lambda: tf.reduce_sum([tf.reduce_sum([-tf.reduce_sum(tf.reduce_mean(
            self.prior_distros[Comp.IR.value][status.value]['warmup_IR'][day].log_prob(
                self.warmup_IR_samples[status.value][day]) + \
            tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Scale(100)]).forward_log_det_jacobian(
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


# Custom LogPoisson Probability Loss function
def calc_poisson(inputs):
    true_rate, predicted_rate = inputs
    poisson = tfp.distributions.Poisson(rate=true_rate)
    return poisson.log_prob(predicted_rate)


class LogPoissonProb(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
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

        log_probs_G_count = tf.map_fn(calc_poisson, (y_t['G_count'], y_p['G_count']),
                                      fn_output_signature=tf.float32)
        log_probs_G_in = tf.map_fn(calc_poisson, (tf.squeeze(y_t['G_in']), y_p['G_in']),
                                      fn_output_signature=tf.float32)
        log_probs_I_count = tf.map_fn(calc_poisson, (y_t['I_count'], y_p['I_count']),
                                      fn_output_signature=tf.float32)
        log_probs_D_in = tf.map_fn(calc_poisson, (tf.squeeze(y_t['D_in']), y_p['D_in']),
                                   fn_output_signature=tf.float32)

        # return negative log likielihood
        return -tf.reduce_sum(tf.reduce_mean(log_probs_G_count,axis=1)) + \
               -tf.reduce_sum(tf.reduce_mean(log_probs_G_in, axis=1)) + \
               -tf.reduce_sum(tf.reduce_mean(log_probs_I_count, axis=1)) + \
               -tf.reduce_sum(tf.reduce_mean(log_probs_D_in, axis=1))


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

            for day in range(len(self.model.unconstrained_warmup_A_params[vax_status])):
                tf.summary.scalar(f'warmup_A_-{-len(self.model.unconstrained_warmup_A_params[vax_status])+day}_mean_{vax_status}', data=tf.squeeze(tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Scale(100)]).forward(self.model.unconstrained_warmup_A_params[vax_status][day]['loc'])), step=epoch)
                tf.summary.scalar(f'warmup_A_-{-len(self.model.unconstrained_warmup_A_params[vax_status]) + day}_scale_{vax_status}',
                                  data=tf.squeeze(tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Scale(100)]).forward(self.model.unconstrained_warmup_A_params[vax_status][day]['scale'])), step=epoch)
                tf.summary.scalar(f'warmup_M_-{-len(self.model.unconstrained_warmup_M_params[vax_status]) + day}_mean_{vax_status}',
                                  data=tf.squeeze(tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Scale(100)]).forward(self.model.unconstrained_warmup_M_params[vax_status][day]['loc'])),
                                  step=epoch)
                tf.summary.scalar(f'warmup_M_-{-len(self.model.unconstrained_warmup_M_params[vax_status]) + day}_scale_{vax_status}',
                                  data=tf.squeeze(tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Scale(100)]).forward(
                                      self.model.unconstrained_warmup_M_params[vax_status][day]['scale'])), step=epoch)

        return

def get_logging_callbacks(log_dir):
    """Get tensorflow callbacks to write tensorboard logs to given log_dir"""
    file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    file_writer.set_as_default()
    logging_callback = VarLogCallback()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    return [logging_callback, tensorboard_callback]

