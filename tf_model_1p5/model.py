from enum import Enum

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense

import tensorflow_probability as tfp
from scipy.stats import beta, truncnorm

class Compartments(Enum):
    asymp = 0
    mild = 1
    extreme = 2
    general_ward = 3


class CovidModel(tf.keras.Model):

    def __init__(self, transition_window, T_serial,
                 alpha_bar_M, beta_bar_M, alpha_bar_X, beta_bar_X, alpha_bar_G, beta_bar_G,
                 lambda_bar_M, sigma_bar_M, lambda_bar_X, sigma_bar_X, lambda_bar_G, sigma_bar_G,
                 nu_bar_M, tau_bar_M, nu_bar_X, tau_bar_X, nu_bar_G, tau_bar_G):
        """Covid Model 1.5

        Args:
            transition_window (int): J in our notation, the number of days to consider a
                possible transition to a more severe state
            T_serial (float): CovidEstim infection hyperparameter
            alpha_bar_M (float): A positive shape hyperparameter controlling the Beta distribution prior on rho_M,
                the likelihood that an Asymptomatic individual progresses to the Mild state
            beta_bar_M (float): The second positive shape hyperparameter controlling the Beta distribution prior on rho_M,
                the likelihood that an Asymptomatic individual progresses to the Mild state
            alpha_bar_X (float): A positive shape hyperparameter controlling the Beta distribution prior on rho_X,
                the likelihood that an individual with Mild symptoms progresses to the eXtreme state
            beta_bar_X (float): The second positive shape hyperparameter controlling the Beta distribution prior on rho_X,
                the likelihood that an individual with Mild symptoms progresses to the eXtreme state
            alpha_bar_G (float): A positive shape hyperparameter controlling the Beta distribution prior on rho_G,
                the likelihood that an individual with eXtreme symptoms progresses to the hospital
            beta_bar_G (float): The second positive shape hyperparameter controlling the Beta distribution prior on rho_G,
                the likelihood that an Asymptomatic individual progresses to the Mild state
            lambda_bar_M (float): The mean of a strictly-positive Normal distribution prior over lambda_M, the rate
                of the Poisson distribution that governs how quickly individuals transition from Asymptomatic to Mild
            sigma_bar_M (float): The standard deviation of a strictly-positive Normal distribution prior over lambda_M,
                the rate of the Poisson distribution that governs how quickly individuals transition from Asymptomatic to Mild
            lambda_bar_X (float): The mean of a strictly-positive Normal distribution prior over lambda_X, the rate
                of the Poisson distribution that governs how quickly individuals transition from Mild to eXtreme
            sigma_bar_X (float): The standard deviation of a strictly-positive Normal distribution prior over lambda_X,
                the rate of the Poisson distribution that governs how quickly individuals transition from Mild to eXtreme
            lambda_bar_G (float): The mean of a strictly-positive Normal distribution prior over lambda_G, the rate
                of the Poisson distribution that governs how quickly individuals transition from eXtreme to the General ward
            sigma_bar_G (float): The standard deviation of a strictly-positive Normal distribution prior over lambda_G,
                the rate of the Poisson distribution that governs how quickly individuals transition from eXtreme to the General Ward
            nu_bar_M (float): The mean of a strictly-positive Normal distribution prior over nu_M,
                which scales the poisson PMF used to determine progression to the next state
            tau_bar_M (float): The standard deviation of a strictly-positive Normal distribution prior over nu_M,
                which scales the poisson PMF used to determine progression to the next state
            nu_bar_X (float): The mean of a strictly-positive Normal distribution prior over nu_X,
                which scales the poisson PMF used to determine progression to the next state
            tau_bar_X (float): The standard deviation of a strictly-positive Normal distribution prior over nu_X,
                which scales the poisson PMF used to determine progression to the next state
            nu_bar_G (float): The mean of a strictly-positive Normal distribution prior over nu_G,
                which scales the poisson PMF used to determine progression to the next state
            tau_bar_G (float): The standard deviation of a strictly-positive Normal distribution prior over nu_G,
                which scales the poisson PMF used to determine progression to the next state
        """
        super(CovidModel, self).__init__()

        self.transition_window = transition_window
        self.T_serial = T_serial
        self.compartments = Compartments

        # create dictionaries to store model parameters / prior distributions
        self._initialize_parameters(lambda_bar_M, lambda_bar_X, lambda_bar_G,
                                    nu_bar_M, nu_bar_X, nu_bar_G)
        self._initialize_priors(alpha_bar_M, beta_bar_M, alpha_bar_X, beta_bar_X, alpha_bar_G, beta_bar_G,
                                lambda_bar_M, sigma_bar_M, lambda_bar_X, sigma_bar_X, lambda_bar_G, sigma_bar_G,
                                nu_bar_M, tau_bar_M, nu_bar_X, tau_bar_X, nu_bar_G, tau_bar_G)

    def call(self, inputs, debug_disable_prior=False, return_all=False):
        """Run covid model 1.5

        Args:
            inputs (tuple(tf.Tensor)): A tuple of all input tensors we need. It should be, in order:
                (rt,
                 warmup_asymp_not_vaxxed, warmup_asymp_vaxxed,
                 warmup_mild_not_vaxxed, warmup_mild_vaxxed,
                 warmup_extreme_not_vaxxed, warmup_extreme_vaxxed)
                rt should be size (1, days_to_forecast), while
                warmup data should be size (1, days_of_warmup)
            debug_disable_prior (bool): If True, will disable adding the prior to the loss. Used to debug gradients
        Returns:
            tf.Tensor: A tensor size (1, days_to_forecast) of incident hospital admissions
        """

        # Tensorflow models are typically a single tensor or a tuple of multiple tensors
        # This function accepts all the input tensors we need (r_t, warmup for AMX, both vaxxed and non-vaxxed)
        # and returns r_t, along with dictionaries keyed on vaccination status for easier use.
        r_t, warmup_asymp, warmup_mild, warmup_extreme = self._parse_inputs(inputs)

        # We need to know how long the warmup data is and how long to forecast for
        # Take the last dimension one of the warmup data tensors
        # Any will do, so we arbitrariliy pick the vax_status=0 asymptomatic
        warmup_days_val = warmup_asymp[0].shape[-1]
        # take the last dimension of r_t
        forecast_days_val = r_t.shape[-1]

        # It's a little weird to iteratively write to tensors one day at a time
        # To do this, we'll use TensorArray's, arrays of tensors where each element is a tensor representing
        # one compartment/vaccine status/ day
        # This helper function creates a nested dictionary keyed on:
        #  compartment->
        #      vaccinestatus->
        #           TensorArray with one tensor per day from:
        #               warmup_start to forecast_end for any quantities with warmup data
        #               forecast_start to forecast_end for the outcome, which does not have warmup data
        forecasted_fluxes = self._initialize_flux_arrays(warmup_asymp, warmup_mild, warmup_extreme,
                                                         warmup_days_val, forecast_days_val)

        # Our model parameters have several constraints (being between 0-1, being positive)
        # So we need to transform them from the unconstrained space they are modeled in
        # This call transforms them and saves them properties of this model object (self):
        #     epsilon, a scalar with value between 0-1
        #     delta, a dictionary of scalars keyed on vaccination status with values between 0-1
        #     rho_M, rho_X, rho_G: dictionaries of tensors keyed on vaccination status with values between 0-1
        #     lambda_M,  lambda_X, lambda_G: dictionaries of positive-valued tensors keyed on vaccination status
        #     nu_M, nu_X, nu_G: dictionaries of positive-valued tensors keyed on vaccination status
        #     poisson_M, poisson_X, poisson_G: dictionaries of probability distribution objects keyed on vaccine status
        #     pi_M, pi_X, pi_G: dictionaries of tensor arrays with 1 element for each of the past J days
        #     previously_asymptomatic, previously_mild, previously_extreme: dictionaries of tensor arrays with 1 element for each of the past J days

        self._constrain_parameters()

        if not debug_disable_prior:
            self._add_prior_loss()

        # forecast from the end of warmup to the end of forecasting
        for day in range(warmup_days_val, forecast_days_val + warmup_days_val):

            # Start with asymptomatic
            asymp_t_1_no_vax = forecasted_fluxes[Compartments.asymp.value][0].read(day - 1)
            asymp_t_1_vax = forecasted_fluxes[Compartments.asymp.value][1].read(day - 1)
            combined_asymp_covariate = asymp_t_1_no_vax + self.epsilon * asymp_t_1_vax

            for vax_status in range(2):

                forecasted_fluxes[Compartments.asymp.value][vax_status] = forecasted_fluxes[Compartments.asymp.value][vax_status].write(day,
                                                                                                  tf.squeeze(
                                                                                                      combined_asymp_covariate *
                                                                                                      self.delta[
                                                                                                          vax_status] *
                                                                                                      r_t[
                                                                                                          day - warmup_days_val] ** (
                                                                                                                  1 / self.T_serial))
                                                                                                  )

                # get last J days of AMX
                for j in range(self.transition_window):

                    self.previously_asymptomatic[vax_status] = \
                        self.previously_asymptomatic[vax_status].write(j,
                            forecasted_fluxes[Compartments.asymp.value][vax_status].read(day - (j + 1))
                        )

                    self.previously_mild[vax_status] = \
                        self.previously_mild[vax_status].write(j,
                            forecasted_fluxes[Compartments.mild.value][ vax_status].read( day - (j + 1))
                        )
                    self.previously_extreme[vax_status] = \
                        self.previously_extreme[vax_status].write(j,
                            forecasted_fluxes[Compartments.extreme.value][vax_status].read(day - (j + 1))
                        )

                previously_asymptomatic_tensor = self.previously_asymptomatic[vax_status].stack()
                previously_mild_tensor = self.previously_mild[vax_status].stack()
                previously_extreme_tensor = self.previously_extreme[vax_status].stack()

                # Today's AMX = sum of last J * rho * pi
                forecasted_fluxes[Compartments.mild.value][vax_status] = \
                    forecasted_fluxes[Compartments.mild.value][vax_status].write(day,
                        tf.reduce_sum(previously_asymptomatic_tensor *self.rho_M[vax_status] *self.pi_M[vax_status],
                                      axis=0)
                        )

                forecasted_fluxes[Compartments.extreme.value][vax_status] = \
                    forecasted_fluxes[Compartments.extreme.value][vax_status].write(day,
                        tf.reduce_sum(previously_mild_tensor * self.rho_X[vax_status] * self.pi_X[vax_status],
                                      axis=0)
                )

                # G has no warmup, day 0 = first day of training
                forecasted_fluxes[Compartments.general_ward.value][vax_status] = \
                    forecasted_fluxes[Compartments.general_ward.value][vax_status].write(day - warmup_days_val,
                        tf.reduce_sum(previously_extreme_tensor * self.rho_G[vax_status] * self.pi_G[vax_status],
                                      axis=0)
                    )

        # Re-combine vaccinated and unvaxxed for our output
        if return_all:
            result = forecasted_fluxes
        else:
            result = forecasted_fluxes[Compartments.general_ward.value][0].stack() + \
                     forecasted_fluxes[Compartments.general_ward.value][1].stack()

        # Tensorflow thinks we didn't use every array, so we gotta mark them as used
        # TODO: did i screw up?
        self._mark_arrays_used(forecasted_fluxes)

        return result

    def _initialize_parameters(self, lambda_bar_M, lambda_bar_X, lambda_bar_G,
                               nu_bar_M, nu_bar_X, nu_bar_G):
        """Helper function to hide the book-keeping behind initializing model parameters

        TODO: Replace with better initializations
        """

        self.model_params = {}

        for enum_c in self.compartments:
            compartment = enum_c.value
            self.model_params[compartment] = {}
            for vax_status in [0, 1]:
                self.model_params[compartment][vax_status] = {}

        self.unconstrained_eps = self.add_weight(shape=tf.TensorShape(1), initializer="random_normal", trainable=True,
                                                 name=f'eps')
        self.unconstrained_delta = {}

        # Fix delta = 1 for non-vax
        self.unconstrained_delta[0] = tf.Variable([1e8], name='A_delta_0', trainable=False)
        # TODO: debug
        # if this isn't add weight the gradient doesn't get tracked for some reason
        self.unconstrained_delta[1] = self.add_weight(shape=tf.TensorShape(1),
                                                      initializer=tf.keras.initializers.RandomNormal(mean=-10),
                                                      trainable=True, name=f'A_delta_1')

        # initialize model parameters
        for vax_status in [0, 1]:
            self.model_params[Compartments.mild.value][vax_status]['unconstrained_rho'] = tf.Variable([5], name=f'M_rho_{vax_status}',
                                                                                   trainable=True, dtype=tf.float32,
                                                                                   shape=tf.TensorShape(1))
            self.model_params[Compartments.mild.value][vax_status]['unconstrained_lambda'] = tf.Variable([lambda_bar_M],
                                                                                      name=f'M_lambda_{vax_status}',
                                                                                      trainable=True, dtype=tf.float32,
                                                                                      shape=tf.TensorShape(1))
            self.model_params[Compartments.mild.value][vax_status]['unconstrained_nu'] = tf.Variable([nu_bar_M], name=f'M_nu_{vax_status}',
                                                                                  trainable=True, dtype=tf.float32,
                                                                                  shape=tf.TensorShape(1))
            self.model_params[Compartments.extreme.value][vax_status]['unconstrained_rho'] = tf.Variable([-5], name=f'X_rho_{vax_status}',
                                                                                      trainable=True, dtype=tf.float32,
                                                                                      shape=tf.TensorShape(1))
            self.model_params[Compartments.extreme.value][vax_status]['unconstrained_lambda'] = tf.Variable([lambda_bar_X],
                                                                                         name=f'X_lambda_{vax_status}',
                                                                                         trainable=True,
                                                                                         dtype=tf.float32,
                                                                                         shape=tf.TensorShape(1))
            self.model_params[Compartments.extreme.value][vax_status]['unconstrained_nu'] = tf.Variable([nu_bar_X],
                                                                                     name=f'X_nu_{vax_status}',
                                                                                     trainable=True, dtype=tf.float32,
                                                                                     shape=tf.TensorShape(1))
            self.model_params[Compartments.general_ward.value][vax_status]['unconstrained_rho'] = tf.Variable([-5],
                                                                                           name=f'G_rho_{vax_status}',
                                                                                           trainable=True,
                                                                                           dtype=tf.float32,
                                                                                           shape=tf.TensorShape(1))
            self.model_params[Compartments.general_ward.value][vax_status]['unconstrained_lambda'] = tf.Variable([lambda_bar_G],
                                                                                              name=f'G_lambda_{vax_status}',
                                                                                              trainable=True,
                                                                                              dtype=tf.float32,
                                                                                              shape=tf.TensorShape(1))
            self.model_params[Compartments.general_ward.value][vax_status]['unconstrained_nu'] = tf.Variable([nu_bar_G],
                                                                                          name=f'G_nu_{vax_status}',
                                                                                          trainable=True,
                                                                                          dtype=tf.float32,
                                                                                          shape=tf.TensorShape(1))

        return

    def _initialize_priors(self, alpha_bar_M, beta_bar_M, alpha_bar_X, beta_bar_X, alpha_bar_G, beta_bar_G,
                           lambda_bar_M, sigma_bar_M, lambda_bar_X, sigma_bar_X, lambda_bar_G, sigma_bar_G,
                           nu_bar_M, tau_bar_M, nu_bar_X, tau_bar_X, nu_bar_G, tau_bar_G,
                           eps_a=1.5, eps_b=1.5, delta_a=1.05, delta_b=20):
        """Helper function to hide the book-keeping behind initializing model priors"""

        self.prior_distros = {}
        for enum_c in self.compartments:
            compartment = enum_c.value
            self.prior_distros[compartment] = {}
            for vax_status in [0, 1]:
                self.prior_distros[compartment][vax_status] = {}

        # Tensorflow doesn't support dictionaries with mixed key types, so we'll give these a compartment and vax status
        self.prior_distros[Compartments.asymp.value][0]['epsilon'] = tfp.distributions.Beta(eps_a, eps_b)
        self.prior_distros[Compartments.asymp.value][1]['delta_1'] = tfp.distributions.Beta(delta_a, delta_b)

        # create prior distributions
        for vax_status in [0, 1]:
            self.prior_distros[Compartments.mild.value][vax_status]['rho'] = tfp.distributions.Beta(alpha_bar_M, beta_bar_M)
            self.prior_distros[Compartments.extreme.value][vax_status]['rho'] = tfp.distributions.Beta(alpha_bar_X, beta_bar_X)
            self.prior_distros[Compartments.general_ward.value][vax_status]['rho'] = tfp.distributions.Beta(alpha_bar_G, beta_bar_G)

            # We want these to be positive so we use a truncated normal with range 0-100
            self.prior_distros[Compartments.mild.value][vax_status]['lambda'] = tfp.distributions.TruncatedNormal(lambda_bar_M,
                                                                                               sigma_bar_M, 0, 20)
            self.prior_distros[Compartments.extreme.value][vax_status]['lambda'] = tfp.distributions.TruncatedNormal(lambda_bar_X,
                                                                                                  sigma_bar_X, 0, 20)
            self.prior_distros[Compartments.general_ward.value][vax_status]['lambda'] = tfp.distributions.TruncatedNormal(lambda_bar_G,
                                                                                                       sigma_bar_G, 0,
                                                                                                       20)

            self.prior_distros[Compartments.mild.value][vax_status]['nu'] = tfp.distributions.TruncatedNormal(nu_bar_M, tau_bar_M, 0, 1000)
            self.prior_distros[Compartments.extreme.value][vax_status]['nu'] = tfp.distributions.TruncatedNormal(nu_bar_X, tau_bar_X, 0,
                                                                                              20)
            self.prior_distros[Compartments.general_ward.value][vax_status]['nu'] = tfp.distributions.Normal(nu_bar_G, tau_bar_G, 0, 1000)

        return

    def _parse_inputs(self, inputs):
        """Helper function to hide the logic in parsing the big mess of input tensors we get"""
        (r_t,
         warmup_asymp_not_vaxxed, warmup_asymp_vaxxed,
         warmup_mild_not_vaxxed, warmup_mild_vaxxed,
         warmup_extreme_not_vaxxed, warmup_extreme_vaxxed) = inputs

        r_t = tf.squeeze(r_t)

        warmup_asymp = {}
        warmup_asymp[0] = tf.squeeze(warmup_asymp_not_vaxxed)
        warmup_asymp[1] = tf.squeeze(warmup_asymp_vaxxed)

        warmup_mild = {}
        warmup_mild[0] = tf.squeeze(warmup_mild_not_vaxxed)
        warmup_mild[1] = tf.squeeze(warmup_mild_vaxxed)

        warmup_extreme = {}
        warmup_extreme[0] = tf.squeeze(warmup_extreme_not_vaxxed)
        warmup_extreme[1] = tf.squeeze(warmup_extreme_vaxxed)

        return r_t, warmup_asymp, warmup_mild, warmup_extreme

    def _initialize_flux_arrays(self, warmup_asymp, warmup_mild, warmup_extreme,
                                warmup_days, forecast_days):
        """Helper function to hide the plumbing in creating TensorArrays for every output

        Args:
            warmup_days (int): Number of days of warmup data
            forecast_days (int): Number of days to forecast
        Returns
            dict{int:dict{int:TensorArray}}: Nested dictionary keyed on compartment->vaccine status,
                containing a TensorArray for every quantity
        """

        forecasted_fluxes = {}

        for enum_c in self.compartments:
            compartment = enum_c.value
            forecasted_fluxes[compartment] = {}

            # No need to store warmup data for the outcome
            if compartment == Compartments.general_ward.value:
                array_size = forecast_days
            else:
                array_size = warmup_days + forecast_days

            for vax_status in range(2):
                forecasted_fluxes[compartment][vax_status] = tf.TensorArray(tf.float32, size=array_size,
                                                                            clear_after_read=False,
                                                                            name=f'{compartment}_{vax_status}')

        # Write the warmup data to the array so we don't have to look in two places:
        for day in range(warmup_days):
            for vax_status in range(2):
                forecasted_fluxes[Compartments.asymp.value][vax_status] = \
                    forecasted_fluxes[Compartments.asymp.value][vax_status].write(day,
                                                               warmup_asymp[vax_status][day])
                forecasted_fluxes[Compartments.mild.value][vax_status] = \
                    forecasted_fluxes[Compartments.mild.value][vax_status].write(day,
                                                              warmup_mild[vax_status][day])
                forecasted_fluxes[Compartments.extreme.value][vax_status] = \
                    forecasted_fluxes[Compartments.extreme.value][vax_status].write(day,
                                                                 warmup_extreme[vax_status][day])

        return forecasted_fluxes

    def _constrain_parameters(self):
        """Helper function to hide the plumbing of creating the constrained parameters and other model TensorArrays"""

        # Must be 0-1
        self.epsilon = tf.math.sigmoid(self.unconstrained_eps)

        self.delta = {}
        for vax_status in range(2):
            self.delta[vax_status] = tf.squeeze(tf.math.sigmoid(self.unconstrained_delta[vax_status]))

        # Initialize dictionaries keyed on vax status
        self.rho_M = {}
        self.rho_X = {}
        self.rho_G = {}
        self.lambda_M = {}
        self.lambda_X = {}
        self.lambda_G = {}
        self.nu_M = {}
        self.nu_X = {}
        self.nu_G = {}
        self.poisson_M = {}
        self.poisson_X = {}
        self.poisson_G = {}
        self.pi_M = {}
        self.pi_X = {}
        self.pi_G = {}

        self.previously_asymptomatic = {}
        self.previously_mild = {}
        self.previously_extreme = {}

        for vax_status in range(2):

            # Rho must be 0-1, so use sigmoid
            # lambda and nu must be positive, so use softplus
            self.rho_M[vax_status] = tf.squeeze(
                tf.math.sigmoid(self.model_params[Compartments.mild.value][vax_status]['unconstrained_rho']))
            self.lambda_M[vax_status] = tf.squeeze(
                tf.math.softplus(self.model_params[Compartments.mild.value][vax_status]['unconstrained_lambda']))
            self.nu_M[vax_status] = tf.squeeze(
                tf.math.softplus(self.model_params[Compartments.mild.value][vax_status]['unconstrained_nu']))

            self.rho_X[vax_status] = tf.squeeze(
                tf.math.sigmoid(self.model_params[Compartments.extreme.value][vax_status]['unconstrained_rho']))
            self.lambda_X[vax_status] = tf.squeeze(
                tf.math.softplus(self.model_params[Compartments.extreme.value][vax_status]['unconstrained_lambda']))
            self.nu_X[vax_status] = tf.squeeze(
                tf.math.softplus(self.model_params[Compartments.extreme.value][vax_status]['unconstrained_nu']))

            self.rho_G[vax_status] = tf.squeeze(
                tf.math.sigmoid(self.model_params[Compartments.general_ward.value][vax_status]['unconstrained_rho']))
            self.lambda_G[vax_status] = tf.squeeze(
                tf.math.softplus(self.model_params[Compartments.general_ward.value][vax_status]['unconstrained_lambda']))
            self.nu_G[vax_status] = tf.squeeze(
                tf.math.softplus(self.model_params[Compartments.general_ward.value][vax_status]['unconstrained_nu']))

            # Create the distributions for each compartment
            self.poisson_M[vax_status] = tfp.distributions.Poisson(rate=self.lambda_M[vax_status])
            self.poisson_X[vax_status] = tfp.distributions.Poisson(rate=self.lambda_X[vax_status])
            self.poisson_G[vax_status] = tfp.distributions.Poisson(rate=self.lambda_G[vax_status])

            # pi is fixed while we forecast so we can create that now
            self.pi_M[vax_status] = tf.TensorArray(tf.float32, size=self.transition_window, clear_after_read=False,
                                                   name='self.pi_M')
            self.pi_X[vax_status] = tf.TensorArray(tf.float32, size=self.transition_window, clear_after_read=False,
                                                   name='self.pi_X')
            self.pi_G[vax_status] = tf.TensorArray(tf.float32, size=self.transition_window, clear_after_read=False,
                                                   name='self.pi_G')
            for j in range(self.transition_window):
                self.pi_M[vax_status] = self.pi_M[vax_status].write(j, self.poisson_M[vax_status].log_prob(j + 1) /
                                                                    self.nu_M[vax_status])
                self.pi_X[vax_status] = self.pi_X[vax_status].write(j, self.poisson_X[vax_status].log_prob(j + 1) /
                                                                    self.nu_X[vax_status])
                self.pi_G[vax_status] = self.pi_G[vax_status].write(j, self.poisson_G[vax_status].log_prob(j + 1) /
                                                                    self.nu_G[vax_status])

            # stacking the TensorArray makes it a tensor again
            self.pi_M[vax_status] = tf.transpose(self.pi_M[vax_status].stack())
            # Softmax so it sums to 1
            self.pi_M[vax_status] = tf.nn.softmax(self.pi_M[vax_status])

            self.pi_X[vax_status] = tf.transpose(self.pi_X[vax_status].stack())
            self.pi_X[vax_status] = tf.nn.softmax(self.pi_X[vax_status])
            self.pi_G[vax_status] = tf.transpose(self.pi_G[vax_status].stack())
            self.pi_G[vax_status] = tf.nn.softmax(self.pi_G[vax_status])

            # Initialize tensor arrays for storing these values
            self.previously_asymptomatic[vax_status] = tf.TensorArray(tf.float32, size=self.transition_window,
                                                                      clear_after_read=False, name=f'prev_asymp')
            self.previously_mild[vax_status] = tf.TensorArray(tf.float32, size=self.transition_window,
                                                              clear_after_read=False, name=f'prev_mild')
            self.previously_extreme[vax_status] = tf.TensorArray(tf.float32, size=self.transition_window,
                                                                 clear_after_read=False, name=f'prev_extreme')

    def _add_prior_loss(self):
        """Helper function for adding loss from model prior"""
        # only 1 epsilon
        # Not sure why only this one needs to get squeezed
        self.add_loss(tf.squeeze(-self.prior_distros[Compartments.asymp.value][0]['epsilon'].log_prob(self.epsilon)))
        # delta_0 is fixed
        self.add_loss(-self.prior_distros[Compartments.asymp.value][1]['delta_1'].log_prob(self.delta[1]))

        for vax_status in range(2):
            # Add losses from param priors
            # Make everything negative because we're minimizing
            self.add_loss(-self.prior_distros[Compartments.mild.value][vax_status]['rho'].log_prob(self.rho_M[vax_status]))
            self.add_loss(-self.prior_distros[Compartments.extreme.value][vax_status]['rho'].log_prob(self.rho_X[vax_status]))
            self.add_loss(-self.prior_distros[Compartments.general_ward.value][vax_status]['rho'].log_prob(self.rho_G[vax_status]))

            self.add_loss(-self.prior_distros[Compartments.mild.value][vax_status]['lambda'].log_prob(self.lambda_M[vax_status]))
            self.add_loss(-self.prior_distros[Compartments.extreme.value][vax_status]['lambda'].log_prob(self.lambda_X[vax_status]))
            self.add_loss(-self.prior_distros[Compartments.general_ward.value][vax_status]['lambda'].log_prob(self.lambda_G[vax_status]))

            self.add_loss(-self.prior_distros[Compartments.mild.value][vax_status]['nu'].log_prob(self.nu_M[vax_status]))
            self.add_loss(-self.prior_distros[Compartments.extreme.value][vax_status]['nu'].log_prob(self.nu_X[vax_status]))
            self.add_loss(-self.prior_distros[Compartments.general_ward.value][vax_status]['nu'].log_prob(self.nu_G[vax_status]))

    def _mark_arrays_used(self, forecasted_fluxes):
        """Helper function that supresses noisy error about not using all arrays"""
        for vax_status in range(2):
            forecasted_fluxes[Compartments.asymp.value][vax_status].mark_used()
            forecasted_fluxes[Compartments.mild.value][vax_status].mark_used()
            forecasted_fluxes[Compartments.extreme.value][vax_status].mark_used()
            forecasted_fluxes[Compartments.general_ward.value][vax_status].mark_used()
            self.previously_asymptomatic[vax_status].mark_used()
            self.previously_mild[vax_status].mark_used()
            self.previously_extreme[vax_status].mark_used()

        return


# Custom LogPoisson Probability Loss function
def calc_poisson(inputs):
    true_rate, predicted_rate = inputs
    poisson = tfp.distributions.Poisson(rate=true_rate)
    return poisson.log_prob(predicted_rate)


class LogPoissonProb(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        log_probs = tf.map_fn(calc_poisson, (tf.squeeze(y_true), y_pred), fn_output_signature=tf.float32)
        # return negative log likielihood
        return -tf.reduce_sum(log_probs)


class VarLogCallback(tf.keras.callbacks.Callback):
    """Logs all our model parameters"""

    def __init__(self, every_nth_epoch=1):
        self.every_nth_epoch = every_nth_epoch

    def on_epoch_end(self, epoch, logs):

        if epoch % self.every_nth_epoch != 0:
            return

        for vax_status in [0, 1]:
            tf.summary.scalar(f'rho_M_{vax_status}', data=tf.squeeze(self.model.rho_M[vax_status]), step=epoch)
            tf.summary.scalar(f'rho_X_{vax_status}', data=tf.squeeze(self.model.rho_X[vax_status]), step=epoch)
            tf.summary.scalar(f'rho_G_{vax_status}', data=tf.squeeze(self.model.rho_G[vax_status]), step=epoch)

            tf.summary.scalar(f'lambda_M_{vax_status}', data=tf.squeeze(self.model.lambda_M[vax_status]), step=epoch)
            tf.summary.scalar(f'lambda_X_{vax_status}', data=tf.squeeze(self.model.lambda_X[vax_status]), step=epoch)
            tf.summary.scalar(f'lambda_G_{vax_status}', data=tf.squeeze(self.model.lambda_G[vax_status]), step=epoch)
            tf.summary.scalar(f'nu_M_{vax_status}', data=tf.squeeze(self.model.nu_M[vax_status]), step=epoch)
            tf.summary.scalar(f'nu_X_{vax_status}', data=tf.squeeze(self.model.nu_X[vax_status]), step=epoch)
            tf.summary.scalar(f'nu_G_{vax_status}', data=tf.squeeze(self.model.nu_G[vax_status]), step=epoch)

            tf.summary.scalar(f'delta_{vax_status}', data=tf.squeeze(self.model.delta[vax_status]), step=epoch)

        tf.summary.scalar(f'eps', data=tf.squeeze(self.model.epsilon), step=epoch)

        return

def get_logging_callbacks(log_dir):
    """Get tensorflow callbacks to write tensorboard logs to given log_dir"""
    logging_callback = VarLogCallback()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    return [logging_callback, tensorboard_callback]

