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
    # X = 2
    # G = 3


class Vax(Enum):
    total = -1
    no = 0
    yes = 1


class CovidModel(tf.keras.Model):

    def __init__(self,
                 vax_statuses, compartments,
                 transition_window, T_serial, rho_M, lambda_M, nu_M,
                 warmup_A_params, posterior_samples=1000):
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
        self._initialize_parameters(T_serial, rho_M, lambda_M, nu_M,
                                    warmup_A_params)

        self._initialize_priors(T_serial, rho_M, lambda_M, nu_M,
                                warmup_A_params)

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



        for day in range(forecast_days):

            if  day-1 <0:
                yesterday_asymp = self.warmup_A_samples_constrained[day-1]
            else:
                yesterday_asymp = forecasted_fluxes[Comp.A.value][Vax.total.value].read(day-1)

            today_asymp = yesterday_asymp*r_t[day] ** (1/self.T_serial_samples_constrained)

            forecasted_fluxes[Comp.A.value][Vax.total.value] = \
                forecasted_fluxes[Comp.A.value][Vax.total.value].write(day, today_asymp)

            for j in range(self.transition_window):

                if day - j - 1 < 0:
                    j_ago_asymp = self.warmup_A_samples_constrained[day-j-1]
                else:
                    j_ago_asymp = forecasted_fluxes[Comp.A.value][Vax.total.value].read(day-j-1)

                self.previously_asymptomatic[Vax.total.value] = \
                    self.previously_asymptomatic[Vax.total.value].write(j, j_ago_asymp)

            previously_asymptomatic_tensor = self.previously_asymptomatic[Vax.total.value].stack()

            # Today's AMX = sum of last J * rho * pi
            forecasted_fluxes[Comp.M.value][Vax.total.value] = \
                forecasted_fluxes[Comp.M.value][Vax.total.value].write(day,
                                                                             tf.reduce_sum(
                                                                                 previously_asymptomatic_tensor *
                                                                                 self.rho_M_samples_constrained * self.pi_M_samples,
                                                                                 axis=0)
                                                                             )

        if not debug_disable_prior:

            self._callable_losses.clear()
            self._add_prior_loss()




        # Re-combine vaccinated and unvaxxed for our output
        if return_all:
            result = forecasted_fluxes
        else:
            result = forecasted_fluxes[Comp.M.value][Vax.total.value]

        # Tensorflow thinks we didn't use every array, so we gotta mark them as used
        # TODO: did i screw up?
        self._mark_arrays_used(forecasted_fluxes)

        return result.stack()

    def _initialize_parameters(self, T_serial, rho_M, lambda_M, nu_M,
                               warmup_A_params):
        """Helper function to hide the book-keeping behind initializing model parameters

        TODO: Replace with better/random initializations
        """

        self.model_params = {}

        self.unconstrained_T_serial = {}
        self.unconstrained_rho_M = {}
        self.unconstrained_lambda_M = {}
        self.unconstrained_nu_M = {}

        self.unconstrained_warmup_A_params = {}

        self.previously_asymptomatic = {}

        for vax_status in [status.value for status in self.vax_statuses]:

            self.unconstrained_T_serial[vax_status] = {}
            self.unconstrained_T_serial[vax_status]['loc'] = \
                tf.Variable(T_serial[vax_status]['posterior_init']['loc'], dtype=tf.float32,
                            name=f'T_serial_A_loc_{vax_status}')
            self.unconstrained_T_serial[vax_status]['scale'] = \
                tf.Variable(T_serial[vax_status]['posterior_init']['scale'], dtype=tf.float32,
                            name=f'T_serial_A_scale_{vax_status}')

            self.unconstrained_rho_M[vax_status] = {}
            self.unconstrained_rho_M[vax_status]['loc'] = \
                tf.Variable(rho_M[vax_status]['posterior_init']['loc'], dtype=tf.float32,
                            name=f'rho_M_loc_{vax_status}')
            self.unconstrained_rho_M[vax_status]['scale'] = \
                tf.Variable(rho_M[vax_status]['posterior_init']['scale'], dtype=tf.float32,
                            name=f'rho_M_scale_{vax_status}')

            self.unconstrained_lambda_M[vax_status] = {}
            self.unconstrained_lambda_M[vax_status]['loc'] = \
                tf.Variable(lambda_M[vax_status]['posterior_init']['loc'], dtype=tf.float32,
                            name=f'lambda_M_loc_{vax_status}')
            self.unconstrained_lambda_M[vax_status]['scale'] = \
                tf.Variable(lambda_M[vax_status]['posterior_init']['scale'], dtype=tf.float32,
                            name=f'lambda_M_scale_{vax_status}')

            self.unconstrained_nu_M[vax_status] = {}
            self.unconstrained_nu_M[vax_status]['loc'] = \
                tf.Variable(nu_M[vax_status]['posterior_init']['loc'], dtype=tf.float32,
                            name=f'nu_M_loc_{vax_status}')
            self.unconstrained_nu_M[vax_status]['scale'] = \
                tf.Variable(nu_M[vax_status]['posterior_init']['scale'], dtype=tf.float32,
                            name=f'nu_M_scale_{vax_status}')

            self.unconstrained_warmup_A_params[vax_status] = []
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

            self.previously_asymptomatic[vax_status] = tf.TensorArray(tf.float32, size=self.transition_window,
                                                                      clear_after_read=False, name=f'prev_asymp')

        return

    def _initialize_priors(self, T_serial, rho_M, lambda_M, nu_M,
                            warmup_A_params):
        """Helper function to hide the book-keeping behind initializing model priors"""

        self.prior_distros = {}
        for enum_c in self.compartments:
            compartment = enum_c.value
            self.prior_distros[compartment] = {}
            for vax_status in [status.value for status in self.vax_statuses]:
                self.prior_distros[compartment][vax_status] = {}

        # create prior distributions
        for vax_status in [status.value for status  in self.vax_statuses]:


            # T serial must be positive
            self.prior_distros[Comp.A.value][vax_status]['T_serial'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.TruncatedNormal(
                    T_serial[vax_status]['prior']['loc'],
                    T_serial[vax_status]['prior']['scale'],
                    0, np.inf),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
            )

            self.prior_distros[Comp.M.value][vax_status]['rho_M'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.Beta(
                    rho_M[vax_status]['prior']['a'],
                    rho_M[vax_status]['prior']['b']),
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

            self.prior_distros[Comp.M.value][vax_status]['nu_M'] = tfp.distributions.TransformedDistribution(
                tfp.distributions.TruncatedNormal(
                    nu_M[vax_status]['prior']['loc'],
                    nu_M[vax_status]['prior']['scale'],
                    0, np.inf),
                bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
            )

            self.prior_distros[Comp.A.value][vax_status]['warmup_A'] = []
            for day in range(self.transition_window):
                self.prior_distros[Comp.A.value][vax_status]['warmup_A'].append(
                    tfp.distributions.TransformedDistribution(
                        tfp.distributions.TruncatedNormal(
                            tf.cast(warmup_A_params[vax_status]['prior'][day]['loc'],dtype=tf.float32),
                            tf.cast(warmup_A_params[vax_status]['prior'][day]['scale'],dtype=tf.float32),
                            0, tf.float32.max),
                        bijector=tfp.bijectors.Invert(tfp.bijectors.Softplus())
                    )
                )



        return


    def  _constrain_parameters(self):
        """Helper function to make sure all of our posterior variance parameters are positive"""

        self.T_serial_params = {}
        self.rho_M_params = {}
        self.lambda_M_params = {}
        self.nu_M_params = {}
        self.warmup_A_params = {}

        for vax_status in [status.value for status in self.vax_statuses]:
            self.T_serial_params[vax_status] = {}
            self.rho_M_params[vax_status] = {}
            self.lambda_M_params[vax_status] = {}
            self.nu_M_params[vax_status] = {}
            self.warmup_A_params[vax_status] = []
            for day in range(self.transition_window):
                self.warmup_A_params[vax_status].append({})


            self.T_serial_params[vax_status]['loc'] = self.unconstrained_T_serial[vax_status]['loc']
            self.T_serial_params[vax_status]['scale'] = tf.math.softplus(self.unconstrained_T_serial[vax_status]['scale'])

            self.rho_M_params[vax_status]['loc'] = self.unconstrained_rho_M[vax_status]['loc']
            self.rho_M_params[vax_status]['scale'] = tf.math.softplus(self.unconstrained_rho_M[vax_status]['scale'])

            self.lambda_M_params[vax_status]['loc'] = self.unconstrained_lambda_M[vax_status]['loc']
            self.lambda_M_params[vax_status]['scale'] = tf.math.softplus(self.unconstrained_lambda_M[vax_status]['scale'])

            self.nu_M_params[vax_status]['loc'] = self.unconstrained_nu_M[vax_status]['loc']
            self.nu_M_params[vax_status]['scale'] = tf.math.softplus(self.unconstrained_nu_M[vax_status]['scale'])


            for day in range(self.transition_window):
                self.warmup_A_params[vax_status][day]['loc'] = \
                    self.unconstrained_warmup_A_params[vax_status][day]['loc']
                self.warmup_A_params[vax_status][day]['scale'] = \
                    tf.math.softplus(self.unconstrained_warmup_A_params[vax_status][day]['scale'])

        return

    def _sample_and_reparameterize(self):
        """Here we again constrain, and our prior distribution will fix it"""

        T_serial_noise = tf.random.normal((self.posterior_samples,))
        # Use reparameterization trick to get unconstrained samples
        self.T_serial_samples = self.T_serial_params[Vax.total.value]['loc'] + \
                                self.T_serial_params[Vax.total.value]['scale'] * T_serial_noise

        # Constrain samples with softplus
        self.T_serial_samples_constrained = tfp.bijectors.Softplus().forward(self.T_serial_samples)

        # Calulate variational posterior probability of un constrained samples
        T_serial_variational_posterior = tfp.distributions.Normal(self.T_serial_params[Vax.total.value]['loc'],
                                                                  self.T_serial_params[Vax.total.value]['scale'])

        self.T_serial_probs = T_serial_variational_posterior.log_prob(self.T_serial_samples)


        rho_M_noise = tf.random.normal((self.posterior_samples,))
        self.rho_M_samples = self.rho_M_params[Vax.total.value]['loc'] + \
                             self.rho_M_params[Vax.total.value]['scale'] * rho_M_noise
        self.rho_M_samples_constrained = tfp.bijectors.Sigmoid().forward(self.rho_M_samples)

        rho_M_variational_posterior = tfp.distributions.Normal(self.rho_M_params[Vax.total.value]['loc'],
                                                               self.rho_M_params[Vax.total.value]['scale'])

        self.rho_M_probs = rho_M_variational_posterior.log_prob(self.rho_M_samples)

        lambda_M_noise = tf.random.normal((self.posterior_samples,))
        self.lambda_M_samples = self.lambda_M_params[Vax.total.value]['loc'] + \
                                self.lambda_M_params[Vax.total.value]['scale'] * lambda_M_noise
        self.lambda_M_samples_constrained = tfp.bijectors.Softplus().forward(self.lambda_M_samples)

        lambda_M_variational_posterior = tfp.distributions.Normal(self.lambda_M_params[Vax.total.value]['loc'],
                                                                  self.lambda_M_params[Vax.total.value]['scale'])

        self.lambda_M_probs = lambda_M_variational_posterior.log_prob(self.lambda_M_samples)


        nu_M_noise = tf.random.normal((self.posterior_samples,))
        self.nu_M_samples = self.nu_M_params[Vax.total.value]['loc'] + \
                            self.nu_M_params[Vax.total.value]['scale'] * nu_M_noise
        self.nu_M_samples_constrained = tfp.bijectors.Softplus().forward(self.nu_M_samples)

        nu_M_variational_posterior = tfp.distributions.Normal(self.nu_M_params[Vax.total.value]['loc'],
                                                              self.nu_M_params[Vax.total.value]['scale'])
        self.nu_M_probs = nu_M_variational_posterior.log_prob(self.nu_M_samples)

        self.warmup_A_samples = []
        self.warmup_A_samples_constrained = []
        self.warmup_A_probs = []
        for day in range(self.transition_window):
            warmup_A_noise = tf.random.normal((self.posterior_samples,))
            self.warmup_A_samples.append(self.warmup_A_params[Vax.total.value][day]['loc'] +
                                         self.warmup_A_params[Vax.total.value][day]['scale'] *
                                         warmup_A_noise)
            self.warmup_A_samples_constrained.append(tfp.bijectors.Softplus().forward(self.warmup_A_samples[-1]))

            warmup_A_variational_posterior = tfp.distributions.Normal(self.warmup_A_params[Vax.total.value][day]['loc'],
                                                                      self.warmup_A_params[Vax.total.value][day]['scale'])


            self.warmup_A_probs.append(warmup_A_variational_posterior.log_prob(self.warmup_A_samples[-1]))

        poisson_M_dist_samples = [tfp.distributions.Poisson(rate=lambda_M)
                                  for lambda_M in self.lambda_M_samples_constrained]


        self.pi_M_samples = tf.TensorArray(tf.float32, size=self.transition_window, clear_after_read=False,
                                           name='pi_M_samples')

        for j in range(self.transition_window):
            poisson_M_samples = np.array([dist.log_prob(j+1) for dist in poisson_M_dist_samples])
            self.pi_M_samples = self.pi_M_samples.write(j, np.array([dist.log_prob(j + 1) for dist in poisson_M_dist_samples]) /
                                                           self.nu_M_samples_constrained)

        self.pi_M_samples = self.pi_M_samples.stack()
        # Softmax so it sums to 1
        self.pi_M_samples = tf.nn.softmax(self.pi_M_samples,axis=0)

        return

    def _initialize_flux_arrays(self, forecast_days):

        forecasted_fluxes = {}

        compartments = [Comp.A.value, Comp.M.value]

        forecasted_fluxes[Comp.A.value] = {}
        forecasted_fluxes[Comp.M.value] = {}

        for compartment in compartments:
            forecasted_fluxes[compartment] = {}
            for vax_status in [status.value for status in self.vax_statuses]:
                forecasted_fluxes[compartment][vax_status] = \
                    tf.TensorArray(tf.float32, size=forecast_days, clear_after_read=False,
                                   name=f'{compartment}_{vax_status}')

        return forecasted_fluxes




    def _add_prior_loss(self, debug=False):
        """Helper function for adding loss from model prior"""

        # Flip the signs from our elbo equation because tensorflow minimizes
        T_serial_prior_prob = -self.prior_distros[Comp.A.value][Vax.total.value]['T_serial'].log_prob(self.T_serial_samples_constrained) + \
            tfp.bijectors.Softplus().forward_log_det_jacobian(self.T_serial_samples)
        T_serial_posterior_prob = -self.T_serial_probs
        self.add_loss(lambda: tf.reduce_sum(tf.reduce_mean(T_serial_prior_prob- T_serial_posterior_prob ,axis=-1)))

        rho_M_prior_prob = -self.prior_distros[Comp.M.value][Vax.total.value]['rho_M'].log_prob(self.rho_M_samples_constrained) + \
            tfp.bijectors.Sigmoid().forward_log_det_jacobian(self.rho_M_samples)
        rho_M_posterior_prob = -self.rho_M_probs
        self.add_loss(lambda: tf.reduce_sum(tf.reduce_mean(rho_M_prior_prob - rho_M_posterior_prob,axis=-1)))

        lambda_M_prior_prob = -self.prior_distros[Comp.M.value][Vax.total.value]['lambda_M'].log_prob(self.lambda_M_samples) + \
            tfp.bijectors.Softplus().forward_log_det_jacobian(self.lambda_M_samples)
        lambda_M_posterior_prob = -self.lambda_M_probs
        self.add_loss(lambda: tf.reduce_sum(tf.reduce_mean(lambda_M_prior_prob - lambda_M_posterior_prob,axis=-1)))

        nu_M_prior_prob = -self.prior_distros[Comp.M.value][Vax.total.value]['nu_M'].log_prob(self.nu_M_samples) + \
            tfp.bijectors.Softplus().forward_log_det_jacobian(self.nu_M_samples)
        nu_M_posterior_prob = -self.nu_M_probs
        self.add_loss(lambda: tf.reduce_sum(tf.reduce_mean(nu_M_prior_prob - nu_M_posterior_prob,axis=-1)))


        for day in range(self.transition_window):
            warmup_A_prior_prob = -self.prior_distros[Comp.A.value][Vax.total.value]['warmup_A'][day].log_prob(self.warmup_A_samples[day]) + \
                tfp.bijectors.Softplus().forward_log_det_jacobian(self.warmup_A_samples[day])
            warmup_A_posterior_prob = -self.warmup_A_probs[day]
            self.add_loss(lambda: tf.reduce_sum(tf.reduce_mean(warmup_A_prior_prob - warmup_A_posterior_prob,axis=-1)))


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
            self.previously_asymptomatic[vax_status].mark_used()

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
        return -tf.reduce_sum(tf.reduce_mean(log_probs,axis=1))


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
    file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
    file_writer.set_as_default()
    logging_callback = VarLogCallback()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    return [logging_callback, tensorboard_callback]

