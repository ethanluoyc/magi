import time
import acme
import collections
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import tree
from acme.utils import loggers
from acme.jax import utils

import tensorflow_probability

tfp = tensorflow_probability.experimental.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors


def compute_target(
    policy,
    critic,
    rng,
    policy_params,
    critic1_target_params,
    critic2_target_params,
    r_t,
    o_tp1,
    d_t,
    alpha,
    discount,
):
  mean_tp1, logstd_tp1 = policy.apply(policy_params, o_tp1)
  action_tp1_base_dist = tfd.MultivariateNormalDiag(loc=mean_tp1,
                                                    scale_diag=jnp.exp(logstd_tp1))
  action_tp1_dist = tfd.TransformedDistribution(action_tp1_base_dist, tfb.Tanh())
  # TODO handle constraints
  # Squash the action to be bounded between [-1, 1]
  next_actions = action_tp1_dist.sample(seed=rng)
  next_log_pis = action_tp1_dist.log_prob(next_actions)
  q1 = critic.apply(critic1_target_params, o_tp1, next_actions)
  q2 = critic.apply(critic2_target_params, o_tp1, next_actions)
  return (r_t + discount * d_t * (jnp.minimum(q1, q2) - alpha * next_log_pis))


def q_value_loss(policy, critic, rng, policy_params, critic1_params, critic2_params,
                 critic1_target_params, critic2_target_params, o_t, a_t, r_t, o_tp1,
                 done, alpha, discount):
  target = compute_target(policy, critic, rng, policy_params, critic1_target_params,
                          critic2_target_params, r_t, o_tp1, done, alpha, discount)
  q1 = critic.apply(critic1_params, o_t, a_t)
  q2 = critic.apply(critic2_params, o_t, a_t)
  q1_loss = jnp.square(q1 - jax.lax.stop_gradient(target))
  q2_loss = jnp.square(q2 - jax.lax.stop_gradient(target))
  return jnp.mean(q1_loss + q2_loss)


def policy_loss(policy, critic, policy_params, critic1_params, critic2_params, rng, o_t,
                a_t, alpha, discount):
  mean_t, logstd_t = policy.apply(policy_params, o_t)
  action_tp1_base_dist = tfd.MultivariateNormalDiag(loc=mean_t,
                                                    scale_diag=jnp.exp(logstd_t))
  action_tp1_dist = tfd.TransformedDistribution(action_tp1_base_dist, tfb.Tanh())
  # TODO handle constraints
  # Squash the action to be bounded between [-1, 1]
  squashed_action_t = action_tp1_dist.sample(seed=rng)
  q1 = critic.apply(critic1_params, o_t, squashed_action_t)
  q2 = critic.apply(critic2_params, o_t, squashed_action_t)
  q = jnp.minimum(q1, q2)
  mean_q = q.mean()
  mean_log_pi = action_tp1_dist.log_prob(squashed_action_t).mean()
  return jax.lax.stop_gradient(alpha) * mean_log_pi - mean_q, jax.lax.stop_gradient(
      mean_log_pi)


def polyak_update(old_params, new_params, rate):
  return jax.tree_multimap(lambda o, n: rate * o + (1 - rate) * n, old_params,
                           new_params)


def dummy_from_spec(spec):
  if isinstance(spec, collections.OrderedDict):
    return jax.tree_map(lambda x: x.generate_value(), spec)
  return spec.generate_value()


class SACLearner(acme.Learner):

  def __init__(self,
               environment_spec,
               policy_network,
               critic_network,
               data_iterator,
               key,
               lr_actor=1e-4,
               lr_critic=1e-4,
               lr_alpha=1e-4,
               init_alpha=0.1,
               discount=0.99,
               rate=1 - 0.005, 
               logger=None):
    self._environment_spec = environment_spec
    self._policy_network = policy_network
    self._critic_network = critic_network
    self._rng = hk.PRNGSequence(key)
    self._data_iterator = data_iterator
    self._discount = discount
    self._rate = rate

    # Set up optimizers
    self._opt_actor = optax.adam(lr_actor)
    self._opt_critic = optax.adam(lr_critic)
    self._opt_alpha = optax.adam(lr_alpha)

    observation_spec = environment_spec.observations
    action_spec = environment_spec.actions

    dummy_obs = tree.map_structure(lambda x: jnp.expand_dims(x, 0),
                                   dummy_from_spec(observation_spec))
    dummy_action = tree.map_structure(lambda x: jnp.expand_dims(x, 0),
                                      dummy_from_spec(action_spec))
    self.target_entropy = -jnp.log(dummy_action.size)

    # initialize parameters
    policy_params = policy_network.init(next(self._rng), dummy_obs)
    critic1_params = critic_network.init(next(self._rng), dummy_obs, dummy_action)
    critic2_params = critic_network.init(next(self._rng), dummy_obs, dummy_action)
    # Copy target network params
    critic1_target_params = jax.tree_map(lambda x: x.copy(), critic1_params)
    critic2_target_params = jax.tree_map(lambda x: x.copy(), critic2_params)
    log_alpha = jnp.log(jnp.array(init_alpha))

    opt_state_actor = self._opt_actor.init(policy_params)
    opt_state_critic1 = self._opt_actor.init(critic1_params)
    opt_state_critic2 = self._opt_actor.init(critic2_params)
    opt_state_alpha = self._opt_actor.init(log_alpha)

    self._params = {
        "policy": policy_params,
        "critic1": critic1_params,
        "critic2": critic2_params,
        "critic1_target": critic1_target_params,
        "critic2_target": critic2_target_params,
        "log_alpha": log_alpha
    }

    self._opt_state = {
        "policy": opt_state_actor,
        "critic1": opt_state_critic1,
        "critic2": opt_state_critic2,
        "log_alpha": opt_state_alpha,
    }

    self._logger = logger if logger is not None else loggers.TerminalLogger('learner', time_delta=10.)

    @jax.jit
    def sgd_step(batch, rng, params, opt_state):
      o_t, a_t, r_t, d_t, o_tp1 = batch.data
      # transitions = batch.data
      # o_t = transitions.observation
      # o_tp1 = transitions.next_observation
      # a_t = transitions.action
      # r_t = transitions.reward
      # d_t = transitions.discount

      # Compute targets for the Q functions
      # y(r, s', d)
      # target = r_t + discount * (1 - d_t) # (alpha * )
      policy_params = params['policy']
      critic1_params = params['critic1']
      critic2_params = params['critic2']
      critic1_target_params = params['critic1_target']
      critic2_target_params = params['critic2_target']
      log_alpha = params['log_alpha']
      alpha = jnp.exp(log_alpha)

      critic1_opt_state = opt_state['critic1']
      critic2_opt_state = opt_state['critic2']
      policy_opt_state = opt_state['policy']
      alpha_opt_state = opt_state['log_alpha']

      key_critic, key_policy = jax.random.split(rng, 2)
      # q_target = compute_target(
      #     self._policy_network, self._critic_network,
      #     key_target, policy_params, critic1_target_params,
      #     critic2_target_params, r_t, o_tp1, d_t, jnp.exp(log_alpha), self._discount
      # )
      # Update Q-function by one step SGD

      (critic_loss_, (grad_critic1, grad_critic2)) = (jax.value_and_grad(
          q_value_loss,
          argnums=(5, 6))(self._policy_network, self._critic_network, key_critic,
                          policy_params, critic1_params, critic2_params,
                          critic1_target_params, critic2_target_params, o_t, a_t, r_t,
                          o_tp1, d_t, alpha, self._discount))
      critic1_updates, critic1_opt_state = self._opt_critic.update(
          grad_critic1, critic1_opt_state, critic1_params)
      critic2_updates, critic2_opt_state = self._opt_critic.update(
          grad_critic2, critic2_opt_state, critic2_params)
      critic1_params = optax.apply_updates(critic1_params,
                                           critic1_updates)  # update the parameters.
      critic2_params = optax.apply_updates(critic1_params,
                                           critic2_updates)  # update the parameters.

      # Update policy by one step of gradient descent
      (actor_loss, log_pi_mean), grad_policy = (jax.value_and_grad(
          policy_loss, argnums=2,
          has_aux=True)(self._policy_network, self._critic_network, policy_params,
                        critic1_params, critic2_params, key_policy, o_t, a_t, alpha,
                        self._discount))
      policy_updates, policy_opt_state = self._opt_actor.update(
          grad_policy, policy_opt_state, policy_params)
      policy_params = optax.apply_updates(policy_params,
                                          policy_updates)  # update the parameters.
      # Update target networks
      critic1_target_params = polyak_update(critic1_target_params, critic1_params,
                                            self._rate)
      critic2_target_params = polyak_update(critic2_target_params, critic2_params,
                                            self._rate)

      # Update alpha(
      log_alpha_loss, grad_log_alpha = jax.value_and_grad(self._loss_alpha)(log_alpha,
                                                                            log_pi_mean)
      log_alpha_updates, alpha_opt_state = self._opt_alpha.update(
          grad_log_alpha, alpha_opt_state, log_alpha)
      log_alpha = optax.apply_updates(log_alpha, log_alpha_updates)

      # Update parameters
      new_params = {
          "policy": policy_params,
          "critic1": critic1_params,
          "critic2": critic2_params,
          "critic1_target": critic1_target_params,
          "critic2_target": critic2_target_params,
          "log_alpha": log_alpha
      }

      # Update state
      new_opt_state = {
          "policy": policy_opt_state,
          "critic1": critic1_opt_state,
          "critic2": critic2_opt_state,
          "log_alpha": alpha_opt_state,
      }
      losses = {
          'policy': actor_loss,
          'critic': critic_loss_,
          'log_alpha': log_alpha_loss
      }
      return losses, new_params, new_opt_state

    self._sgd_step = sgd_step

  def get_variables(self, names):
    return [self._params['policy']]

  def _loss_alpha(
      self,
      log_alpha: jnp.ndarray,
      mean_log_pi: jnp.ndarray,
  ) -> jnp.ndarray:
    return -log_alpha * (mean_log_pi + self.target_entropy)

  def step(self):
    batch = next(self._data_iterator)
    start_time = time.time()
    losses, self._params, self._opt_state = self._sgd_step(batch, next(self._rng),
                                                           self._params,
                                                           self._opt_state)
    losses = utils.to_numpy(losses)
    self._logger.write({'elapsed_time': time.time() - start_time, **losses})
