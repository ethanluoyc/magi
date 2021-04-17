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
import numpy as np

import tensorflow_probability

tfp = tensorflow_probability.experimental.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors


def compute_target(
    policy,
    critic,
    rng,
    policy_params,
    critic_target_params,
    r_t,
    o_tp1,
    d_t,
    alpha,
    discount,
):
  mean_tp1, logstd_tp1 = policy.apply(policy_params, o_tp1)
  action_tp1_base_dist = tfd.Normal(loc=mean_tp1, scale=jnp.exp(logstd_tp1))
  action_tp1_dist = tfd.Independent(tfd.TransformedDistribution(action_tp1_base_dist, tfb.Tanh()), 1)
  # TODO handle constraints
  # Squash the action to be bounded between [-1, 1]
  next_actions = action_tp1_dist.sample(seed=rng)
  next_log_pis = action_tp1_dist.log_prob(next_actions)
  q1_next_target, q2_next_target = critic.apply(critic_target_params, o_tp1, next_actions)
  min_qf_next_target = jnp.minimum(q1_next_target, q2_next_target) - alpha * next_log_pis
  next_q_values = r_t + discount * d_t * min_qf_next_target
  return jax.lax.stop_gradient(next_q_values)


def q_value_loss(policy, critic, rng, policy_params, critic_params,
                 critic_target_params, o_t, a_t, r_t, o_tp1,
                 d_t, alpha, discount):
  next_q_values = compute_target(
    policy, critic, rng, policy_params, critic_target_params, 
    r_t, o_tp1, d_t, alpha, discount)
  next_q_values = jax.lax.stop_gradient(next_q_values)
  q1, q2 = critic.apply(critic_params, o_t, a_t)
  q1_loss = jnp.square(q1 - next_q_values).mean()
  q2_loss = jnp.square(q2 - next_q_values).mean()
  return 0.5 * (q1_loss + q2_loss)


def policy_loss(policy, critic, policy_params, critic_params, rng, o_t,
                a_t, alpha, discount):
  alpha = jax.lax.stop_gradient(alpha)
  mean_t, logstd_t = policy.apply(policy_params, o_t)
  action_tp1_base_dist = tfd.Normal(loc=mean_t, scale=jnp.exp(logstd_t))
  action_tp1_dist = tfd.Independent(tfd.TransformedDistribution(action_tp1_base_dist, tfb.Tanh()), 1)
  # TODO handle constraints
  # Squash the action to be bounded between [-1, 1]
  squashed_action_t = action_tp1_dist.sample(seed=rng)
  assert len(squashed_action_t.shape) == 2
  log_prob = action_tp1_dist.log_prob(squashed_action_t)
  assert len(log_prob.shape) == 1
  q1, q2 = critic.apply(critic_params, o_t, squashed_action_t)
  assert len(q1.shape) == 1
  assert len(q2.shape) == 1
  min_qf = jnp.minimum(q1, q2)
  assert len(min_qf.shape) == 1
  return (jax.lax.stop_gradient(alpha) * log_prob - jax.lax.stop_gradient(min_qf)).mean(), log_prob

def action_log_prob(policy, policy_params, rng, o_t):
  mean_t, logstd_t = policy.apply(policy_params, o_t)
  action_tp1_base_dist = tfd.Normal(loc=mean_t, scale=jnp.exp(logstd_t))
  action_tp1_dist = tfd.Independent(tfd.TransformedDistribution(action_tp1_base_dist, tfb.Tanh()), 1)
  # TODO handle constraints
  # Squash the action to be bounded between [-1, 1]
  squashed_action_t = action_tp1_dist.sample(seed=rng)
  log_prob = action_tp1_dist.log_prob(squashed_action_t)
  return squashed_action_t, log_prob

def polyak_update(old_params, new_params, tau):
  return jax.tree_multimap(lambda o, n: tau * n + (1 - tau) * o, old_params,
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
               lr_actor=3e-4,
               lr_critic=3e-4,
               lr_alpha=3e-4,
               init_alpha=1.0,
               discount=0.99,
               tau=5e-3, 
               logger=None):
    self._environment_spec = environment_spec
    self._policy_network = policy_network
    self._critic_network = critic_network
    self._rng = hk.PRNGSequence(key)
    self._data_iterator = data_iterator
    self._discount = discount
    self._tau = tau

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
    self.target_entropy = -np.prod(dummy_action.shape)

    # initialize parameters
    policy_params = policy_network.init(next(self._rng), dummy_obs)
    critic_params = critic_network.init(next(self._rng), dummy_obs, dummy_action)
    critic_target_params = jax.tree_map(lambda x: x.copy(), critic_params)
    # Copy target network params
    log_alpha = jnp.log(jnp.array(init_alpha))

    opt_state_actor = self._opt_actor.init(policy_params)
    opt_state_critic = self._opt_critic.init(critic_params)
    opt_state_alpha = self._opt_alpha.init(log_alpha)

    self._params = {
        "policy": policy_params,
        "critic": critic_params,
        "critic_target": critic_target_params,
        "log_alpha": log_alpha
    }

    self._opt_state = {
        "policy": opt_state_actor,
        "critic": opt_state_critic,
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
      critic_params = params['critic']
      critic_target_params = params['critic_target']
      log_alpha = params['log_alpha']
      alpha = jnp.exp(log_alpha)

      critic_opt_state = opt_state['critic']
      policy_opt_state = opt_state['policy']
      alpha_opt_state = opt_state['log_alpha']

      key_critic, key_policy, key_alpha = jax.random.split(rng, 3)
      # q_target = compute_target(
      #     self._policy_network, self._critic_network,
      #     key_target, policy_params, critic1_target_params,
      #     critic2_target_params, r_t, o_tp1, d_t, jnp.exp(log_alpha), self._discount
      # )
      # Update Q-function by one step SGD

      (critic_loss_, grad_critic) = (jax.value_and_grad(
          q_value_loss,
          argnums=4)(self._policy_network, self._critic_network, key_critic,
                          policy_params, critic_params,
                          critic_target_params, o_t, a_t, r_t,
                          o_tp1, d_t, alpha, self._discount))
      grad_critic_norm = optax.global_norm(grad_critic)
      critic_updates, critic_opt_state = self._opt_critic.update(
          grad_critic, critic_opt_state, critic_params)
      critic_params = optax.apply_updates(critic_params,
                                          critic_updates)  # update the parameters.

      # Update policy by one step of gradient descent
      (actor_loss, _), grad_policy = (jax.value_and_grad(
          policy_loss, argnums=2,
          has_aux=True)(self._policy_network, self._critic_network, policy_params,
                        critic_params, key_policy, o_t, a_t, alpha,
                        self._discount))
      policy_grad_norm = optax.global_norm(grad_policy)
      policy_updates, policy_opt_state = self._opt_actor.update(
          grad_policy, policy_opt_state, policy_params)
      policy_params = optax.apply_updates(policy_params,
                                          policy_updates)  # update the parameters.
      # Update target networks
      critic_target_params = optax.incremental_update(
        critic_params,
        critic_target_params, self._tau)
      #
      _, log_probs = action_log_prob(self._policy_network, policy_params, key_alpha, o_t)
      # Update alpha
      log_alpha_loss, grad_log_alpha = jax.value_and_grad(self._loss_alpha)(log_alpha,
                                                                            log_probs)
      log_alpha_updates, alpha_opt_state = self._opt_alpha.update(
          grad_log_alpha, alpha_opt_state, log_alpha)
      log_alpha = optax.apply_updates(log_alpha, log_alpha_updates)

      # Update parameters
      new_params = {
          "policy": policy_params,
          "critic": critic_params,
          "critic_target": critic_target_params,
          "log_alpha": log_alpha
      }

      # Update state
      new_opt_state = {
          "policy": policy_opt_state,
          "critic": critic_opt_state,
          "log_alpha": alpha_opt_state,
      }
      results = {
          'policy_loss': actor_loss,
          'critic_loss': critic_loss_,
          'log_alpha_loss': log_alpha_loss,
          'alpha': jnp.exp(log_alpha),
          "entropy": jnp.mean(-log_probs),
          "critic_global_norm": grad_critic_norm,
          "policy_grad_norm": policy_grad_norm,
      }
      return results, new_params, new_opt_state

    self._sgd_step = sgd_step

  def get_variables(self, names):
    return [self._params['policy']]

  def _loss_alpha(
      self,
      log_alpha: jnp.ndarray,
      log_probs: jnp.ndarray,
  ) -> jnp.ndarray:
    # Eqn. 18
    alpha_losses = -1.0 * jnp.exp(log_alpha) * (jax.lax.stop_gradient(log_probs) + self.target_entropy)
    return jnp.mean(alpha_losses, 0)

  def step(self):
    batch = next(self._data_iterator)
    start_time = time.time()
    losses, self._params, self._opt_state = self._sgd_step(batch, next(self._rng),
                                                           self._params,
                                                           self._opt_state)
    losses = utils.to_numpy(losses)
    self._logger.write({'elapsed_time': time.time() - start_time, **losses})