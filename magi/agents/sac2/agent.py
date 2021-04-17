"""Soft Actor-Critic implementation"""
from functools import partial
from typing import Iterable, Iterator, List, Optional, Tuple

import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import reverb
import tensorflow_probability
from acme import core, datasets, specs
from acme.utils import loggers, counting
from acme.adders import reverb as adders
from acme.jax import utils, variable_utils
from reverb import rate_limiters

tfp = tensorflow_probability.experimental.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors

# TODO(yl): debug numerical issues with using tfp distributions instead of
# manually handling tanh transformations
# @jax.jit
# def gaussian_log_prob(
#     log_std: jnp.ndarray,
#     noise: jnp.ndarray,
# ) -> jnp.ndarray:
#   """
#     Calculate log probabilities of gaussian distributions.
#     """
#   return -0.5 * (jnp.square(noise) + 2 * log_std + jnp.log(2 * math.pi))

# @jax.jit
# def gaussian_and_tanh_log_prob(
#     log_std: jnp.ndarray,
#     noise: jnp.ndarray,
#     action: jnp.ndarray,
# ) -> jnp.ndarray:
#   """
#     Calculate log probabilities of gaussian distributions and tanh transformation.
#     """
#   return gaussian_log_prob(log_std,
#                            noise) - jnp.log(nn.relu(1.0 - jnp.square(action)) + 1e-6)

# @partial(jax.jit, static_argnums=2)
# def reparameterize_gaussian_and_tanh(
#     dist,
#     key: jnp.ndarray,
#     return_log_pi: bool = True,
# ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#   """
#     Sample from gaussian distributions and tanh transforamation.
#     """
#   # dist = _output_to_dist(mean, log_std)
#   action = dist.sample(seed=key)
#   if return_log_pi:
#     return action, dist.log_prob(action)
#   return action
#   # std = jnp.exp(log_std)
#   # noise = jax.random.normal(key, std.shape)
#   # action = jnp.tanh(mean + noise * std)
#   # if return_log_pi:
#   #   return action, gaussian_and_tanh_log_prob(log_std, noise, action).sum(axis=1)
#   # else:
#   #   return action


def heuristic_target_entropy(action_spec):
  """Compute the heuristic target entropy"""
  return -float(np.prod(action_spec.shape))


# TODO(yl): The actors are probably redundant, consider using the generic actor in Acme.
class SACActor(core.Actor):
  """A SAC actor."""

  def __init__(
      self,
      forward_fn,
      key,
      variable_client,
      adder=None,
  ):

    # Store these for later use.
    self._adder = adder
    self._variable_client = variable_client
    self._key = key

    @jax.jit
    def forward(params, key, observation):
      key, subkey = jax.random.split(key)
      # TODO(yl): Currently this assumes that the forward_fn
      # can handle both batched and unbatched inputs, consider lifing this assumption
      dist = forward_fn(params, observation)
      action = dist.sample(seed=subkey)
      return action, key

    self._forward = forward
    # Make sure not to use a random policy after checkpoint restoration by
    # assigning variables before running the environment loop.
    if self._variable_client is not None:
      self._variable_client.update_and_wait()

  def select_action(self, observation):
    # Forward.
    action, self._key = self._forward(self._params, self._key, observation)
    action = utils.to_numpy(action)
    return action

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder is not None:
      self._adder.add_first(timestep)

  def observe(
      self,
      action,
      next_timestep: dm_env.TimeStep,
  ):
    if self._adder is not None:
      self._adder.add(action, next_timestep)

  def update(self, wait: bool = True):  # not the default wait = False
    if self._variable_client is not None:
      self._variable_client.update(wait=wait)

  @property
  def _params(self) -> Optional[hk.Params]:
    if self._variable_client is None:
      # If self._variable_client is None then we assume self._forward  does not
      # use the parameters it is passed and just return None.
      return None
    return self._variable_client.params


class RandomActor(core.Actor):
  """An actor that samples random actions."""

  def __init__(
      self,
      action_spec,
      key,
      adder=None,
  ):
    # Store these for later use.
    self._adder = adder
    self._key = key
    self._action_spec = action_spec

    @partial(jax.jit, backend='cpu')
    def forward(key, observation):
      del observation
      key, subkey = jax.random.split(key)
      action_dist = tfd.Uniform(low=jnp.broadcast_to(self._action_spec.minimum,
                                                     self._action_spec.shape),
                                high=jnp.broadcast_to(self._action_spec.maximum,
                                                      self._action_spec.shape))
      action = action_dist.sample(seed=subkey)
      return action, key

    self._forward_fn = forward

  def select_action(self, observation):
    # Forward.
    action, self._key = self._forward_fn(self._key, observation)
    return utils.to_numpy(action)

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder is not None:
      self._adder.add_first(timestep)

  def observe(
      self,
      action,
      next_timestep: dm_env.TimeStep,
  ):
    if self._adder is not None:
      self._adder.add(action, next_timestep)

  def update(self, wait: bool = False):
    pass


class SACLearner(core.Learner):

  def __init__(self,
               environment_spec: specs.EnvironmentSpec,
               policy: hk.Transformed,
               critic: hk.Transformed,
               key: jax.random.PRNGKey,
               dataset: Iterator[reverb.ReplaySample],
               gamma: float = 0.99,
               tau: float = 5e-3,
               lr_actor: float = 3e-4,
               lr_critic: float = 3e-4,
               lr_alpha: float = 3e-4,
               init_alpha: float = 1.0,
               adam_b1_alpha: float = 0.9,
               logger: Optional[loggers.Logger] = None,
               counter: Optional[counting.Counter] = None):
    self._rng = hk.PRNGSequence(key)
    self._iterator = dataset
    self._gamma = gamma

    self._logger = logger if logger is not None else loggers.make_default_logger(
        label='learner', save_data=False)
    self._counter = counter if counter is not None else counting.Counter()

    # Define fake input for critic.
    dummy_state = utils.add_batch_dim(utils.zeros_like(environment_spec.observations))
    dummy_action = utils.add_batch_dim(utils.zeros_like(environment_spec.actions))

    # Critic.
    self.critic = critic
    self.params_critic = self.params_critic_target = self.critic.init(
        next(self._rng), dummy_state, dummy_action)
    opt_init, self.opt_critic = optax.adam(lr_critic)
    self.opt_state_critic = opt_init(self.params_critic)
    # Actor.
    self.actor = policy
    self.params_actor = self.actor.init(next(self._rng), dummy_state)
    opt_init, self.opt_actor = optax.adam(lr_actor)
    self.opt_state_actor = opt_init(self.params_actor)
    # Entropy coefficient.
    self.target_entropy = heuristic_target_entropy(environment_spec.actions)
    self.log_alpha = jnp.array(np.log(init_alpha), dtype=jnp.float32)
    opt_init, self.opt_alpha = optax.adam(lr_alpha, b1=adam_b1_alpha)
    self.opt_state_alpha = opt_init(self.log_alpha)

    @jax.jit
    def _update_actor(params_actor, opt_state, key, params_critic, log_alpha, state):
      (loss, aux), grad = jax.value_and_grad(self._loss_actor,
                                             has_aux=True)(params_actor, params_critic,
                                                           log_alpha, state, key)
      update, opt_state = self.opt_actor(grad, opt_state)
      params_actor = optax.apply_updates(params_actor, update)
      return params_actor, opt_state, loss, aux

    @jax.jit
    def _update_critic(
        params_critic,
        opt_state,
        key,
        params_critic_target,
        params_actor,
        log_alpha,
        state,
        action,
        reward,
        discount,
        next_state,
        weight,
    ):
      (loss,
       aux), grad = jax.value_and_grad(self._loss_critic,
                                       has_aux=True)(params_critic,
                                                     params_critic_target, params_actor,
                                                     log_alpha, state, action, reward,
                                                     discount, next_state, weight, key)
      update, opt_state = self.opt_critic(grad, opt_state)
      params_critic = optax.apply_updates(params_critic, update)
      return params_critic, opt_state, loss, aux

    @jax.jit
    def _update_alpha(log_alpha, opt_state, mean_log_pi):
      (loss, aux), grad = jax.value_and_grad(self._loss_alpha,
                                             has_aux=True)(log_alpha, mean_log_pi)
      update, opt_state = self.opt_alpha(grad, opt_state)
      log_alpha = optax.apply_updates(log_alpha, update)
      return log_alpha, opt_state, loss, aux

    self._update_actor = _update_actor
    self._update_critic = _update_critic
    self._update_alpha = _update_alpha
    self._update_target = jax.jit(partial(optax.incremental_update, step_size=tau))

  @partial(jax.jit, static_argnums=0)
  def _calculate_loss_critic_and_abs_td(
      self,
      value_list: List[jnp.ndarray],
      target: jnp.ndarray,
      weight: np.ndarray,
  ) -> jnp.ndarray:
    abs_td = jnp.abs(target - value_list[0])
    loss_critic = (jnp.square(abs_td) * weight).mean()
    for value in value_list[1:]:
      loss_critic += (jnp.square(target - value) * weight).mean()
    return loss_critic, jax.lax.stop_gradient(abs_td)

  def step(self):
    batch = next(self._iterator)
    state, action, reward, discount, next_state = batch.data
    # No PER for now
    weight = jnp.ones_like(reward)
    discount = discount * self._gamma

    # Update critic.
    self.params_critic, self.opt_state_critic, loss_critic, _ = self._update_critic(
        self.params_critic,
        self.opt_state_critic,
        key=next(self._rng),
        params_critic_target=self.params_critic_target,
        params_actor=self.params_actor,
        log_alpha=self.log_alpha,
        state=state,
        action=action,
        reward=reward,
        discount=discount,
        next_state=next_state,
        weight=weight,
    )
    # Update actor
    self.params_actor, self.opt_state_actor, loss_actor, mean_log_pi = self._update_actor(
        self.params_actor,
        self.opt_state_actor,
        key=next(self._rng),
        params_critic=self.params_critic,
        log_alpha=self.log_alpha,
        state=state)
    self.log_alpha, self.opt_state_alpha, loss_alpha, _ = self._update_alpha(
        self.log_alpha,
        self.opt_state_alpha,
        mean_log_pi=mean_log_pi,
    )
    self._counter.increment(steps=1)
    results = {
        'loss_alpha': loss_alpha,
        'loss_actor': loss_actor,
        'loss_critic': loss_critic
    }
    self._logger.write(results)
    # Update target network.
    self.params_critic_target = self._update_target(self.params_critic,
                                                    self.params_critic_target)

  @partial(jax.jit, static_argnums=0)
  def _calculate_target(
      self,
      params_critic_target: hk.Params,
      log_alpha: jnp.ndarray,
      reward: np.ndarray,
      discount: np.ndarray,
      next_state: np.ndarray,
      next_action: jnp.ndarray,
      next_log_pi: jnp.ndarray,
  ) -> jnp.ndarray:
    next_qs = jnp.stack(self.critic.apply(params_critic_target, next_state,
                                          next_action)).min(axis=0)
    next_q = next_qs - jnp.exp(log_alpha) * next_log_pi
    assert len(next_q.shape) == 1
    assert len(reward.shape) == 1
    return jax.lax.stop_gradient(reward + discount * next_q)

  @partial(jax.jit, static_argnums=0)
  def _loss_critic(self, params_critic: hk.Params, params_critic_target: hk.Params,
                   params_actor: hk.Params, log_alpha: jnp.ndarray, state: np.ndarray,
                   action: np.ndarray, reward: np.ndarray, discount: np.ndarray,
                   next_state: np.ndarray, weight: np.ndarray or List[jnp.ndarray],
                   key) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # next_action, next_log_pi = self._sample_action(params_actor, next_state, key)
    next_action_dist = self.actor.apply(params_actor, state)
    next_action = next_action_dist.sample(seed=key)
    next_log_pi = next_action_dist.log_prob(next_action)
    target = self._calculate_target(params_critic_target, log_alpha, reward, discount,
                                    next_state, next_action, next_log_pi)
    q_list = self.critic.apply(params_critic, state, action)
    abs_td = jnp.abs(target - q_list[0])
    loss = (jnp.square(abs_td) * weight).mean()
    for value in q_list[1:]:
      loss += (jnp.square(target - value) * weight).mean()
    return loss, jax.lax.stop_gradient(abs_td)

  @partial(jax.jit, static_argnums=0)
  def _loss_actor(self, params_actor: hk.Params, params_critic: hk.Params,
                  log_alpha: jnp.ndarray, state: np.ndarray,
                  key) -> Tuple[jnp.ndarray, jnp.ndarray]:
    action_dist = self.actor.apply(params_actor, state)
    action = action_dist.sample(seed=key)
    log_pi = action_dist.log_prob(action)

    mean_q = jnp.stack(self.critic.apply(params_critic, state,
                                         action)).min(axis=0).mean()
    mean_log_pi = log_pi.mean()
    return jax.lax.stop_gradient(
        jnp.exp(log_alpha)) * mean_log_pi - mean_q, jax.lax.stop_gradient(mean_log_pi)

  @partial(jax.jit, static_argnums=0)
  def _loss_alpha(
      self,
      log_alpha: jnp.ndarray,
      mean_log_pi: jnp.ndarray,
  ) -> jnp.ndarray:
    # TODO(yl): Investigate if it should be log_alpha or exp(log_alpha)
    return -log_alpha * (self.target_entropy + mean_log_pi), None

  def get_variables(self, names):
    del names
    return [self.params_actor]


class SACAgent(core.Actor):

  def __init__(self,
               environment_spec,
               policy,
               critic,
               seed,
               gamma=0.99,
               buffer_size=10**6,
               batch_size=256,
               start_steps=10000,
               tau=5e-3,
               lr_actor=3e-4,
               lr_critic=3e-4,
               lr_alpha=3e-4,
               init_alpha=1.0,
               adam_b1_alpha=0.9,
               logger=None,
               counter=None):
    # self.rng = hk.PRNGSequence(seed)
    learner_key, actor_key, random_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    self._num_observations = 0
    self._start_steps = start_steps

    replay_table = reverb.Table(name=adders.DEFAULT_PRIORITY_TABLE,
                                sampler=reverb.selectors.Uniform(),
                                remover=reverb.selectors.Fifo(),
                                max_size=buffer_size,
                                rate_limiter=rate_limiters.MinSize(1),
                                signature=adders.NStepTransitionAdder.signature(
                                    environment_spec=environment_spec))
    self._server = reverb.Server([replay_table], port=None)

    # The adder is used to insert observations into replay.
    # discount is 1.0 as we are multiplying gamma during learner step
    address = f'localhost:{self._server.port}'
    # The dataset provides an interface to sample from replay.
    dataset = datasets.make_reverb_dataset(server_address=address,
                                           environment_spec=environment_spec,
                                           batch_size=batch_size,
                                           prefetch_size=1,
                                           transition_adder=True)
    self._learner = SACLearner(environment_spec,
                               policy,
                               critic,
                               key=learner_key,
                               dataset=dataset.as_numpy_iterator(),
                               gamma=gamma,
                               tau=tau,
                               lr_actor=lr_actor,
                               lr_critic=lr_critic,
                               lr_alpha=lr_alpha,
                               init_alpha=init_alpha,
                               adam_b1_alpha=adam_b1_alpha,
                               logger=logger,
                               counter=counter)

    adder = adders.NStepTransitionAdder(client=reverb.Client(address),
                                        n_step=1,
                                        discount=1.0)

    client = variable_utils.VariableClient(self._learner, '')
    self._actor = SACActor(policy.apply, actor_key, client, adder=adder)
    self._random_actor = RandomActor(environment_spec.actions, random_key, adder=adder)

  def select_action(self, observation):
    return self._get_active_actor().select_action(observation)

  def observe_first(self, timestep: dm_env.TimeStep):
    return self._get_active_actor().observe_first(timestep)

  def observe(self, action, next_timestep: dm_env.TimeStep):
    self._num_observations += 1
    self._get_active_actor().observe(action, next_timestep)

  def _get_active_actor(self):
    if self._num_observations < self._start_steps:
      return self._random_actor
    else:
      return self._actor

  def update(self):
    if self._num_observations < self._start_steps:
      return
    self._learner.step()
    self._actor.update(wait=True)

  def get_variables(self, names):
    return [self._learner.get_variables(names)]