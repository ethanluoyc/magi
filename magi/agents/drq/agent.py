import dataclasses
from functools import partial
import time
from typing import Any, List, Mapping, Optional, Tuple, Iterator

from acme import types

from acme.adders import reverb as adders
from acme import core
from acme import datasets
from acme.jax import utils
from acme.jax import variable_utils
from acme import specs
from acme.utils import counting
from acme.utils import loggers
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import reverb
from reverb import rate_limiters
from reverb.replay_sample import ReplaySample
import tensorflow_probability

from magi.agents import actors
from magi.agents.sac import acting
from magi.agents.drq.augmentations import batched_random_crop

tfp = tensorflow_probability.experimental.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors

batched_random_crop = jax.jit(batched_random_crop)


def soft_update(
    target_params: hk.Params,
    online_params: hk.Params,
    tau: float,
) -> hk.Params:
  """
    Update target network using Polyak-Ruppert Averaging.
    """
  return jax.tree_multimap(lambda t, s: (1 - tau) * t + tau * s, target_params,
                           online_params)


# Loss functions
def make_critic_loss_fn(encoder_apply, actor_apply, critic_apply, gamma):

  def _loss_critic(params_critic: hk.Params, key: jnp.ndarray,
                   params_critic_target: hk.Params, params_actor: hk.Params,
                   log_alpha: jnp.ndarray, batch: reverb.ReplaySample):
    data: types.Transition = batch.data
    next_features = jax.lax.stop_gradient(
        encoder_apply(params_critic['encoder'], data.next_observation))
    next_dist = actor_apply(params_actor, next_features)
    next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=key)
    assert len(next_log_probs.shape) == 1

    # Calculate q target values
    next_last_conv_target = encoder_apply(params_critic_target['encoder'],
                                          data.next_observation)
    next_q1, next_q2 = critic_apply(params_critic_target['critic'],
                                    next_last_conv_target, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)
    next_q -= jnp.exp(log_alpha) * next_log_probs
    target_q = data.reward + data.discount * gamma * next_q
    target_q = jax.lax.stop_gradient(target_q)
    assert len(target_q.shape) == 1
    assert len(next_q1.shape) == 1
    assert len(next_q2.shape) == 1
    # Calculate predicted Q
    features = encoder_apply(params_critic['encoder'], data.observation)
    q1, q2 = critic_apply(params_critic['critic'], features, data.action)
    assert len(q1.shape) == 1
    assert len(q2.shape) == 1
    # abs_td = jnp.abs(target_q - q1)
    loss_critic = (jnp.square(target_q - q1) + jnp.square(target_q - q2)).mean(axis=0)
    return loss_critic, {'q1': q1.mean(), 'q2': q2.mean()}

  return _loss_critic


def make_actor_loss_fn(encoder_apply, actor_apply, critic_apply):

  def _loss_actor(params_actor: hk.Params, key, params_critic: hk.Params,
                  log_alpha: jnp.ndarray, batch: reverb.ReplaySample):
    data: types.Transition = batch.data
    last_conv = encoder_apply(params_critic['encoder'], data.observation)
    assert len(last_conv.shape) == 2
    actions, log_probs = actor_apply(params_actor,
                                     last_conv).sample_and_log_prob(seed=key)
    assert len(log_probs.shape) == 1
    q1, q2 = critic_apply(params_critic['critic'], last_conv, actions)
    assert len(q1.shape) == 1
    assert len(q2.shape) == 1
    q = jnp.minimum(q1, q2)
    actor_loss = (log_probs * jnp.exp(log_alpha) - q).mean(axis=0)
    entropy = -log_probs.mean()
    return actor_loss, {'log_probs': log_probs, 'entropy': entropy}

  return _loss_actor


def _loss_alpha(log_alpha: jnp.ndarray, log_probs: jnp.ndarray,
                target_entropy) -> jnp.ndarray:
  temperature = jnp.exp(log_alpha)
  entropy = -log_probs.mean()
  return temperature * (entropy - target_entropy), {'alpha': temperature}


@dataclasses.dataclass
class DrQConfig:
  """Configuration parameters for SAC-AE.

  Notes:
    These parameters are taken from [1].
    Note that hyper-parameters such as log-stddev bounds on the policy should
    be configured in the network builder.
  """

  min_replay_size: int = 1
  max_replay_size: int = 1_000_000
  replay_table_name: str = adders.DEFAULT_PRIORITY_TABLE

  discount: float = 0.99
  batch_size: int = 128
  initial_num_steps: int = 1000

  critic_learning_rate: float = 3e-4
  critic_target_update_frequency: int = 1
  critic_q_soft_update_rate: float = 0.005

  actor_learning_rate: float = 3e-4
  actor_update_frequency: int = 1

  max_grad_norm: Optional[float] = None

  temperature_learning_rate: float = 3e-4
  temperature_adam_b1: float = 0.5
  init_temperature: float = 0.1


class DrQAgent(core.Actor, core.VariableSource):

  def __init__(self,
               environment_spec: specs.EnvironmentSpec,
               networks: Mapping[str, Any],
               seed: int,
               config: Optional[DrQConfig] = None,
               counter: Optional[counting.Counter] = None,
               logger: Optional[loggers.Logger] = None):
    # Setup reverb
    if config is None:
      config = DrQConfig()
    replay_table = reverb.Table(name=adders.DEFAULT_PRIORITY_TABLE,
                                sampler=reverb.selectors.Uniform(),
                                remover=reverb.selectors.Fifo(),
                                max_size=config.max_replay_size,
                                rate_limiter=rate_limiters.MinSize(
                                    config.min_replay_size),
                                signature=adders.NStepTransitionAdder.signature(
                                    environment_spec=environment_spec))

    # Hold a reference to server to prevent from being gc'ed.
    self._server = reverb.Server([replay_table], port=None)

    address = f'localhost:{self._server.port}'
    # The dataset provides an interface to sample from replay.
    dataset = datasets.make_reverb_dataset(server_address=address,
                                           environment_spec=environment_spec,
                                           batch_size=config.batch_size,
                                           transition_adder=True)

    self._iterator = dataset.as_numpy_iterator()

    self._rng = hk.PRNGSequence(seed)
    self._initial_num_steps = config.initial_num_steps
    self._num_observations = 0
    self._num_learning_steps = 0
    self._max_grad_norm = config.max_grad_norm
    # Other parameters.
    self._actor_update_frequency = config.actor_update_frequency
    self._critic_target_update_frequency = config.critic_target_update_frequency
    self._counter = counter if counter is not None else counting.Counter()
    self._logger = logger if logger is not None else loggers.make_default_logger(
        label='learner', save_data=False)

    example_obs = utils.add_batch_dim(utils.zeros_like(environment_spec.observations))
    example_action = utils.add_batch_dim(utils.zeros_like(environment_spec.actions))
    # Setup parameters and pure functions
    # Encoder.
    self._encoder = hk.without_apply_rng(hk.transform(networks['encoder']))
    self._encoder_params = self._encoder.init(next(self._rng), example_obs)
    self._encoder_target_params = jax.tree_map(lambda x: x.copy(), self._encoder_params)
    example_last_conv = self._encoder.apply(self._encoder_params, example_obs)

    # Critic from latent to Q values
    self._critic = hk.without_apply_rng(hk.transform(networks['critic']))
    self._critic_params = self._critic.init(next(self._rng), example_last_conv,
                                            example_action)
    self._critic_target_params = jax.tree_map(lambda x: x.copy(), self._critic_params)

    # Actor.
    self._actor = hk.without_apply_rng(hk.transform(networks['actor']))
    self._actor_params = self._actor.init(next(self._rng), example_last_conv)
    self._opt_actor = optax.adam(config.actor_learning_rate)
    self._opt_state_actor = self._opt_actor.init(self._actor_params)

    # Entropy coefficient.
    self._log_alpha = jnp.array(np.log(config.init_temperature), dtype=jnp.float32)
    self._opt_alpha = optax.adam(config.temperature_learning_rate,
                                 b1=config.temperature_adam_b1)
    self._opt_state_alpha = self._opt_alpha.init(self._log_alpha)

    # Critic
    self._opt_critic = optax.adam(config.critic_learning_rate)
    self._opt_state_critic = self._opt_critic.init(self._params_entire_critic)

    # Setup losses
    critic_loss_fn = make_critic_loss_fn(self._encoder.apply, self._actor.apply,
                                         self._critic.apply, config.discount)
    actor_loss_fn = make_actor_loss_fn(self._encoder.apply, self._actor.apply,
                                       self._critic.apply)
    target_entropy = -float(np.prod(environment_spec.actions.shape))
    alpha_loss_fn = partial(_loss_alpha, target_entropy=target_entropy)

    @jax.jit
    def _update_critic(critic_params, critic_opt_state, key, critic_target_params,
                       actor_params, log_alpha, batch):
      loss_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
      (loss, aux), grad = loss_grad_fn(critic_params, key, critic_target_params,
                                       actor_params, log_alpha, batch)
      update, new_opt_state = self._opt_critic.update(grad, critic_opt_state)
      new_params = optax.apply_updates(critic_params, update)
      return new_opt_state, new_params, loss, aux

    @jax.jit
    def _update_actor(params_actor: hk.Params, actor_opt_state, key,
                      params_critic: hk.Params, log_alpha: jnp.ndarray,
                      batch: reverb.ReplaySample):
      loss_grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
      (loss, aux), grad = loss_grad_fn(params_actor, key, params_critic, log_alpha,
                                       batch)
      update, new_opt_state = self._opt_actor.update(grad, actor_opt_state)
      new_params = optax.apply_updates(params_actor, update)
      return new_opt_state, new_params, loss, aux

    @jax.jit
    def _update_alpha(log_alpha, alpha_opt_state, log_probs):
      loss_grad_fn = jax.value_and_grad(alpha_loss_fn, has_aux=True)
      (loss, aux), grad = loss_grad_fn(log_alpha, log_probs)
      update, new_opt_state = self._opt_alpha.update(grad, alpha_opt_state)
      new_params = optax.apply_updates(log_alpha, update)
      return new_opt_state, new_params, loss, aux

    self._update_critic = _update_critic
    self._update_actor = _update_actor
    self._update_alpha = _update_alpha

    # Update functions for target networks.
    self._update_encoder_target = jax.jit(
        partial(soft_update, tau=config.critic_q_soft_update_rate))
    self._update_critic_target = jax.jit(
        partial(soft_update, tau=config.critic_q_soft_update_rate))

    # Setup actors
    # The adder is used to insert observations into replay.
    # discount is 1.0 as we are multiplying gamma during learner step
    adder = adders.NStepTransitionAdder(client=reverb.Client(address),
                                        n_step=1,
                                        discount=1.0)

    def forward_fn(params, observation):
      feature_map = self._encoder.apply(params['encoder'], observation)
      return self._actor.apply(params['actor'], feature_map)

    self._forward_fn = forward_fn

    client = variable_utils.VariableClient(self, '')
    self._policy_actor = acting.SACActor(self._forward_fn,
                                         next(self._rng),
                                         is_eval=False,
                                         variable_client=client,
                                         adder=adder)
    self._random_actor = actors.RandomActor(environment_spec.actions,
                                            next(self._rng),
                                            adder=adder)
    # Set up actor for running evaluation loop

  def select_action(self, observation):
    return self._get_active_actor().select_action(observation)

  def observe_first(self, timestep: dm_env.TimeStep):
    return self._get_active_actor().observe_first(timestep)

  def observe(self, action, next_timestep: dm_env.TimeStep):
    self._num_observations += 1
    self._get_active_actor().observe(action, next_timestep)

  def _get_active_actor(self):
    if self._num_observations < self._initial_num_steps:
      return self._random_actor
    else:
      return self._policy_actor

  def _should_update(self):
    return self._num_observations >= self._initial_num_steps

  def update(self, wait: bool = True):
    if not self._should_update():
      return
    batch: reverb.ReplaySample = next(self._iterator)
    start = time.time()
    transitions = batch.data
    # data: types.Transition
    state = batched_random_crop(next(self._rng), transitions.observation)
    next_state = batched_random_crop(next(self._rng), transitions.next_observation)
    batch = ReplaySample(
        batch.info, transitions._replace(observation=state,
                                         next_observation=next_state))

    metrics = {}
    # Update critic.
    self._opt_state_critic, new_params_entire_critic, loss_critic, critic_stats = self._update_critic(
        self._params_entire_critic, self._opt_state_critic, next(self._rng),
        self._params_entire_critic_target, self._actor_params, self._log_alpha, batch)
    self._encoder_params = new_params_entire_critic['encoder']
    self._critic_params = new_params_entire_critic['critic']
    metrics.update(loss_critic=loss_critic, **critic_stats)

    # Update actor and alpha.
    if self._num_learning_steps % self._actor_update_frequency == 0:
      self._opt_state_actor, self._actor_params, loss_actor, actor_stats = self._update_actor(
          self._actor_params,
          self._opt_state_actor,
          next(self._rng),
          self._params_entire_critic,
          self._log_alpha,
          batch,
      )
      self._opt_state_alpha, self._log_alpha, loss_alpha, alpha_stats = self._update_alpha(
          self._log_alpha,
          self._opt_state_alpha,
          actor_stats['log_probs'],
      )
      metrics['alpha_loss'] = loss_alpha
      metrics['actor_loss'] = loss_actor
      metrics['entropy'] = actor_stats['entropy']
      metrics['alpha'] = alpha_stats['alpha']

    # Update target network.
    if self._num_learning_steps % self._critic_target_update_frequency == 0:
      new_encoder_target_params = self._update_encoder_target(
          self._encoder_target_params, self._encoder_params)
      new_critic_target_params = self._update_critic_target(self._critic_target_params,
                                                            self._critic_params)
      self._encoder_target_params = new_encoder_target_params
      self._critic_target_params = new_critic_target_params
    self._policy_actor.update(wait=wait)

    self._num_learning_steps += 1
    metrics = utils.to_numpy(metrics)
    counts = self._counter.increment(steps=1, time_elapsed=time.time() - start)

    self._logger.write({**counts, **metrics})

  def get_variables(self, names):
    del names
    return [{'encoder': self._encoder_params, 'actor': self._actor_params}]

  def make_actor(self, is_eval=True):
    client = variable_utils.VariableClient(self, '')
    return acting.SACActor(self._forward_fn,
                           next(self._rng),
                           is_eval=is_eval,
                           variable_client=client)

  @property
  def _params_entire_critic(self):
    return hk.data_structures.to_immutable_dict({
        'encoder': self._encoder_params,
        'critic': self._critic_params,
    })

  @property
  def _params_entire_critic_target(self):
    return hk.data_structures.to_immutable_dict({
        'encoder': self._encoder_target_params,
        'critic': self._critic_target_params,
    })
