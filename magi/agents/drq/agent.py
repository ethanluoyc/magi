import dataclasses
from functools import partial
import time
from typing import Any, List, Mapping, Optional, Tuple

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
import tensorflow_probability

from magi.agents import actors
from magi.agents.sac import acting
from magi.agents.drq.augmentations import batched_random_crop

tfp = tensorflow_probability.experimental.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors

batched_random_crop = jax.jit(batched_random_crop)


@jax.jit
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


@jax.jit
def weight_decay(params: hk.Params) -> jnp.ndarray:
  """
    Calculate the sum of L2 norms of parameters.
    """
  leaves, _ = jax.tree_flatten(params)
  return 0.5 * sum(jnp.vdot(x, x) for x in leaves)


@partial(jax.jit, static_argnums=(0, 1, 4))
def optimize(
    fn_loss: Any,
    opt: Any,
    opt_state: Any,
    params_to_update: hk.Params,
    max_grad_norm: float or None,
    *args,
    **kwargs,
) -> Tuple[Any, hk.Params, jnp.ndarray, Any]:
  (loss, aux), grad = jax.value_and_grad(fn_loss, has_aux=True)(
      params_to_update,
      *args,
      **kwargs,
  )
  if max_grad_norm is not None:
    grad = clip_gradient_norm(grad, max_grad_norm)
  update, opt_state = opt(grad, opt_state)
  params_to_update = optax.apply_updates(params_to_update, update)
  return opt_state, params_to_update, loss, aux


@jax.jit
def clip_gradient_norm(
    grad: Any,
    max_grad_norm: float,
) -> Any:
  """
    Clip norms of gradients.
    """

  def _clip_gradient_norm(g):
    clip_coef = max_grad_norm / (jax.lax.stop_gradient(jnp.linalg.norm(g)) + 1e-6)
    clip_coef = jnp.clip(clip_coef, a_max=1.0)
    return g * clip_coef

  return jax.tree_map(lambda g: _clip_gradient_norm(g), grad)


# Loss functions
def make_critic_loss_fn(encoder_apply, actor_apply, critic_apply, gamma):

  @jax.jit
  def _loss_critic(params_critic: hk.Params, params_critic_target: hk.Params,
                   params_actor: hk.Params, log_alpha: jnp.ndarray, state: np.ndarray,
                   action: np.ndarray, reward: np.ndarray, discount: np.ndarray,
                   next_state: np.ndarray, weight: np.ndarray or List[jnp.ndarray],
                   key) -> Tuple[jnp.ndarray, jnp.ndarray]:
    next_last_conv = encoder_apply(params_critic['encoder'], next_state)
    next_dist = actor_apply(params_actor, next_last_conv)
    next_actions, next_log_probs = next_dist.sample_and_log_prob(key)

    # Calculate q target values
    next_last_conv_target = encoder_apply(params_critic_target['encoder'], next_state)
    next_q1, next_q2 = critic_apply(params_critic_target['critic'],
                                    next_last_conv_target, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)
    next_q -= jnp.exp(log_alpha) * next_log_probs
    target_q = jax.lax.stop_gradient(reward + discount * gamma * next_q)
    # target_q = jax.lax.stop_gradient(next_q - gamma * reward + gamma * discount * jnp.exp(log_alpha) * next_log_probs)
    # Calculate predicted Q
    last_conv = encoder_apply(params_critic['encoder'], state)
    q1, q2 = critic_apply(params_critic['critic'], last_conv, action)
    abs_td = jnp.abs(target_q - q1)
    loss_critic = ((jnp.square(target_q - q1) + jnp.square(target_q - q2))
                   * weight).mean()
    return loss_critic, jax.lax.stop_gradient(abs_td)

  return _loss_critic


def make_actor_loss_fn(encoder_apply, actor_apply, critic_apply):

  @jax.jit
  def _loss_actor(params_actor: hk.Params, params_critic: hk.Params,
                  log_alpha: jnp.ndarray, state: np.ndarray,
                  key) -> Tuple[jnp.ndarray, jnp.ndarray]:
    last_conv = jax.lax.stop_gradient(encoder_apply(params_critic['encoder'], state))
    action, log_probs = actor_apply(params_actor, last_conv).sample_and_log_prob(key)
    q1, q2 = critic_apply(params_critic['critic'], last_conv, action)
    q = jnp.minimum(q1, q2)
    actor_loss = (log_probs * jnp.exp(log_alpha) - q).mean()
    entropy = -log_probs.mean()
    return actor_loss, entropy

  return _loss_actor


def _loss_alpha(log_alpha: jnp.ndarray, entropy: jnp.ndarray,
                target_entropy) -> jnp.ndarray:
  temperature = jnp.exp(log_alpha)
  return temperature * (entropy - target_entropy).mean(), None


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
    self._encoder_params = self._encoder_target_params = self._encoder.init(
        next(self._rng), example_obs)
    example_last_conv = self._encoder.apply(self._encoder_params, example_obs)

    # Critic from latent to Q values
    self._critic = hk.without_apply_rng(hk.transform(networks['critic']))
    self._critic_params = self._critic_target_params = self._critic.init(
        next(self._rng), example_last_conv, example_action)

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
    self._critic_loss_fn = make_critic_loss_fn(self._encoder.apply, self._actor.apply,
                                               self._critic.apply, config.discount)
    self._actor_loss_fn = make_actor_loss_fn(self._encoder.apply, self._actor.apply,
                                             self._critic.apply)
    target_entropy = -float(np.prod(environment_spec.actions.shape))
    self._alpha_loss_fn = partial(_loss_alpha, target_entropy=target_entropy)

    # Update functions for target networks.
    self._update_encoder_target = jax.jit(
        partial(soft_update, tau=config.critic_q_soft_update_rate))
    self._update_critic_target = jax.jit(
        partial(soft_update, tau=config.critic_q_soft_update_rate))

    # Setup actors
    self._client = variable_utils.VariableClient(self, '')
    # The adder is used to insert observations into replay.
    # discount is 1.0 as we are multiplying gamma during learner step
    adder = adders.NStepTransitionAdder(client=reverb.Client(address),
                                        n_step=1,
                                        discount=1.0)

    def forward_fn(params, observation):
      feature_map = self._encoder.apply(params['encoder'], observation)
      return self._actor.apply(params['actor'], feature_map)

    self._forward_fn = forward_fn

    self._policy_actor = acting.SACActor(self._forward_fn,
                                         next(self._rng),
                                         is_eval=False,
                                         variable_client=self._client,
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
    batch = next(self._iterator)
    start = time.time()
    transitions = jax.device_put(batch.data)
    state = batched_random_crop(next(self._rng), transitions.observation)
    next_state = batched_random_crop(next(self._rng), transitions.next_observation)
    action = transitions.action
    reward = transitions.reward
    discount = transitions.discount

    # No PER for now
    weight = jnp.ones_like(reward)

    metrics = {}
    # Update critic.
    self._opt_state_critic, params_entire_critic, loss_critic, abs_td = optimize(
        self._critic_loss_fn,
        self._opt_critic.update,
        self._opt_state_critic,
        self._params_entire_critic,
        self._max_grad_norm,
        params_critic_target=self._params_entire_critic_target,
        params_actor=self._actor_params,
        log_alpha=self._log_alpha,
        state=state,
        action=action,
        reward=reward,
        discount=discount,
        next_state=next_state,
        weight=weight,
        key=next(self._rng),
    )
    del abs_td  # No PER for now
    self._encoder_params = params_entire_critic['encoder']
    self._critic_params = params_entire_critic['critic']
    metrics['critic_loss'] = loss_critic

    # Update actor and alpha.
    if self._num_learning_steps % self._actor_update_frequency == 0:
      self._opt_state_actor, self._actor_params, loss_actor, entropy = optimize(
          self._actor_loss_fn,
          self._opt_actor.update,
          self._opt_state_actor,
          self._actor_params,
          self._max_grad_norm,
          params_critic=self._params_entire_critic,
          log_alpha=self._log_alpha,
          state=state,
          key=next(self._rng))
      self._opt_state_alpha, self._log_alpha, loss_alpha, _ = optimize(
          self._alpha_loss_fn,
          self._opt_alpha.update,
          self._opt_state_alpha,
          self._log_alpha,
          None,
          entropy=entropy,
      )
      metrics['alpha_loss'] = loss_alpha
      metrics['actor_loss'] = loss_actor

    # Update target network.
    if self._num_learning_steps % self._critic_target_update_frequency == 0:
      self._encoder_target_params = self._update_encoder_target(
          self._encoder_target_params, self._encoder_params)
      self._critic_target_params = self._update_critic_target(
          self._critic_target_params, self._critic_params)
    self._policy_actor.update(wait=wait)

    self._num_learning_steps += 1
    metrics = utils.to_numpy(metrics)
    counts = self._counter.increment(steps=1, time_elapsed=time.time() - start)

    self._logger.write({**counts, **metrics})

  def get_variables(self, names):
    del names
    return [{'encoder': self._encoder_params, 'actor': self._actor_params}]

  def make_actor(self, is_eval=True):
    return acting.SACActor(self._forward_fn,
                           next(self._rng),
                           is_eval=is_eval,
                           variable_client=self._client)

  @property
  def _params_entire_critic(self):
    return {
        'encoder': self._encoder_params,
        'critic': self._critic_params,
    }

  @property
  def _params_entire_critic_target(self):
    return {
        'encoder': self._encoder_target_params,
        'critic': self._critic_target_params,
    }
