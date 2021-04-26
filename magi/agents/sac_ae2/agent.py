from functools import partial
from typing import List, Tuple, Any, Optional
from functools import partial
import math

import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import nn
import optax
from magi.agents.sac_ae2.networks import ContinuousQFunction, SACDecoder, SACEncoder, SACLinear, StateDependentGaussianPolicy
from acme import core, datasets, specs
from acme.utils import loggers, counting
from acme.adders import reverb as adders
from acme.jax import utils, variable_utils
from reverb import rate_limiters
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


def make_networks(environment_spec, num_critics, units_critic, units_actor, feature_dim,
                  log_std_min, log_std_max):

  def fn_critic(x, a):
    # Define without linear layer.
    return ContinuousQFunction(
        num_critics=num_critics,
        hidden_units=units_critic,
    )(x, a)

  def fn_actor(x):
    # Define with linear layer.
    x = SACLinear(feature_dim=feature_dim)(x)
    return StateDependentGaussianPolicy(
        action_size=environment_spec.actions.shape[0],
        hidden_units=units_actor,
        log_std_min=log_std_min,
        log_std_max=log_std_max,
        clip_log_std=False,
    )(x)

  def encoder(x):
    return SACEncoder(num_filters=32, num_layers=4)(x)

  def linear(x):
    return SACLinear(feature_dim=feature_dim)(x)

  def decoder(x):
    return SACDecoder(environment_spec.observations, num_filters=32, num_layers=4)(x)

  # Encoder.
  return {
      'encoder': encoder,
      'decoder': decoder,
      'critic': fn_critic,
      'actor': fn_actor,
      'linear': linear
  }

@jax.jit
def preprocess_state(
    state: np.ndarray,
    key: jnp.ndarray,
) -> jnp.ndarray:
  """
    Preprocess pixel states to fit into [-0.5, 0.5].
    """
  state = state.astype(jnp.float32)
  state = jnp.floor(state / 8)
  state = state / 32
  state = state + jax.random.uniform(key, state.shape) / 32
  state = state - 0.5
  return state


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


@jax.jit
def gaussian_log_prob(
    log_std: jnp.ndarray,
    noise: jnp.ndarray,
) -> jnp.ndarray:
  """
    Calculate log probabilities of gaussian distributions.
    """
  return -0.5 * (jnp.square(noise) + 2 * log_std + jnp.log(2 * math.pi))


@jax.jit
def gaussian_and_tanh_log_prob(
    log_std: jnp.ndarray,
    noise: jnp.ndarray,
    action: jnp.ndarray,
) -> jnp.ndarray:
  """
    Calculate log probabilities of gaussian distributions and tanh transformation.
    """
  return gaussian_log_prob(log_std,
                           noise) - jnp.log(nn.relu(1.0 - jnp.square(action)) + 1e-6)


@partial(jax.jit, static_argnums=3)
def reparameterize_gaussian_and_tanh(
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    key: jnp.ndarray,
    return_log_pi: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """
    Sample from gaussian distributions and tanh transforamation.
    """
  std = jnp.exp(log_std)
  noise = jax.random.normal(key, std.shape)
  action = jnp.tanh(mean + noise * std)
  if return_log_pi:
    return action, gaussian_and_tanh_log_prob(log_std, noise, action).sum(axis=-1)
  else:
    return action


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


def _calculate_log_pi(
    action: np.ndarray,
    log_pi: np.ndarray,
) -> jnp.ndarray:
  return log_pi


def _calculate_loss_critic_and_abs_td(
    value_list: List[jnp.ndarray],
    target: jnp.ndarray,
    weight: np.ndarray,
) -> jnp.ndarray:
  abs_td = jnp.abs(target - value_list[0])
  loss_critic = (jnp.square(abs_td) * weight).mean()
  for value in value_list[1:]:
    loss_critic += (jnp.square(target - value) * weight).mean()
  return loss_critic, jax.lax.stop_gradient(abs_td)


def _sample_action(
    actor_apply,
    params_actor: hk.Params,
    state: np.ndarray,
    key: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  mean, log_std = actor_apply(params_actor, state)
  return reparameterize_gaussian_and_tanh(mean, log_std, key, True)


# Loss functions
def make_critic_loss_fn(encoder_apply, actor_apply, linear_apply, critic_apply, gamma):

  @jax.jit
  def _loss_critic(params_critic: hk.Params, params_critic_target: hk.Params,
                   params_actor: hk.Params, log_alpha: jnp.ndarray, state: np.ndarray,
                   action: np.ndarray, reward: np.ndarray, discount: np.ndarray,
                   next_state: np.ndarray, weight: np.ndarray or List[jnp.ndarray],
                   key) -> Tuple[jnp.ndarray, jnp.ndarray]:
    last_conv = encoder_apply(params_critic["encoder"], state)
    next_last_conv = jax.lax.stop_gradient(
        encoder_apply(params_critic["encoder"], next_state))
    next_action, next_log_pi = _sample_action(actor_apply, params_actor, next_last_conv,
                                              key)
    target = _calculate_target(linear_apply, critic_apply, params_critic_target,
                               log_alpha, reward, discount, next_last_conv, next_action,
                               next_log_pi, gamma)
    q_list = _calculate_value_list(linear_apply, critic_apply, params_critic, last_conv,
                                   action)
    return _calculate_loss_critic_and_abs_td(q_list, target, weight)

  return _loss_critic


@partial(jax.jit, static_argnums=(0, 1))
def _calculate_target(linear_apply, critic_apply, params_critic_target: hk.Params,
                      log_alpha: jnp.ndarray, reward: np.ndarray, discount: np.ndarray,
                      next_state: np.ndarray, next_action: jnp.ndarray,
                      next_log_pi: jnp.ndarray, gamma) -> jnp.ndarray:
  next_q = _calculate_value(linear_apply, critic_apply, params_critic_target,
                            next_state, next_action)
  next_q -= jnp.exp(log_alpha) * _calculate_log_pi(next_action, next_log_pi)
  return jax.lax.stop_gradient(reward + discount * gamma * next_q)


@partial(jax.jit, static_argnums=(0, 1))
def _calculate_value_list(
    linear_apply,
    critic_apply,
    params_critic: hk.Params,
    last_conv: np.ndarray,
    action: np.ndarray,
) -> List[jnp.ndarray]:
  feature = linear_apply(params_critic["linear"], last_conv)
  return critic_apply(params_critic["critic"], feature, action)


@partial(jax.jit, static_argnums=(0, 1))
def _calculate_value(
    linear_apply,
    critic_apply,
    params_critic: hk.Params,
    state: np.ndarray,
    action: np.ndarray,
) -> jnp.ndarray:
  return jnp.asarray(
      _calculate_value_list(linear_apply, critic_apply, params_critic, state,
                            action)).min(axis=0)


def make_actor_loss_fn(encoder_apply, actor_apply, linear_apply, critic_apply):

  @jax.jit
  def _loss_actor(params_actor: hk.Params, params_critic: hk.Params,
                  log_alpha: jnp.ndarray, state: np.ndarray,
                  key) -> Tuple[jnp.ndarray, jnp.ndarray]:
    last_conv = jax.lax.stop_gradient(encoder_apply(params_critic["encoder"], state))
    action, log_pi = _sample_action(actor_apply, params_actor, last_conv, key)
    mean_q = _calculate_value(linear_apply, critic_apply, params_critic, last_conv,
                              action).mean()
    mean_log_pi = _calculate_log_pi(action, log_pi).mean()
    return jax.lax.stop_gradient(
        jnp.exp(log_alpha)) * mean_log_pi - mean_q, jax.lax.stop_gradient(mean_log_pi)

  return _loss_actor


def make_ae_loss_fn(encoder_apply, linear_apply, decoder_apply, lambda_latent,
                    lambda_weight, preprocess_target_fn):

  def _loss_ae(
      params_ae: hk.Params,
      state: np.ndarray,
      key: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Preprocess states.
    target = preprocess_target_fn(state, key)
    # Reconstruct states.
    last_conv = encoder_apply(params_ae["encoder"], state)
    feature = linear_apply(params_ae["linear"], last_conv)
    reconst = decoder_apply(params_ae["decoder"], feature)
    # MSE for reconstruction errors.
    loss_reconst = jnp.square(target - reconst).mean()
    # L2 penalty of latent representations following RAE.
    loss_latent = 0.5 * jnp.square(feature).sum(axis=1).mean()
    # Weight decay for the decoder.
    loss_weight = weight_decay(params_ae["decoder"])
    return loss_reconst + lambda_latent * loss_latent + lambda_weight * loss_weight, None

  return _loss_ae


def _loss_alpha(log_alpha: jnp.ndarray, mean_log_pi: jnp.ndarray,
                target_entropy) -> jnp.ndarray:
  return -log_alpha * (target_entropy + mean_log_pi), None


class SACAEActor(core.Actor):
  """A SAC actor."""

  def __init__(
      self,
      forward_fn,
      encode_fn,
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
      o = utils.add_batch_dim(observation)
      last_conv = encode_fn(params['encoder'], o)
      mean, log_std = forward_fn(params['actor'], last_conv)
      # TODO(yl): Currently this assumes that the forward_fn
      return reparameterize_gaussian_and_tanh(mean, log_std, subkey, False)[0], key

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


class SACAEEvalActor(core.Actor):
  """A SAC actor."""

  def __init__(
      self,
      forward_fn,
      encode_fn,
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
      o = utils.add_batch_dim(observation)
      last_conv = encode_fn(params['encoder'], o)
      mean, _ = forward_fn(params['actor'], last_conv)
      return jnp.tanh(mean)[0], key

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


class SAC_AE:

  def __init__(
      self,
      environment_spec,
      seed,
      max_grad_norm=None,
      gamma=0.99,
      num_critics=2,
      buffer_size=10**6,
      batch_size=128,
      start_steps=1000,
      update_interval=1,
      tau=0.01,
      tau_ae=0.05,
      lr_actor=1e-3,
      lr_critic=1e-3,
      lr_ae=1e-3,
      lr_alpha=1e-4,
      units_actor=(1024, 1024),
      units_critic=(1024, 1024),
      log_std_min=-10.0,
      log_std_max=2.0,
      init_alpha=0.1,
      adam_b1_alpha=0.5,
      feature_dim=50,
      lambda_latent=1e-6,
      lambda_weight=1e-7,
      update_interval_actor=2,
      update_interval_ae=1,
      update_interval_target=2,
      target_processor=preprocess_state
  ):
    # Setup reverb
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
                                           transition_adder=True)

    self._iterator = dataset.as_numpy_iterator()

    self._rng = hk.PRNGSequence(seed)
    self._start_steps = start_steps
    self._num_observations = 0
    self._learning_step = 0
    self._max_grad_norm = max_grad_norm
    # Other parameters.
    self._update_interval = update_interval
    self._update_interval_actor = update_interval_actor
    self._update_interval_ae = update_interval_ae
    self._update_interval_target = update_interval_target

    networks = make_networks(environment_spec, num_critics, units_critic, units_actor,
                             feature_dim, log_std_min, log_std_max)
    fake_obs = utils.add_batch_dim(utils.zeros_like(environment_spec.observations))
    fake_action = utils.add_batch_dim(utils.zeros_like(environment_spec.actions))
    # Setup parameters and pure functions
    # Encoder.
    self.encoder = hk.without_apply_rng(hk.transform(networks['encoder']))
    self.params_encoder = self.params_encoder_target = self.encoder.init(
        next(self._rng), fake_obs)
    fake_last_conv = self.encoder.apply(self.params_encoder, fake_obs)

    # Linear layer for critic and decoder.
    self.linear = hk.without_apply_rng(hk.transform(networks['linear']))
    self.params_linear = self.params_linear_target = self.linear.init(
        next(self._rng), fake_last_conv)
    fake_feature = self.linear.apply(self.params_linear, fake_last_conv)

    # Critic from latent to Q values
    self.critic = hk.without_apply_rng(hk.transform(networks['critic']))
    self.params_critic = self.params_critic_target = self.critic.init(
        next(self._rng), fake_feature, fake_action)

    # Actor.
    self.actor = hk.without_apply_rng(hk.transform(networks['actor']))
    self.params_actor = self.actor.init(next(self._rng), fake_last_conv)
    self.opt_actor = optax.adam(lr_actor)
    self.opt_state_actor = self.opt_actor.init(self.params_actor)

    # Entropy coefficient.
    self.target_entropy = -float(environment_spec.actions.shape[0])
    self.log_alpha = jnp.array(np.log(init_alpha), dtype=jnp.float32)
    self.opt_alpha = optax.adam(lr_alpha, b1=adam_b1_alpha)
    self.opt_state_alpha = self.opt_alpha.init(self.log_alpha)

    # Decoder.
    self.decoder = hk.without_apply_rng(hk.transform(networks['decoder']))
    self.params_decoder = self.decoder.init(next(self._rng), fake_feature)
    self.opt_ae = optax.adam(lr_ae)
    self.opt_state_ae = self.opt_ae.init(self.params_ae)

    # Critic
    self.opt_critic = optax.adam(lr_critic)
    self.opt_state_critic = self.opt_critic.init(self.params_entire_critic)

    # Setup losses
    self._critic_loss_fn = make_critic_loss_fn(self.encoder.apply, self.actor.apply,
                                               self.linear.apply, self.critic.apply,
                                               gamma)
    self._actor_loss_fn = make_actor_loss_fn(self.encoder.apply, self.actor.apply,
                                             self.linear.apply, self.critic.apply)
    self._ae_loss_fn = make_ae_loss_fn(self.encoder.apply, self.linear.apply,
                                       self.decoder.apply, lambda_latent, lambda_weight, target_processor)
    self._alpha_loss_fn = partial(_loss_alpha, target_entropy=self.target_entropy)

    # Update functions for target networks.
    self._update_target_ae = jax.jit(partial(soft_update, tau=tau_ae))
    self._update_target = jax.jit(partial(soft_update, tau=tau))

    # Setup actors
    client = variable_utils.VariableClient(self, '')
    adder = adders.NStepTransitionAdder(client=reverb.Client(address),
                                        n_step=1,
                                        discount=1.0)
    self._actor = SACAEActor(self.actor.apply,
                             self.encoder.apply,
                             next(self._rng),
                             client,
                             adder=adder)
    self._random_actor = RandomActor(environment_spec.actions,
                                     next(self._rng),
                                     adder=adder)
    # Set up actor for running evaluation loop
    self.eval_actor = SACAEEvalActor(self.actor.apply, self.encoder.apply,
                                     next(self._rng), client)

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

  def _should_update(self):
    return (self._num_observations % self._update_interval
            == 0) and (self._num_observations >= self._start_steps)

  def update(self, wait=True):
    if not self._should_update():
      return
    self._learning_step += 1
    batch = next(self._iterator)
    transitions = batch.data
    state = transitions.observation
    next_state = transitions.next_observation
    action = transitions.action
    reward = transitions.reward
    discount = transitions.discount

    # No PER for now
    weight = jnp.ones_like(reward)

    # Update critic.
    self.opt_state_critic, params_entire_critic, loss_critic, abs_td = optimize(
        self._critic_loss_fn,
        self.opt_critic.update,
        self.opt_state_critic,
        self.params_entire_critic,
        self._max_grad_norm,
        params_critic_target=self.params_entire_critic_target,
        params_actor=self.params_actor,
        log_alpha=self.log_alpha,
        state=state,
        action=action,
        reward=reward,
        discount=discount,
        next_state=next_state,
        weight=weight,
        key=next(self._rng),
    )
    self.params_encoder = params_entire_critic["encoder"]
    self.params_linear = params_entire_critic["linear"]
    self.params_critic = params_entire_critic["critic"]

    # Update actor and alpha.
    if self._learning_step % self._update_interval_actor == 0:
      self.opt_state_actor, self.params_actor, loss_actor, mean_log_pi = optimize(
          self._actor_loss_fn,
          self.opt_actor.update,
          self.opt_state_actor,
          self.params_actor,
          self._max_grad_norm,
          params_critic=self.params_entire_critic,
          log_alpha=self.log_alpha,
          state=state,
          key=next(self._rng))
      self.opt_state_alpha, self.log_alpha, loss_alpha, _ = optimize(
          self._alpha_loss_fn,
          self.opt_alpha.update,
          self.opt_state_alpha,
          self.log_alpha,
          None,
          mean_log_pi=mean_log_pi,
      )

    # Update autoencoder.
    if self._learning_step % self._update_interval_ae == 0:
      self.opt_state_ae, params_ae, loss_ae, _ = optimize(
          self._ae_loss_fn,
          self.opt_ae.update,
          self.opt_state_ae,
          self.params_ae,
          self._max_grad_norm,
          state=state,
          key=next(self._rng),
      )
      self.params_encoder = params_ae["encoder"]
      self.params_linear = params_ae["linear"]
      self.params_decoder = params_ae["decoder"]

    # Update target network.
    if self._learning_step % self._update_interval_target == 0:
      self.params_encoder_target = self._update_target_ae(self.params_encoder_target,
                                                          self.params_encoder)
      self.params_linear_target = self._update_target_ae(self.params_linear_target,
                                                         self.params_linear)
      self.params_critic_target = self._update_target(self.params_critic_target,
                                                      self.params_critic)
    self._actor.update(wait=wait)

  @property
  def params_ae(self):
    return {
        "encoder": self.params_encoder,
        "linear": self.params_linear,
        "decoder": self.params_decoder,
    }

  @property
  def params_entire_critic(self):
    return {
        "encoder": self.params_encoder,
        "linear": self.params_linear,
        "critic": self.params_critic,
    }

  @property
  def params_entire_critic_target(self):
    return {
        "encoder": self.params_encoder_target,
        "linear": self.params_linear_target,
        "critic": self.params_critic_target,
    }

  def get_variables(self, names):
    del names
    return [{'encoder': self.params_encoder, 'actor': self.params_actor}]