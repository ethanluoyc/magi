from functools import partial
from typing import List, Tuple, Any
from functools import partial
import math

import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import nn
import optax as optix
from magi.agents.sac_ae2.networks import ContinuousQFunction, SACDecoder, SACEncoder, SACLinear, StateDependentGaussianPolicy
from magi.agents.sac_ae2.replay import ReplayBuffer
from gym.spaces import Box
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
import tree

tfp = tensorflow_probability.experimental.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors


def fake_state(state_shape):
  return jnp.zeros(state_shape)


def fake_action(num_actions):
  return jnp.zeros((1, num_actions))


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


@jax.jit
def evaluate_gaussian_and_tanh_log_prob(
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    action: jnp.ndarray,
) -> jnp.ndarray:
  """
    Calculate log probabilities of gaussian distributions and tanh transformation given samples.
    """
  noise = (jnp.arctanh(action) - mean) / (jnp.exp(log_std) + 1e-8)
  return gaussian_and_tanh_log_prob(log_std, noise, action).sum(axis=1, keepdims=True)


@partial(jax.jit, static_argnums=3)
def reparameterize_gaussian(
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    key: jnp.ndarray,
    return_log_pi: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """
    Sample from gaussian distributions.
    """
  std = jnp.exp(log_std)
  noise = jax.random.normal(key, std.shape)
  action = mean + noise * std
  if return_log_pi:
    return action, gaussian_log_prob(log_std, noise).sum(axis=1, keepdims=True)
  else:
    return action


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
    return action, gaussian_and_tanh_log_prob(log_std, noise, action).sum(axis=1,
                                                                          keepdims=True)
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
  params_to_update = optix.apply_updates(params_to_update, update)
  return opt_state, params_to_update, loss, aux


@jax.jit
def clip_gradient(
    grad: Any,
    max_value: float,
) -> Any:
  """
    Clip gradients.
    """
  return jax.tree_map(lambda g: jnp.clip(g, -max_value, max_value), grad)


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
def make_critic_loss_fn(encoder_apply, actor_apply, linear_apply, critic_apply,
                        discount):

  @jax.jit
  def _loss_critic(params_critic: hk.Params, params_critic_target: hk.Params,
                   params_actor: hk.Params, log_alpha: jnp.ndarray, state: np.ndarray,
                   action: np.ndarray, reward: np.ndarray, done: np.ndarray,
                   next_state: np.ndarray, weight: np.ndarray or List[jnp.ndarray],
                   key) -> Tuple[jnp.ndarray, jnp.ndarray]:
    last_conv = encoder_apply(params_critic["encoder"], state)
    next_last_conv = jax.lax.stop_gradient(
        encoder_apply(params_critic["encoder"], next_state))
    next_action, next_log_pi = _sample_action(actor_apply, params_actor, next_last_conv,
                                              key)
    target = _calculate_target(linear_apply, critic_apply, params_critic_target,
                               log_alpha, reward, done, next_last_conv, next_action,
                               next_log_pi, discount)
    q_list = _calculate_value_list(linear_apply, critic_apply, params_critic, last_conv,
                                   action)
    return _calculate_loss_critic_and_abs_td(q_list, target, weight)

  return _loss_critic


@partial(jax.jit, static_argnums=(0, 1))
def _calculate_target(
    linear_apply,
    critic_apply,
    params_critic_target: hk.Params,
    log_alpha: jnp.ndarray,
    reward: np.ndarray,
    done: np.ndarray,
    next_state: np.ndarray,
    next_action: jnp.ndarray,
    next_log_pi: jnp.ndarray,
    discount,
) -> jnp.ndarray:
  next_q = _calculate_value(linear_apply, critic_apply, params_critic_target,
                            next_state, next_action)
  next_q -= jnp.exp(log_alpha) * _calculate_log_pi(next_action, next_log_pi)
  return jax.lax.stop_gradient(reward + (1.0 - done) * discount * next_q)


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
                    lambda_weight):

  def _loss_ae(
      params_ae: hk.Params,
      state: np.ndarray,
      key: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Preprocess states.
    target = preprocess_state(state, key)
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


class SAC_AE:
  name = "SAC+AE"

  def __init__(
      self,
      environment_spec,
      seed,
      max_grad_norm=None,
      gamma=0.99,
      nstep=1,
      num_critics=2,
      buffer_size=10**6,
      batch_size=128,
      start_steps=1000,
      update_interval=1,
      tau=0.01,
      tau_ae=0.05,
      fn_actor=None,
      fn_critic=None,
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
  ):
    # assert len(state_space.shape) == 3 and state_space.shape[:2] == (84, 84)
    # assert (state_space.high == 255).all()
    # self.buffer = ReplayBuffer(
    #     buffer_size=buffer_size,
    #     state_space=state_space,
    #     action_space=action_space,
    #     gamma=gamma,
    #     nstep=nstep,
    # )
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
    self._action_spec = environment_spec.actions

    self._iterator = dataset.as_numpy_iterator()
    self.rng = hk.PRNGSequence(seed)
    self._adder = adders.NStepTransitionAdder(client=reverb.Client(address),
                                              n_step=1,
                                              discount=1.0)
    if fn_critic is None:

      def fn_critic(x, a):
        # Define without linear layer.
        return ContinuousQFunction(
            num_critics=num_critics,
            hidden_units=units_critic,
        )(x, a)

    if fn_actor is None:

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

    fake_feature = jnp.empty((1, feature_dim))
    fake_last_conv = jnp.empty((1, 39200))
    if not hasattr(self, "fake_args_critic"):
      self.fake_args_critic = (fake_feature,
                               utils.add_batch_dim(utils.zeros_like(environment_spec.actions)))
    if not hasattr(self, "fake_args_actor"):
      self.fake_args_actor = (fake_last_conv,)

    self.agent_step = 0
    self.episode_step = 0
    self.start_steps = start_steps
    self.learning_step = 0
    self.update_interval = update_interval
    self.tau = tau
    self.batch_size = batch_size
    self.max_grad_norm = max_grad_norm
    self.discount = gamma

    # Encoder.
    self.encoder = hk.without_apply_rng(
        hk.transform(lambda s: SACEncoder(num_filters=32, num_layers=4)(s)))
    self.params_encoder = self.params_encoder_target = self.encoder.init(
        next(self.rng), utils.add_batch_dim(utils.zeros_like(environment_spec.observations)))

    # Linear layer for critic and decoder.
    self.linear = hk.without_apply_rng(
        hk.transform(lambda x: SACLinear(feature_dim=feature_dim)(x)))
    self.params_linear = self.params_linear_target = self.linear.init(
        next(self.rng), fake_last_conv)

    self.critic = hk.without_apply_rng(hk.transform(fn_critic))
    self.params_critic = self.params_critic_target = self.critic.init(
        next(self.rng), *self.fake_args_critic)

    # Actor.
    self.actor = hk.without_apply_rng(hk.transform(fn_actor))
    self.params_actor = self.actor.init(next(self.rng), *self.fake_args_actor)
    opt_init, self.opt_actor = optix.adam(lr_actor)
    self.opt_state_actor = opt_init(self.params_actor)
    # Entropy coefficient.
    self.target_entropy = -float(self._action_spec.shape[0])
    self.log_alpha = jnp.array(np.log(init_alpha), dtype=jnp.float32)
    opt_init, self.opt_alpha = optix.adam(lr_alpha, b1=adam_b1_alpha)

    self.opt_state_alpha = opt_init(self.log_alpha)
    # Decoder.
    self.decoder = hk.without_apply_rng(
        hk.transform(lambda x: SACDecoder(environment_spec.observations, num_filters=32, num_layers=4)
                     (x)))
    self.params_decoder = self.decoder.init(next(self.rng), fake_feature)
    opt_init, self.opt_ae = optix.adam(lr_ae)
    self.opt_state_ae = opt_init(self.params_ae)

    # Re-define the optimizer for critic.
    opt_init, self.opt_critic = optix.adam(lr_critic)
    self.opt_state_critic = opt_init(self.params_entire_critic)

    # Other parameters.
    self._update_target_ae = jax.jit(partial(soft_update, tau=tau_ae))
    self.lambda_latent = lambda_latent
    self.lambda_weight = lambda_weight
    self.update_interval_actor = update_interval_actor
    self.update_interval_ae = update_interval_ae
    self.update_interval_target = update_interval_target
    self._update_target = jax.jit(partial(soft_update, tau=tau))
    # Setup losses
    self._loss_critic = make_critic_loss_fn(self.encoder.apply, self.actor.apply,
                                            self.linear.apply, self.critic.apply,
                                            self.discount)
    self._loss_actor = make_actor_loss_fn(self.encoder.apply, self.actor.apply,
                                          self.linear.apply, self.critic.apply)
    self._loss_ae = make_ae_loss_fn(self.encoder.apply, self.linear.apply,
                                    self.decoder.apply, self.lambda_latent,
                                    self.lambda_weight)
    self._loss_alpha = partial(_loss_alpha, target_entropy=self.target_entropy)

  @partial(jax.jit, static_argnums=0)
  def _preprocess(
      self,
      params_encoder: hk.Params,
      state: np.ndarray,
  ) -> jnp.ndarray:
    return self.encoder.apply(params_encoder, state)

  def select_action(self, state, is_eval=False):
    if is_eval:
      last_conv = self._preprocess(self.params_encoder, utils.add_batch_dim(state))
      action = self._select_action(self.params_actor, last_conv)[0]
    else:
      if self.agent_step <= self.start_steps:
        # Use random actions
        action_dist = tfd.Uniform(low=jnp.broadcast_to(self._action_spec.minimum,
                                                      self._action_spec.shape),
                                  high=jnp.broadcast_to(self._action_spec.maximum,
                                                        self._action_spec.shape))
        action = action_dist.sample(seed=next(self.rng))
      else:
        last_conv = self._preprocess(self.params_encoder, utils.add_batch_dim(state))
        action = self._explore(self.params_actor, last_conv, next(self.rng))[0]
    return utils.to_numpy(action)

  def greedy_select_action(self, state):
    last_conv = self._preprocess(self.params_encoder, utils.add_batch_dim(state))
    action = self._select_action(self.params_actor, last_conv)[0]
    return utils.to_numpy(action)

  @partial(jax.jit, static_argnums=0)
  def _select_action(
      self,
      params_actor: hk.Params,
      state: np.ndarray,
  ) -> jnp.ndarray:
    mean, _ = self.actor.apply(params_actor, state)
    return jnp.tanh(mean)

  @partial(jax.jit, static_argnums=0)
  def _explore(
      self,
      params_actor: hk.Params,
      state: np.ndarray,
      key: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    mean, log_std = self.actor.apply(params_actor, state)
    return reparameterize_gaussian_and_tanh(mean, log_std, key, False)

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder is not None:
      self._adder.add_first(timestep)

  def observe(
      self,
      action,
      next_timestep: dm_env.TimeStep,
  ):
    self.agent_step += 1
    if self._adder is not None:
      self._adder.add(action, next_timestep)


  def update(self, wait=True):
    if not self.is_update():
      return
    self.learning_step += 1
    # weight, batch = self.buffer.sample(self.batch_size)
    batch = next(self._iterator)
    # state, action, reward, done, next_state = batch
    transitions = batch.data
    state = transitions.observation
    next_state = transitions.next_observation
    action = transitions.action
    reward = jnp.expand_dims(transitions.reward, -1)
    discount = jnp.expand_dims(transitions.discount, -1)

    # No PER for now
    weight = jnp.ones_like(reward)


    # Update critic.
    self.opt_state_critic, params_entire_critic, loss_critic, abs_td = optimize(
        self._loss_critic,
        self.opt_critic,
        self.opt_state_critic,
        self.params_entire_critic,
        self.max_grad_norm,
        params_critic_target=self.params_entire_critic_target,
        params_actor=self.params_actor,
        log_alpha=self.log_alpha,
        state=state,
        action=action,
        reward=reward,
        done=(1-discount),
        next_state=next_state,
        weight=weight,
        key=next(self.rng),
    )
    self.params_encoder = params_entire_critic["encoder"]
    self.params_linear = params_entire_critic["linear"]
    self.params_critic = params_entire_critic["critic"]

    # Update actor and alpha.
    if self.learning_step % self.update_interval_actor == 0:
      self.opt_state_actor, self.params_actor, loss_actor, mean_log_pi = optimize(
          self._loss_actor,
          self.opt_actor,
          self.opt_state_actor,
          self.params_actor,
          self.max_grad_norm,
          params_critic=self.params_entire_critic,
          log_alpha=self.log_alpha,
          state=state,
          key=next(self.rng))
      self.opt_state_alpha, self.log_alpha, loss_alpha, _ = optimize(
          self._loss_alpha,
          self.opt_alpha,
          self.opt_state_alpha,
          self.log_alpha,
          None,
          mean_log_pi=mean_log_pi,
      )

    # Update autoencoder.
    if self.learning_step % self.update_interval_actor == 0:
      self.opt_state_ae, params_ae, loss_ae, _ = optimize(
          self._loss_ae,
          self.opt_ae,
          self.opt_state_ae,
          self.params_ae,
          self.max_grad_norm,
          state=state,
          key=next(self.rng),
      )
      self.params_encoder = params_ae["encoder"]
      self.params_linear = params_ae["linear"]
      self.params_decoder = params_ae["decoder"]

    # Update target network.
    if self.learning_step % self.update_interval_target == 0:
      self.params_encoder_target = self._update_target_ae(self.params_encoder_target,
                                                          self.params_encoder)
      self.params_linear_target = self._update_target_ae(self.params_linear_target,
                                                         self.params_linear)
      self.params_critic_target = self._update_target(self.params_critic_target,
                                                      self.params_critic)

  def is_update(self):
    return self.agent_step % self.update_interval == 0 and self.agent_step >= self.start_steps

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
