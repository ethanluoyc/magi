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
import tree

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
      h = encode_fn(params['encoder'], o)
      # TODO(yl): Currently this assumes that the forward_fn
      # can handle both batched and unbatched inputs, consider lifing this assumption
      dist = forward_fn(params['policy'], h)
      action = dist.sample(seed=subkey)
      return utils.squeeze_batch_dim(action), key

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


class SACAELearner(core.Learner):

  def __init__(self,
               environment_spec: specs.EnvironmentSpec,
               policy: hk.Transformed,
               critic: hk.Transformed,
               linear: hk.Transformed,
               encoder: hk.Transformed,
               decoder: hk.Transformed,
               key: jax.random.PRNGKey,
               dataset: Iterator[reverb.ReplaySample],
               *,
               gamma: float = 0.99,
               lr_actor: float = 1e-3,
               actor_update_freq: int = 2,
               lr_critic: float = 1e-3,
               critic_tau: float = 0.01,
               critic_target_update_freq: int = 2,
               lr_alpha: float = 1e-4,
               lr_encoder: float = 1e-3,
               encoder_tau: float = 0.05,
               lr_decoder: float = 1e-3,
               decoder_update_freq: int = 1,
               decoder_lambda=1e-6,
               ae_l2_cost: 1e-7,
               init_alpha: float = 0.1,
               adam_b1_alpha=0.5,
               logger: Optional[loggers.Logger] = None,
               counter: Optional[counting.Counter] = None):
    self._rng = hk.PRNGSequence(key)
    self._iterator = dataset
    self._gamma = gamma
    self._step = 0

    self._decoder_update_freq = decoder_update_freq
    self._actor_update_freq = actor_update_freq
    self._critic_target_update_freq = critic_target_update_freq

    self._logger = logger if logger is not None else loggers.make_default_logger(
        label='learner', save_data=False)
    self._counter = counter if counter is not None else counting.Counter()

    # Define fake input for critic.
    dummy_obs = utils.add_batch_dim(utils.zeros_like(environment_spec.observations))
    dummy_action = utils.add_batch_dim(utils.zeros_like(environment_spec.actions))

    # Set up encoder
    self.encoder_params = encoder.init(next(self._rng), dummy_obs)
    encoder_opt = optax.adam(lr_encoder)
    # self.encoder_opt_state = encoder_opt.init(self.encoder_params)

    dummy_map = encoder.apply(self.encoder_params, dummy_obs)
    actor_linear_params = linear.init(next(self._rng), dummy_map)
    critic_linear_params = linear.init(next(self._rng), dummy_map)
    dummy_embedding = linear.apply(actor_linear_params, dummy_map)

    # Set up decoder
    self.decoder_params = decoder.init(next(self._rng), dummy_embedding)
    ae_opt = optax.adam(lr_decoder)

    self.ae_opt_state = ae_opt.init(self.decoder_params)

    # Set up policy
    self.policy_params = {
      'policy': policy.init(next(self._rng), dummy_embedding),
      'linear': actor_linear_params,
    }
    policy_opt = optax.adam(lr_actor)
    self.policy_opt_state = policy_opt.init(self.policy_params)

    # Set up critic
    self.critic_params = {
      'critic': critic.init(next(self._rng), dummy_embedding, dummy_action),
      'linear': critic_linear_params,
    }
    critic_opt = optax.adam(lr_critic)
    self.critic_opt_state = critic_opt.init(self.entire_critic_params)

    # Setup log alpha
    self.log_alpha = jnp.array(np.log(init_alpha), dtype=jnp.float32)
    log_alpha_opt = optax.adam(lr_alpha, b1=adam_b1_alpha)
    self.log_alpha_opt_state = log_alpha_opt.init(self.log_alpha)

    target_entropy = heuristic_target_entropy(environment_spec.actions)

    # Setup target params
    self.critic_target_params = tree.map_structure(lambda x: x.copy(),
                                                   self.critic_params)
    self.critic_encoder_target_params = tree.map_structure(lambda x: x.copy(),
                                                           self.encoder_params)

    def _calculate_target(
        params_critic_target: hk.Params,
        log_alpha: jnp.ndarray,
        reward: np.ndarray,
        discount: np.ndarray,
        next_state: np.ndarray,
        next_action: jnp.ndarray,
        next_log_pi: jnp.ndarray,
    ) -> jnp.ndarray:
      next_qs = jnp.stack(critic_forward(params_critic_target, next_state,
                                       next_action)).min(axis=0)
      next_q = next_qs - jnp.exp(log_alpha) * next_log_pi
      assert len(next_q.shape) == 1
      assert len(reward.shape) == 1
      return jax.lax.stop_gradient(reward + discount * next_q)

    def critic_forward(params, encoded, action):
      h = linear.apply(params['linear'], encoded)
      return critic.apply(params['critic'], h, action)

    def policy_forward(params, encoded):
      h = linear.apply(params['linear'], encoded)
      return policy.apply(params['policy'], h)

    self.policy_forward = policy_forward

    def _loss_critic(params_entire_critic: hk.Params, params_entire_critic_target: hk.Params,
                     params_actor: hk.Params,
                     log_alpha: jnp.ndarray, state: np.ndarray, action: np.ndarray,
                     reward: np.ndarray, discount: np.ndarray, next_state: np.ndarray,
                     weight: np.ndarray or List[jnp.ndarray],
                     key) -> Tuple[jnp.ndarray, jnp.ndarray]:
      encoded = encoder.apply(params_entire_critic['encoder'], state)
      next_encoded = encoder.apply(params_entire_critic['encoder'], next_state)
      # next_action_dist = policy.apply(params_actor, next_encoded)
      next_action_dist = policy_forward(params_actor, next_encoded)
      next_action = next_action_dist.sample(seed=key)
      next_log_pi = next_action_dist.log_prob(next_action)
      target = _calculate_target(params_entire_critic_target['critic'], log_alpha, reward, discount,
                                 next_encoded, next_action, next_log_pi)
      q_list = critic_forward(params_entire_critic['critic'], encoded, action)
      abs_td = jnp.abs(target - q_list[0])
      loss = (jnp.square(abs_td) * weight).mean()
      for value in q_list[1:]:
        loss += (jnp.square(target - value) * weight).mean()
      return loss, jax.lax.stop_gradient(abs_td)

    def _loss_actor(params_actor: hk.Params, params_critic: hk.Params,
                    log_alpha: jnp.ndarray, state: np.ndarray,
                    key) -> Tuple[jnp.ndarray, jnp.ndarray]:
      h = encoder.apply(params_critic['encoder'], state)
      h = jax.lax.stop_gradient(h)
      action_dist = policy_forward(params_actor, h)
      action = action_dist.sample(seed=key)
      log_pi = action_dist.log_prob(action)

      mean_q = jnp.stack(critic_forward(params_critic['critic'], h, action)).min(axis=0).mean()
      mean_log_pi = log_pi.mean()
      return jax.lax.stop_gradient(
          jnp.exp(log_alpha)) * mean_log_pi - mean_q, jax.lax.stop_gradient(mean_log_pi)

    def _loss_alpha(log_alpha: jnp.ndarray, mean_log_pi: jnp.ndarray,
                    target_entropy) -> jnp.ndarray:
      # TODO(yl): Investigate if it should be log_alpha or exp(log_alpha)
      return -jnp.exp(log_alpha) * (target_entropy + mean_log_pi), None

    def _update_actor(params_actor, opt_state, key, params_critic,
                      log_alpha, state):
      (loss, aux), grad = jax.value_and_grad(_loss_actor,
                                             has_aux=True)(params_actor,
                                                           params_critic, log_alpha,
                                                           state, key)
      update, opt_state = policy_opt.update(grad, opt_state)
      params_actor = optax.apply_updates(params_actor, update)
      return params_actor, opt_state, loss, aux

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
      (loss, aux), grad = jax.value_and_grad(_loss_critic, has_aux=True)(
          params_critic, params_critic_target, params_actor, log_alpha,
          state, action, reward, discount, next_state, weight, key)
      update, opt_state = critic_opt.update(grad, opt_state)
      params_critic = optax.apply_updates(params_critic, update)
      return params_critic, opt_state, loss, aux

    def _update_alpha(log_alpha, opt_state, mean_log_pi):
      (loss, aux), grad = jax.value_and_grad(_loss_alpha,
                                             has_aux=True)(log_alpha, mean_log_pi,
                                                           target_entropy)
      update, opt_state = log_alpha_opt.update(grad, opt_state)
      log_alpha = optax.apply_updates(log_alpha, update)
      return log_alpha, opt_state, loss, aux

    def _update_autoencoder(params_ae, opt_state, obs):

      def _loss_fn(params, obs):
        encoder_params, decoder_params = params['encoder'], params['decoder']
        h = encoder.apply(encoder_params, obs)
        rec_obs = decoder.apply(decoder_params, h)
        # TODO(yl): consider moving this out
        target = obs['pixels'].astype(jnp.float32) / 255.
        rec_loss = jnp.mean(jnp.square(target - rec_obs))
        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = 0.5 * jnp.mean(jnp.sum(jnp.square(h), axis=-1))
        l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
        loss = rec_loss + decoder_lambda * latent_loss + l2_loss * ae_l2_cost
        return loss, None

      (loss, aux), grad = jax.value_and_grad(_loss_fn, has_aux=True)(params_ae, obs)

      update, opt_state = encoder_opt.update(grad, opt_state)
      new_params_ae = optax.apply_updates(params_ae, update)
      return new_params_ae, opt_state, loss, aux

    def _update_target(entire_critic_new, entire_critic_target):
      updated_encoder_target_p = optax.incremental_update(entire_critic_new['encoder'], entire_critic_target['encoder'], step_size=encoder_tau)
      updated_critic_target_p = optax.incremental_update(entire_critic_new['critic'], entire_critic_target['critic'], step_size=critic_tau)
      return {
        'encoder': updated_encoder_target_p, 'critic': updated_critic_target_p
      }

    self._update_critic = jax.jit(_update_critic)
    self._update_actor = jax.jit(_update_actor)
    self._update_alpha = jax.jit(_update_alpha)
    self._update_autoencoder = jax.jit(_update_autoencoder)
    self._update_target = jax.jit(_update_target)

  @property
  def ae_params(self):
    return {
        "encoder": self.encoder_params,
        "decoder": self.decoder_params,
    }

  @property
  def entire_critic_params(self):
    return {
        "encoder": self.encoder_params,
        "critic": self.critic_params,
    }

  @property
  def entire_critic_target_params(self):
    return {
        "encoder": self.critic_encoder_target_params,
        "critic": self.critic_target_params,
    }

  def step(self):
    batch = next(self._iterator)
    transitions = batch.data
    state = transitions.observation
    next_state = transitions.next_observation
    action = transitions.action
    reward = transitions.reward
    discount = transitions.discount

    # No PER for now
    weight = jnp.ones_like(reward)
    discount = discount * self._gamma

    results = {}
    # TODO clarify the parameter flows
    # Update critic
    new_critic_params, self.critic_opt_state, loss_critic, _ = self._update_critic(
        self.entire_critic_params, self.critic_opt_state, next(self._rng),
        self.entire_critic_target_params, self.policy_params,
        self.log_alpha, state, action, reward, discount, next_state, weight)
    self.critic_params = new_critic_params['critic']
    self.encoder_params = new_critic_params['encoder']
    results['loss_critic'] = loss_critic

    # Update actor and alpha
    if self._step % self._actor_update_freq:
      self.policy_params, self.policy_opt_state, loss_actor, mean_log_pi = self._update_actor(
          self.policy_params, self.policy_opt_state, next(self._rng),
          self.entire_critic_target_params, self.log_alpha,
          state)
      results['loss_actor'] = loss_actor

      self.log_alpha, self.log_alpha_opt_state, loss_alpha, _, = self._update_alpha(
          self.log_alpha, self.log_alpha_opt_state, mean_log_pi)
      results['loss_alpha'] = loss_alpha
      results['alpha'] = jnp.exp(self.log_alpha)

    # Update target network.
    if self._step % self._critic_target_update_freq:
      new_target_params = self._update_target(self.entire_critic_params, self.entire_critic_target_params)
      self.critic_encoder_target_params = new_target_params['encoder']
      self.critic_target_params = new_target_params['critic']

    if self._step % self._decoder_update_freq:
      new_ae_params, self.ae_opt_state, loss_ae, _ = self._update_autoencoder(
          self.ae_params, self.ae_opt_state, state)
      self.encoder_params = new_ae_params['encoder']
      self.decoder_params = new_ae_params['decoder']
      results['ae_loss'] = loss_ae

    self._step += 1
    self._counter.increment(steps=1)
    self._logger.write({**results, **self._counter.get_counts()})

  def get_variables(self, names):
    del names
    return [{'policy': self.policy_params, 'encoder': self.encoder_params}]


class SACAEAgent(core.Actor):

  def __init__(self,
               environment_spec,
               policy,
               critic,
               encoder,
               decoder,
               linear,
               seed,
               gamma=0.99,
               buffer_size=10**6,
               batch_size=256,
               start_steps=10000,
               lr_actor: float = 1e-3,
               actor_update_freq: int = 2,
               lr_critic: float = 1e-3,
               critic_tau: float = 0.01,
               critic_target_update_freq: int = 2,
               lr_alpha: float = 1e-4,
               lr_encoder: float = 1e-3,
               encoder_tau: float = 0.05,
               lr_decoder: float = 1e-3,
               decoder_update_freq: int = 1,
               decoder_lambda=1e-6,
               ae_l2_cost=1e-7,
               init_alpha: float = 0.1,
               adam_b1_alpha=0.5,
               logger=None,
               counter=None):
    # self.rng = hk.PRNGSequence(seed)
    learner_key, actor_key, actor_key2, random_key = jax.random.split(
        jax.random.PRNGKey(seed), 4)
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
    self._learner = SACAELearner(environment_spec,
                                 policy,
                                 critic,
                                 linear,
                                 encoder,
                                 decoder,
                                 key=learner_key,
                                 dataset=dataset.as_numpy_iterator(),
                                 gamma=gamma,
                                 lr_actor=lr_actor,
                                 actor_update_freq=actor_update_freq,
                                 lr_critic=lr_critic,
                                 critic_tau=critic_tau,
                                 critic_target_update_freq=critic_target_update_freq,
                                 lr_alpha=lr_alpha,
                                 lr_encoder=lr_encoder,
                                 encoder_tau=encoder_tau,
                                 lr_decoder=lr_decoder,
                                 decoder_lambda=decoder_lambda,
                                 ae_l2_cost=ae_l2_cost,
                                 decoder_update_freq=decoder_update_freq,
                                 init_alpha=init_alpha,
                                 adam_b1_alpha=adam_b1_alpha,
                                 logger=logger,
                                 counter=counter)

    adder = adders.NStepTransitionAdder(client=reverb.Client(address),
                                        n_step=1,
                                        discount=1.0)

    client = variable_utils.VariableClient(self._learner, '')
    self._actor = SACAEActor(self._learner.policy_forward,
                             encoder.apply,
                             actor_key,
                             client,
                             adder=adder)
    self._eval_actor = SACAEActor(self._learner.policy_forward, encoder.apply, actor_key2, client)
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