import functools

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jax import lax
import numpy as onp
import tree


@chex.dataclass
class Normalizer:
  mean: jnp.ndarray
  std: jnp.ndarray

  def __call__(self, inputs):
    # NOTE(yl) The normalization is causing trouble compared to the original impl.
    # when normalization is used, the dynamics rollout explodes, consequently
    # CEM fails (the costs go up to >1e10)
    # This is probably a precision issue, not sure at the moment
    # return input
    return (inputs - self.mean) / (self.std)


class EnsembleModel:

  def __init__(self,
               network,
               preprocess_obs,
               postprocess_obs,
               process_target,
               num_ensembles: int = 5):
    self._network = hk.without_apply_rng(hk.transform(network))
    self._preprocess_obs = preprocess_obs
    self._postprocess_obs = postprocess_obs
    self._process_target = process_target
    self._num_ensembles = num_ensembles

    def gaussian_nll(params, normalizer, x, a, xnext) -> jnp.ndarray:
      """Compute the loss of the network, including L2."""
      proc_x = preprocess_obs(x)
      inputs = jnp.concatenate([proc_x, a], axis=-1)
      inputs = normalizer(inputs)
      mean, logvar = self._network.apply(params, inputs)
      target = process_target(x, xnext)
      inv_var = jnp.exp(-logvar)
      assert mean.shape == target.shape == logvar.shape
      mse_loss = jnp.square(mean - target)
      logp_loss = mse_loss * inv_var + logvar
      # mean along batch and output dimension
      # logp_loss = logp_loss.mean(-1).mean(-1)
      return logp_loss

    def batched_loss(ensem_params, normalizer, x, a, xnext) -> jnp.ndarray:
      """Compute the loss of the network, including L2."""
      nll_loss = jax.vmap(gaussian_nll, (0, None, 0, 0, 0))(ensem_params, normalizer, x,
                                                            a, xnext)
      return nll_loss.mean(axis=(1, 2)).sum()

    def evaluate(params, normalizer, x, a, xnext) -> jnp.ndarray:
      """Compute the validation loss of a single network, MSE.
      """
      # Validation is MSE
      proc_x = preprocess_obs(x)
      inputs = jnp.concatenate([proc_x, a], axis=-1)
      inputs = normalizer(inputs)
      mean, _ = self._network.apply(params, inputs)
      # dist = tfd.Independent(tfd.Normal(loc=mean, scale=std), 1)
      target = process_target(x, xnext)
      mse_loss = jnp.mean(jnp.square(target - mean).mean(axis=-1), axis=-1)
      return mse_loss

    def batched_eval(ensem_params, normalizer, x, a, xnext):
      """Compute the validation loss for the ensembles, MSE
      Args:
        params: ensemble parameters of shape [E, ...]
        normalizer: normalizer for normalizing the inputs
        x, a, x: training data of shape [E, B, ...]
      Returns:
        mse_loss: mean squared error of shape [E] from the ensembles
      """
      losses = jax.vmap(evaluate, (0, None, None, None, None))(ensem_params, normalizer,
                                                               x, a, xnext)
      # Return the validation MSE per ensemble
      return losses

    self._loss_fn = batched_loss
    self._loss_eval = batched_eval

  @property
  def num_ensembles(self):
    return self._num_ensembles

  def init(self, rng, observation, action):
    inputs = jnp.concatenate([self._preprocess_obs(observation), action], axis=-1)
    params_list = []
    rngs = jax.random.split(rng, self._num_ensembles)
    for r in rngs:
      params_list.append(self._network.init(r, inputs))
    # pylint: disable=no-value-for-parameter
    ensem_params = jax.tree_multimap(lambda *x: jnp.stack(x), *params_list)

    mean = jnp.zeros(inputs.shape[-1], dtype=jnp.float32)
    std = jnp.ones(inputs.shape[-1], dtype=jnp.float32)
    return ensem_params, Normalizer(mean=mean, std=std)

  def update_normalizer(self, x, a, xnext):
    del xnext
    new_input = jnp.concatenate([self._preprocess_obs(x), a], axis=-1)
    new_input = onp.asarray(new_input)
    new_mean = onp.mean(new_input, axis=0)
    new_std = onp.std(new_input, axis=0, dtype=onp.float64)
    # We are using a larger eps here for handling observation dims
    # that do not change during training. The original implementation uses
    # 1e-12, which is okay only if the inputs are float64, but is too small
    # for float32 which JAX uses by default.
    #
    # Without this, environments such as reacher or pusher will not work as
    # the observation includes positions of goal which do not change.
    # This needs to be investigated further. In particular, simply changing the eps
    # here does not seem to fix problems.
    # affect how we normalize. While the original impl simply does
    # (o - mean) / std. In the case of small std, the normalized inputs will explode.
    new_std[new_std < 1e-12] = 1.0
    new_mean = jnp.array(new_mean.astype(onp.float32))
    new_std = jnp.array(new_std.astype(onp.float32))
    return Normalizer(mean=new_mean, std=new_std)

  @functools.partial(jax.jit, static_argnums=(0,))
  def loss(self, params, state, observation, action, next_observation):
    return self._loss_fn(params, state, observation, action, next_observation)

  @functools.partial(jax.jit, static_argnums=(0,))
  def evaluate(self, params, state, observation, action, next_observation):
    return self._loss_eval(params, state, observation, action, next_observation)


class ModelEnv:

  def __init__(self, network, obs_preprocess, obs_postprocess, shuffle=True):
    self._network = hk.without_apply_rng(hk.transform(network))
    self._obs_preprocess = obs_preprocess
    self._obs_postprocess = obs_postprocess
    self._shuffle = shuffle

  def reset(self, ensem_params, normalizer, key, observations):
    num_ensembles = tree.flatten(ensem_params)[0].shape[0]
    batch_size = observations.shape[0]
    if batch_size % num_ensembles:
      raise NotImplementedError('Ragged batch not supported.')
    indices = jnp.arange(batch_size)
    if self._shuffle:
      indices = jax.random.permutation(key, indices)
    return (normalizer, indices)

  def step(self, ensem_params, state, key, observations, actions):
    # TODO(yl): Add sampling propagation indices instead of being fully deterministic
    assert observations.ndim == 2
    assert actions.ndim == 2
    normalizer, shuffle_indices = state
    batch_size = observations.shape[0]

    shuffled_observations = observations[shuffle_indices]
    shuffled_actions = actions[shuffle_indices]

    shuffled_states = self._obs_preprocess(shuffled_observations)
    num_ensembles = tree.flatten(ensem_params)[0].shape[0]
    new_batch_size, ragged = divmod(batch_size, num_ensembles)
    shuffled_inputs = jnp.concatenate([shuffled_states, shuffled_actions], axis=-1)
    normalized_inputs = normalizer(shuffled_inputs)
    # from jax.experimental import host_callback as hcb
    if ragged:
      raise NotImplementedError(
          f'Ragged batch not supported. ({batch_size} % {num_ensembles} == {ragged})')
    reshaped_inputs = tree.map_structure(
        lambda x: x.reshape((num_ensembles, new_batch_size, *x.shape[1:])),
        normalized_inputs)
    mean, logvar = jax.vmap(self._network.apply, in_axes=(0, 0))(ensem_params,
                                                                 reshaped_inputs)
    # hcb.id_print((mean[0, 0], logvar[0, 0]), what='in')
    std = jnp.exp(logvar * 0.5)
    mean, std = tree.map_structure(lambda x: x.reshape((batch_size, mean.shape[-1])),
                                   (mean, std))
    shuffled_predictions = jax.random.normal(key, shape=mean.shape) * std + mean
    shuffled_output = self._obs_postprocess(shuffled_observations, shuffled_predictions)
    # Shuffle back
    output = jax.ops.index_update(shuffled_output, shuffle_indices, shuffled_output)
    return output

  def unroll(self, params, state, rng, x_init, actions):
    """Unroll model along a sequence of actions.
    Args:
      ensem_params: hk.Params.
      rng: JAX random key.
      x_init [B, D]
      actions [T, B, A]
    """
    rng, rng_init = jax.random.split(rng)
    state = self.reset(params, state, rng_init, x_init)

    def step(input_, a_t):
      rng, x_t, params, state = input_
      rng, rng_step = jax.random.split(rng)
      x_tp1 = self.step(params, state, rng_step, x_t, a_t)
      return (rng, x_tp1, params, state), x_tp1

    return lax.scan(step, (rng, x_init, params, state), actions)
