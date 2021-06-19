import functools

import chex
import haiku as hk
import jax
import jax.numpy as jnp
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


def gaussian_nll(pred_mean, pred_logvar, target) -> jnp.ndarray:
    """Negative Gaussian log-likelihood."""
    inv_var = jnp.exp(-pred_logvar)
    assert pred_mean.shape == target.shape == pred_logvar.shape
    mse_loss = jnp.square(pred_mean - target)
    logp_loss = mse_loss * inv_var + pred_logvar
    return logp_loss


class EnsembleModel:
    def __init__(
        self,
        network,
        preprocess_obs,
        postprocess_obs,
        process_target,
        num_ensembles: int = 5,
    ):
        self._network = hk.without_apply_rng(hk.transform(network))
        self._preprocess_obs = preprocess_obs
        self._postprocess_obs = postprocess_obs
        self._process_target = process_target
        self._num_ensembles = num_ensembles

        def loss(params, normalizer, x, a, xnext) -> jnp.ndarray:
            """Compute the loss of the network, including L2."""
            proc_x = preprocess_obs(x)
            inputs = jnp.concatenate([proc_x, a], axis=-1)
            inputs = normalizer(inputs)
            mean, logvar = self._network.apply(params, inputs)
            target = process_target(x, xnext)
            return gaussian_nll(mean, logvar, target)

        def batched_loss(ensem_params, normalizer, x, a, xnext) -> jnp.ndarray:
            """Compute the loss of the network, including L2."""
            nll_loss = jax.vmap(loss, (0, None, 0, 0, 0))(
                ensem_params, normalizer, x, a, xnext
            )
            # This is consistent with mbrl-lib,
            # which averages over the batch and event dims and sum over the ensembles
            return nll_loss.mean(axis=(1, 2)).sum()

        def evaluate(params, normalizer, x, a, xnext) -> jnp.ndarray:
            """Compute the validation loss of a single network, MSE."""
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
            losses = jax.vmap(evaluate, (0, None, None, None, None))(
                ensem_params, normalizer, x, a, xnext
            )
            # Return the validation MSE per ensemble
            return losses

        self._loss_fn = batched_loss
        self._eval_fn = batched_eval

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

    def apply(self, params, normalizer, x, a):
        proc_x = self._preprocess_obs(x)
        inputs = jnp.concatenate([proc_x, a], axis=-1)
        inputs = normalizer(inputs)
        mean, logvar = jax.vmap(self._network.apply, (0, None))(params, inputs)
        return mean, logvar

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
        return self._eval_fn(params, state, observation, action, next_observation)


class ModelEnv:
    def __init__(
        self,
        network,
        obs_preprocess,
        obs_postprocess,
        cost_fn,
        terminal_fn,
        shuffle=True,
    ):
        self._network = hk.without_apply_rng(hk.transform(network))
        self._obs_preprocess = obs_preprocess
        self._obs_postprocess = obs_postprocess
        self._shuffle = shuffle
        self._cost_fn = cost_fn
        self._terminal_fn = terminal_fn

    def reset(self, ensem_params, normalizer, key, observations):
        num_ensembles = tree.flatten(ensem_params)[0].shape[0]
        batch_size = observations.shape[0]
        if batch_size % num_ensembles:
            raise NotImplementedError("Ragged batch not supported.")
        indices = jnp.arange(batch_size)
        if self._shuffle:
            indices = jax.random.permutation(key, indices)
        return (normalizer, indices)

    def step(
        self,
        ensem_params,
        normalizer,
        shuffle_indices,
        key,
        observations,
        actions,
        goal,
    ):
        # TODO(yl): Add sampling propagation indices instead of being fully deterministic
        assert observations.ndim == 2
        assert actions.ndim == 2
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
                f"Ragged batch not supported. ({batch_size} % {num_ensembles} == {ragged})"
            )
        reshaped_inputs = tree.map_structure(
            lambda x: x.reshape((num_ensembles, new_batch_size, x.shape[-1])),
            normalized_inputs,
        )
        mean, logvar = jax.vmap(self._network.apply, in_axes=(0, 0))(
            ensem_params, reshaped_inputs
        )
        std = jnp.exp(logvar * 0.5)
        mean, std = tree.map_structure(
            lambda x: x.reshape((batch_size, mean.shape[-1])), (mean, std)
        )
        # Shuffle back
        mean = jax.ops.index_update(mean, shuffle_indices, mean)
        std = jax.ops.index_update(std, shuffle_indices, std)
        # shuffled_predictions = jax.random.normal(key, shape=mean.shape) * std + mean
        predictions = jax.random.normal(key, shape=mean.shape) * std + mean
        output = self._obs_postprocess(observations, predictions)
        return (
            output,
            self._cost_fn(output, actions, goal),
            self._terminal_fn(output, actions, goal),
        )

    def unroll(
        self,
        params,
        normalizer,
        rng,
        initial_state,
        action_sequences,
        goal,
        num_particles: int,
    ):
        """Unroll model along a sequence of actions.
        Args:
          ensem_params: hk.Params.
          rng: JAX random key.
          x_init [B, D]
          actions [B, T, A]
        """
        population_size, horizon, _ = action_sequences.shape
        initial_obs_batch = jnp.tile(
            initial_state, (num_particles * population_size, 1)
        ).astype(jnp.float32)
        rng, rng_reset = jax.random.split(rng)
        normalizer, propagation_id = self.reset(
            params, normalizer, rng_reset, initial_obs_batch
        )
        batch_size = initial_obs_batch.shape[0]
        total_costs = jnp.zeros((batch_size))
        terminated = jnp.zeros((batch_size), dtype=jnp.bool_)
        obs = initial_obs_batch
        for t in range(horizon):
            rng, rng_step = jax.random.split(rng)
            actions_for_step = action_sequences[:, t, :]
            action_batch = jnp.repeat(actions_for_step, num_particles, axis=0)
            obs, costs, dones = self.step(
                params, normalizer, propagation_id, rng_step, obs, action_batch, goal
            )
            costs = jnp.where(terminated, 0, costs)
            terminated = dones | terminated
            total_costs = total_costs + costs

        total_costs = total_costs.reshape(-1, num_particles)
        return total_costs.mean(axis=1)
