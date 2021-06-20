import functools
from typing import Any, Callable, NamedTuple, Tuple

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


class ModelState(NamedTuple):
    normalizer: Normalizer


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

        def loss(params: hk.Params, normalizer: Normalizer, x, a, xnext) -> jnp.ndarray:
            """Compute the loss of the network, including L2."""
            proc_x = preprocess_obs(x)
            inputs = jnp.concatenate([proc_x, a], axis=-1)
            inputs = normalizer(inputs)
            mean, logvar = self._network.apply(params, inputs)
            target = process_target(x, xnext)
            return gaussian_nll(mean, logvar, target)

        def batched_loss(
            ensem_params: hk.Params, state: ModelState, x, a, xnext
        ) -> jnp.ndarray:
            """Compute the loss of the network, including L2."""
            nll_loss = jax.vmap(loss, (0, None, 0, 0, 0))(
                ensem_params, state.normalizer, x, a, xnext
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

        def batched_eval(ensem_params: hk.Params, state: ModelState, x, a, xnext):
            """Compute the validation loss for the ensembles, MSE
            Args:
              params: ensemble parameters of shape [E, ...]
              normalizer: normalizer for normalizing the inputs
              x, a, x: training data of shape [E, B, ...]
            Returns:
              mse_loss: mean squared error of shape [E] from the ensembles
            """
            losses = jax.vmap(evaluate, (0, None, None, None, None))(
                ensem_params, state.normalizer, x, a, xnext
            )
            # Return the validation MSE per ensemble
            return losses

        self._loss_fn = batched_loss
        self._eval_fn = batched_eval

    @property
    def num_ensembles(self):
        return self._num_ensembles

    def init(self, rng, observation, action) -> Tuple[hk.Params, ModelState]:
        inputs = jnp.concatenate([self._preprocess_obs(observation), action], axis=-1)
        params_list = []
        rngs = jax.random.split(rng, self._num_ensembles)
        for r in rngs:
            params_list.append(self._network.init(r, inputs))
        # pylint: disable=no-value-for-parameter
        ensem_params = jax.tree_multimap(lambda *x: jnp.stack(x), *params_list)

        mean = jnp.zeros(inputs.shape[-1], dtype=jnp.float32)
        std = jnp.ones(inputs.shape[-1], dtype=jnp.float32)
        return ensem_params, ModelState(Normalizer(mean=mean, std=std))

    def apply(self, params: hk.Params, state: ModelState, x, a):
        proc_x = self._preprocess_obs(x)
        inputs = jnp.concatenate([proc_x, a], axis=-1)
        inputs = state.normalizer(inputs)
        mean, logvar = jax.vmap(self._network.apply, (0, None))(params, inputs)
        return mean, logvar

    def update_normalizer(self, state: ModelState, x, a, xnext) -> ModelState:
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
        return state._replace(normalizer=Normalizer(mean=new_mean, std=new_std))

    def propagate(
        self, ensem_params: hk.Params, state: ModelState, key, observations, actions
    ):
        """Propagate the samples through the ensembles.
        Given a batch B of observations and actions, reshape the batch into
        [E, B // E] and propagate the samples through each of the networks in the ensemble.
        """
        assert observations.ndim == 2
        assert actions.ndim == 2
        batch_size = observations.shape[0]

        processed_obs = self._preprocess_obs(observations)
        num_ensembles = tree.flatten(ensem_params)[0].shape[0]
        new_batch_size, ragged = divmod(batch_size, num_ensembles)
        shuffled_inputs = jnp.concatenate([processed_obs, actions], axis=-1)
        normalized_inputs = state.normalizer(shuffled_inputs)
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
        predictions = jax.random.normal(key, shape=mean.shape) * std + mean
        return self._postprocess_obs(observations, predictions)

    @functools.partial(jax.jit, static_argnums=(0,))
    def loss(
        self,
        params: hk.Params,
        state: ModelState,
        observation,
        action,
        next_observation,
    ):
        return self._loss_fn(params, state, observation, action, next_observation)

    @functools.partial(jax.jit, static_argnums=(0,))
    def evaluate(
        self,
        params: hk.Params,
        state: ModelState,
        observation,
        action,
        next_observation,
    ):
        return self._eval_fn(params, state, observation, action, next_observation)


class ModelEnv:
    def __init__(
        self,
        model: EnsembleModel,
        cost_fn: Callable,
        terminal_fn: Callable,
        shuffle: bool = True,
    ):
        self._model = model
        self._shuffle = shuffle
        self._cost_fn = cost_fn
        self._terminal_fn = terminal_fn

    def reset(self, ensem_params: hk.Params, state: ModelState, key, observations):
        num_ensembles = tree.flatten(ensem_params)[0].shape[0]
        batch_size = observations.shape[0]
        if batch_size % num_ensembles:
            raise NotImplementedError("Ragged batch not supported.")
        indices = jnp.arange(batch_size)
        if self._shuffle:
            indices = jax.random.permutation(key, indices)
        return (state, indices)

    def step(
        self,
        ensem_params: hk.Params,
        state: ModelState,
        shuffle_indices,
        key,
        observations,
        actions,
        goal,
    ):
        # TODO(yl): Add sampling propagation indices instead of being fully deterministic
        assert observations.ndim == 2
        assert actions.ndim == 2

        # Shuffle the inputs
        shuffled_observations = observations[shuffle_indices]
        shuffled_actions = actions[shuffle_indices]

        shuffled_output = self._model.propagate(
            ensem_params, state, key, shuffled_observations, shuffled_actions
        )
        # Shuffle back
        next_obs = jax.ops.index_update(
            shuffled_output, shuffle_indices, shuffled_output
        )
        return (
            next_obs,
            self._cost_fn(next_obs, actions, goal),
            self._terminal_fn(next_obs, actions, goal),
        )

    def unroll(
        self,
        params: hk.Params,
        state: ModelState,
        rng: jnp.ndarray,
        initial_state: jnp.ndarray,
        action_sequences: jnp.ndarray,
        goal: Any,
        num_particles: int,
    ):
        """Unroll model along a sequence of actions.
        Args:
          ensem_params: hk.Params.
          rng: JAX random key.
          x_init [B, D]
          actions [B, T, A]
        """
        population_size = action_sequences.shape[0]
        initial_obs_batch = jnp.tile(
            initial_state, (num_particles * population_size, 1)
        ).astype(jnp.float32)
        rng, rng_reset = jax.random.split(rng)
        state, propagation_id = self.reset(params, state, rng_reset, initial_obs_batch)
        batch_size = initial_obs_batch.shape[0]
        total_costs = jnp.zeros((batch_size))
        terminated = jnp.zeros((batch_size), dtype=jnp.bool_)
        obs = initial_obs_batch

        actions = jnp.swapaxes(action_sequences, 0, 1)  # Reshape to [T, B, A]

        def step(inputs, actions_for_step):
            rng, obs, state, terminated, total_costs = inputs
            rng, rng_step = jax.random.split(rng)
            action_batch = jnp.repeat(actions_for_step, num_particles, axis=0)
            obs, costs, dones = self.step(
                params, state, propagation_id, rng_step, obs, action_batch, goal
            )
            costs = jnp.where(terminated, 0, costs)
            terminated = dones | terminated
            total_costs = total_costs + costs
            return (rng, obs, state, terminated, total_costs), None

        output, _ = jax.lax.scan(
            step, (rng, obs, state, terminated, total_costs), actions
        )
        total_costs = output[-1]
        total_costs = total_costs.reshape(-1, num_particles)
        return total_costs.mean(axis=1)
