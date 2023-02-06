"""Tests for ensemble models."""
import haiku as hk
import jax
import jax.numpy as jnp
import tree
from absl.testing import absltest
from jax import lax

from magi.agents.pets import models


class ModelEnvTest(absltest.TestCase):
    def test_model_env(self):
        """Test that model env works."""
        batch_size, obs_size, act_size = 32, 3, 2
        observations = jnp.zeros((batch_size, obs_size))
        actions = jnp.zeros((batch_size, act_size))

        def network(x, a):
            input_ = jnp.concatenate([x, a], axis=-1)
            model = models.GaussianMLP(obs_size, hidden_sizes=[10])
            return model(input_)

        seed = 0
        num_ensembles = 2
        ensemble = models.ensemble_transform(network, num_ensembles)
        forward_fn = hk.without_apply_rng(hk.transform(network)).apply
        key = jax.random.PRNGKey(seed)
        ensem_params = ensemble.init(key, observations, actions)
        ensem_mean, ensem_std = ensemble.apply(ensem_params, key, observations, actions)

        def step_fn(ensem_params, key, observations, actions):
            """Run a step forward with all of the network ensembles
            Give a batch of o_t and a_t with E ensembles.
            This produces the (E, B, D) next states
            """
            mean, std = ensemble.apply(ensem_params, key, observations, actions)
            return jax.random.normal(key, shape=mean.shape) * std + mean

        def step_fn_same_shape(ensem_params, key, observations, actions):
            num_ensembles = tree.flatten(ensem_params)[0].shape[0]
            batch_size = observations.shape[0]
            new_batch_size, ragged = divmod(batch_size, num_ensembles)
            if ragged:
                raise NotImplementedError("Ragged batch not supported")
            reshaped_obs, reshaped_act = tree.map_structure(
                lambda x: x.reshape((num_ensembles, new_batch_size, *x.shape[1:])),
                (observations, actions),
            )
            mean, std = jax.vmap(forward_fn)(ensem_params, reshaped_obs, reshaped_act)
            mean, std = tree.map_structure(
                lambda x: x.reshape((-1, *x.shape[2:])), (mean, std)
            )
            return jax.random.normal(key, shape=mean.shape) * std + mean

        def unroll(params, rng, x_init, actions):
            """Unroll model along a sequence of actions.
            Args:
              ensem_params: hk.Params.
              rng: JAX random key.
              x_init [B, D]
              actions [T, B, A]
            """

            def step(input_, a_t):
                rng, x_t = input_
                rng, rng_step = jax.random.split(rng)
                x_tp1 = step_fn_same_shape(params, rng_step, x_t, a_t)
                return (rng, x_tp1), x_tp1

            return lax.scan(step, (rng, x_init), actions)

        next_state = step_fn(ensem_params, key, observations, actions)
        self.assertEqual(ensem_mean.shape, (num_ensembles, batch_size, obs_size))
        self.assertEqual(ensem_std.shape, (num_ensembles, batch_size, obs_size))
        self.assertEqual(next_state.shape, (num_ensembles, batch_size, obs_size))

        next_state2 = step_fn_same_shape(ensem_params, key, observations, actions)
        self.assertEqual(next_state2.shape, (batch_size, obs_size))

        num_timesteps = 3
        _, unrolled_states = unroll(
            ensem_params,
            key,
            observations,
            jnp.zeros((num_timesteps, batch_size, act_size)),
        )
        self.assertEqual(unrolled_states.shape, (num_timesteps, batch_size, obs_size))

        def cost_fn(o, a):
            """Cost function for a single step."""
            return jnp.sum(jnp.square(o), axis=-1) + jnp.sum(jnp.square(a), axis=-1)

        def objective(actions, x_init, params, key):
            """Objective function for trajectory planning.
            Args:
              actions: [T, P, A]
              x_init: [D]
              params, key, num_particles
            """
            num_particles = tree.flatten(actions)[0].shape[1]
            # xinit_particles has shape [P, D]
            xinit_particles = jnp.broadcast_to(x_init, (num_particles,) + x_init.shape)
            # unrolled_states has shape [T, P, D]
            _, unrolled_states = unroll(params, key, xinit_particles, actions)
            # costs is [P]
            costs = jnp.sum(jax.vmap(cost_fn)(unrolled_states, actions), axis=0)
            return costs

        num_parts = batch_size
        costs = objective(
            jnp.zeros((num_timesteps, num_parts, act_size)),
            jnp.zeros((obs_size,)),
            ensem_params,
            key,
        )
        self.assertEqual(costs.shape, (num_parts,))


if __name__ == "__main__":
    absltest.main()
