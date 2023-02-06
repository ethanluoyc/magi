"""TDMPC planner"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from magi.agents.tdmpc import planning


class PlannerTest(parameterized.TestCase):
    @parameterized.parameters(
        {"offset": 0.1},
        {"offset": 0.5},
    )
    def test_planner_solve(self, offset):
        horizon, obs_size, act_size = 1, 1, 3

        def policy_fn(params, z, key):
            del params, key
            return jnp.ones(z.shape[:-1] + (act_size,))

        def next_fn(params, z, action):
            del params
            return z, -jnp.sum(jnp.square(action - offset), axis=-1)

        def critic_fn(params, z, action):
            del params, action
            return jnp.zeros(z.shape[:-1])

        def encoder_fn(params, obs):
            del params
            return obs

        params = ()
        key = jax.random.PRNGKey(0)
        observation = jnp.zeros((obs_size,))
        previous_trajectory = planning.get_initial_trajectory(act_size, horizon)
        trajectory_mean, trajectory_std, score, actions = planning.td_mpc_planner(
            policy_fn,
            encoder_fn,
            next_fn,
            critic_fn,
            params,
            observation,
            previous_trajectory,
            key,
            n_policy_trajectories=32,
            n_sample_trajectories=512,
            epsilon=0.0,
            k=64,
            discount=1.0,
            temperature=1.0,
            momentum=0.0,
            n_iterations=20,
        )
        chex.assert_shape(trajectory_mean, (horizon, act_size))
        chex.assert_shape(trajectory_std, (horizon, act_size))
        chex.assert_shape(score, (64,))
        chex.assert_tree_all_close(actions, jnp.full((horizon, act_size), offset))

    def test_compute_n_step_return(self):
        params = ()
        act_size = 1

        def next_fn(params, z, action):
            del params
            return z, jnp.sum(jnp.square(action), axis=-1)

        def policy_fn(params, z, key):
            del params, key
            return jnp.ones(z.shape[:-1] + (act_size,))

        actions = jnp.array([1.0, 2.0, 3.0])
        actions = jnp.reshape(actions, actions.shape + (1, 1))
        horizon = actions.shape[0]
        z = jnp.ones((horizon, 1, 1))

        bootstrap_value = 8.0

        def critic_fn(params, z, action):
            del params, action, z
            return bootstrap_value

        discount = 0.99
        key = jax.random.PRNGKey(0)

        return_prediction = (
            planning._compute_n_step_return(  # pylint: disable=protected-access
                next_fn,
                critic_fn,
                policy_fn,
                params,
                z,
                actions,
                key,
                discount=discount,
            )
        )

        return_expected = (
            jnp.sum(jnp.square(actions[0]))
            + jnp.sum(jnp.square(actions[1])) * (discount**1)
            + jnp.sum(jnp.square(actions[2])) * (discount**2)
            + bootstrap_value * (discount**3)
        )
        np.testing.assert_allclose(return_expected, return_prediction)


if __name__ == "__main__":
    absltest.main()
