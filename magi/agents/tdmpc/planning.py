"""TD-MPC inference algorithm."""
from typing import Any, Callable, Tuple

import chex
import jax
import jax.numpy as jnp

Params = Any
Observation = jax.Array
Embedding = jax.Array
Action = jax.Array
Reward = jax.Array
Trajectory = jax.Array

EncoderFn = Callable[[Params, Observation], Embedding]
ModelFn = Callable[[Params, Embedding, Action], Tuple[Embedding, Reward]]
CriticFn = Callable[[Params, Embedding, Action], jax.Array]
PolicyFn = Callable[[Params, Embedding, jax.random.PRNGKeyArray], Action]


def get_initial_trajectory(action_dims: int, horizon: int) -> Trajectory:
    """Create empty planning trajectory."""
    return jnp.zeros((horizon, action_dims))


def td_mpc_planner(
    policy_fn: PolicyFn,
    encoder_fn: EncoderFn,
    model_fn: ModelFn,
    critic_fn: CriticFn,
    params: Params,
    observation: Observation,
    previous_trajectory: Trajectory,
    key: jax.random.PRNGKeyArray,
    *,
    n_policy_trajectories: int,
    n_sample_trajectories: int,
    n_iterations: int,
    k: int,
    discount: float,
    epsilon: float,
    temperature: float,
    momentum: float,
) -> Tuple[Trajectory, Trajectory, jax.Array, Action]:
    """Run the TD-MPC planner.

    This will perform the planning step as described in Alg. 1 of the TD-MPC paper:
    [https://arxiv.org/abs/2203.04955].

    Args:
        policy_fn: policy for selection actions.
        encoder_fn: encoder for encoding observations.
        model_fn: latent dynamics model for computing next latent state and reward.
        critic_fn: Q-function for computing bootstrap return estimate. Expects
            the output to be 1D array of the value estimate. Therefore, twin critic
            should be transformed to adapt to using this function.
        params: network parameters for the learned models. Will be forwarded to
            the policy, encoder, model and critic.
        observation: current observation of shape [O].
        previous_trajectory: array of shape [H, A]. Previous trajectory computed by
            the planner.
        key: JAX random key.
        n_policy_trajectories: number of policy trajectories (N_pi).
        n_sample_trajectories: number of sample trajectories (N).
        n_iterations: number of iterations to run the TD-MPC planner.
        k: top-K number of trajectories used for updating the parameters.
        discount: discount used for computing the value estimate.
        epsilon: minimum stddev for updating the std parameters to prevent pre-mature
            convergence.
        temperature: temperature parameter used for computing the normalized empirical
            estimate of the value of the trajectories.
        momentum: momentum parameter used for updating the mean parameter.

    Returns:
        mean: [H, A] array. Final mean parameter for adapted trajectory distribution.
        std: [H, A] array. Final std parameter for adapted trajectory distribution.
        scores: [k] array. Final normalized empirical value estimate.
        actions: [A] array. The actions selected by the planner.

    """
    chex.assert_rank(previous_trajectory, 2)
    horizon = previous_trajectory.shape[0]
    action_dims = previous_trajectory.shape[1]

    # Encode state z_t = h(s_t)
    observation = jnp.expand_dims(observation, axis=0)  # [1, O]
    z = encoder_fn(params, observation)  # [1, Z]

    policy_actions = None
    if n_policy_trajectories > 0:
        rollout_key, key = jax.random.split(key)
        policy_actions = _rollout_policy(
            params, model_fn, policy_fn, z, rollout_key, horizon, n_policy_trajectories
        )

    # Repeat for sample trajectories
    z = jnp.tile(z, (n_sample_trajectories + n_policy_trajectories, 1))  # [N + N_pi, Z]

    # Reuse the trajectory optimization using previous shifted mean
    initial_mean = jnp.roll(previous_trajectory, shift=-1)  # [H, A]
    initial_mean = initial_mean.at[-1].set(0)

    # Always use large initial variance to avoid local minima
    initial_std = 2 * jnp.ones((horizon, action_dims))  # [H, A]
    initial_state = (
        initial_mean,
        initial_std,
        # Score
        jnp.zeros((k,)),
        # Elite actions
        jnp.zeros((k, horizon, action_dims)),
        # Random key
        key,
    )

    def loop_body_fn(_, state):
        """A single iteration of the TD-MPC planning loop"""
        mean, std, _, _, key = state
        action_key, value_key, key = jax.random.split(key, 3)

        # Sample N trajectories of length H from N(mu^{j-1}, (sigma^{j-1})^2 I)
        action_noise = jax.random.normal(
            action_key, (n_sample_trajectories, horizon, action_dims)
        )  # [N, H, A]
        actions = mean + std * action_noise  # [N, H, A]
        # clip actions to be in the canonical space.
        actions = jnp.clip(actions, -1.0, 1.0)

        # Sample N_pi trajectories of length H using learned model.
        # Reuse samples sampled at the beginning of the iteration.
        if policy_actions is not None:
            actions = jnp.concatenate(
                [actions, policy_actions], axis=0
            )  # [N + N_pi, H, A]

        chex.assert_shape(
            [actions],
            (n_sample_trajectories + n_policy_trajectories, horizon, action_dims),
        )

        # Compute the top-k action trajectories.
        # Estimate trajectory returns (\phi_Gamma) using (d_theta, R_theta, Q_theta)
        # [N + N_pi]
        values = _compute_n_step_return(
            model_fn,
            critic_fn,
            policy_fn,
            params,
            z,
            jnp.swapaxes(actions, 0, 1),
            value_key,
            discount=discount,
        )

        # Avoid NaNs
        # TODO(yl): Replace 0 with something else in case we use negative rewards.
        values = jnp.nan_to_num(values, 0)
        chex.assert_shape([values], (n_sample_trajectories + n_policy_trajectories,))

        ## Update parameters mu, signa for next iteration (Eqn. 5)
        # Select top-k returns trajectories
        # top_k_idxs [K]
        top_k_indices = jax.lax.top_k(values, k)[1]  # [K]

        # Select \phi^* and the corresponding action
        top_k_values = values[top_k_indices]  # [K]
        top_k_actions = actions[top_k_indices]  # [K, H, A]
        chex.assert_shape([top_k_values], (k,))
        chex.assert_shape([top_k_actions], (k, horizon, action_dims))

        ## Update parameters according to (Eqn. 4)
        # Compute the normalized empirical estimate \Omega_i
        max_value = jnp.max(top_k_values, axis=0)
        scores = jnp.exp(temperature * (top_k_values - max_value))  # [K]
        chex.assert_shape([scores], (k,))

        # Omega_i / sum_i(Omega_i)
        scores = scores / jnp.sum(scores, axis=0)  # [K]
        chex.assert_shape([scores], (k,))
        # Expand right-most dimensions of weights to allow broadcasting
        weights = jnp.reshape(
            scores, scores.shape + (1,) * (top_k_actions.ndim - scores.ndim)
        )

        new_mean = jnp.sum(weights * top_k_actions, axis=0) / (
            jnp.sum(scores, axis=0) + 1e-9
        )  # [H, A]
        new_std = jnp.sqrt(
            jnp.sum(weights * jnp.square(top_k_actions - new_mean), axis=0)
            / (jnp.sum(scores, axis=0) + 1e-9)
        )  # [H, A]
        chex.assert_shape([new_mean, new_std], (horizon, action_dims))

        ## Constrain the stddev using Eqn 5.
        #  Eqn. 5 suggests lower-bounding the std to avoid exploration collapse
        #  In the original implementation, the std is also clipped to be below 2 stddev.
        new_std = jnp.clip(new_std, epsilon, 2.0)
        new_mean = momentum * mean + (1 - momentum) * new_mean
        return (new_mean, new_std, scores, top_k_actions, key)

    # For each iteraction j = 1...J do
    mean, std, scores, top_k_actions, key = jax.lax.fori_loop(
        0, n_iterations, loop_body_fn, initial_state
    )

    chex.assert_shape(scores, (k,))  # [K]
    chex.assert_axis_dimension(scores, 0, k)
    actions = top_k_actions[jax.random.choice(key, k, replace=True, p=scores)]  # [H, A]
    return mean, std, scores, actions


def _compute_n_step_return(
    model_fn: ModelFn,
    critic_fn: CriticFn,
    policy_fn: PolicyFn,
    params: Params,
    z: jax.Array,
    actions: jax.Array,
    key: jax.random.PRNGKeyArray,
    *,
    discount: float,
) -> jax.Array:
    horizon = actions.shape[0]
    return_, discount_ = 0, 1

    for t in range(horizon):
        z, reward = model_fn(params, z, actions[t])
        return_ += discount_ * reward
        discount_ *= discount

    return_ += discount_ * critic_fn(params, z, policy_fn(params, z, key))

    return return_


def _rollout_policy(
    params: Params,
    model_fn: ModelFn,
    policy_fn: PolicyFn,
    z: Embedding,
    key: jax.random.PRNGKeyArray,
    horizon: int,
    num_trajectories: int,
) -> Trajectory:
    policy_actions = []
    z = jnp.tile(z, [num_trajectories, 1])
    for _ in range(horizon):
        key_, key = jax.random.split(key)
        actions = policy_fn(params, z, key_)
        policy_actions.append(actions)
        z, _ = model_fn(params, z, actions)
    return jnp.stack(policy_actions, axis=1)
