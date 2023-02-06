"""Loss functions for Soft Actor-Critic."""
import haiku as hk
import jax
import jax.numpy as jnp
from acme import types


def alpha_loss_fn(
    log_alpha: jnp.ndarray, entropy: jnp.ndarray, target_entropy: float
) -> jnp.ndarray:
    "Compute the temperature loss for EC-SAC."
    return log_alpha * (entropy - target_entropy), ()


def actor_loss_fn(
    actor,
    critic,
    actor_params: hk.Params,
    key,
    critic_params: hk.Params,
    log_alpha: jnp.ndarray,
    observation: jnp.ndarray,
):
    "Compute the soft actor loss in SAC."
    action_dist = actor.apply(actor_params, observation)
    actions = action_dist.sample(seed=key)
    log_probs = action_dist.log_prob(actions)

    q1, q2 = critic.apply(critic_params, observation, actions)
    q = jnp.minimum(q1, q2)
    entropy = -log_probs.mean()
    actor_loss = jnp.exp(log_alpha) * log_probs - q
    return jnp.mean(actor_loss), {"entropy": entropy}


def critic_loss_fn(
    actor,
    critic,
    critic_params: hk.Params,
    key: jax.random.PRNGKey,
    critic_target_params: hk.Params,
    actor_params: hk.Params,
    log_alpha: jnp.ndarray,
    transitions: types.Transition,
    gamma: float,
):
    "Compute the soft critic loss in SAC."
    data = transitions
    next_action_dist = actor.apply(actor_params, data.next_observation)
    next_actions = next_action_dist.sample(seed=key)
    next_log_probs = next_action_dist.log_prob(next_actions)

    next_q1, next_q2 = critic.apply(
        critic_target_params, data.next_observation, next_actions
    )
    next_q = jnp.minimum(next_q1, next_q2)
    next_q -= jnp.exp(log_alpha) * next_log_probs
    target = jax.lax.stop_gradient(data.reward + data.discount * gamma * next_q)
    q1, q2 = critic.apply(critic_params, data.observation, data.action)
    critic_loss = jnp.square(target - q1) + jnp.square(target - q2)
    return jnp.mean(critic_loss), {"q1": q1.mean(), "q2": q2.mean()}
