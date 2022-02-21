"""Learner component for DrQ."""
from functools import partial
import time
from typing import Iterator, NamedTuple, Optional

from acme import core
from acme import types as acme_types
from acme.jax import types as jax_types
from acme.utils import counting
from acme.utils import loggers
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import reverb

from magi.agents.drq import augmentations


def soft_update(
    target_params: hk.Params,
    online_params: hk.Params,
    tau: float,
) -> hk.Params:
    """
    Update target network using Polyak-Ruppert Averaging.
    """
    return jax.tree_multimap(
        lambda t, s: (1 - tau) * t + tau * s, target_params, online_params
    )


# Loss functions
def make_critic_loss_fn(encoder_apply, actor_apply, critic_apply, gamma):
    def _loss_critic(
        params_critic: hk.Params,
        key: jnp.ndarray,
        params_critic_target: hk.Params,
        params_actor: hk.Params,
        log_alpha: jnp.ndarray,
        data: acme_types.Transition,
    ):
        next_encoded = jax.lax.stop_gradient(
            encoder_apply(params_critic["encoder"], data.next_observation)
        )
        next_dist = actor_apply(params_actor, next_encoded)
        if hasattr(next_dist, "sample_and_log_prob"):
            # Support Distrax sample_and_log_prob shortcut
            next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=key)
        else:
            # Fall back to calling sample and log_prob separately
            next_actions = next_dist.sample(seed=key)
            next_log_probs = next_dist.log_prob(next_actions)

        # Calculate q target values
        next_encoded_target = encoder_apply(
            params_critic_target["encoder"], data.next_observation
        )
        next_q1, next_q2 = critic_apply(
            params_critic_target["critic"], next_encoded_target, next_actions
        )
        next_q = jnp.minimum(next_q1, next_q2)
        next_q -= jnp.exp(log_alpha) * next_log_probs
        target_q = data.reward + data.discount * gamma * next_q
        target_q = jax.lax.stop_gradient(target_q)
        # Calculate predicted Q
        features = encoder_apply(params_critic["encoder"], data.observation)
        q1, q2 = critic_apply(params_critic["critic"], features, data.action)
        chex.assert_rank(
            (next_log_probs, target_q, next_q1, next_q2, q1, q2), [1, 1, 1, 1, 1, 1]
        )
        # abs_td = jnp.abs(target_q - q1)
        loss_critic = (jnp.square(target_q - q1) + jnp.square(target_q - q2)).mean(
            axis=0
        )
        return loss_critic, {"q1": q1.mean(), "q2": q2.mean()}

    return _loss_critic


def make_actor_loss_fn(encoder_apply, actor_apply, critic_apply):
    def _loss_actor(
        params_actor: hk.Params,
        key,
        params_critic: hk.Params,
        log_alpha: jnp.ndarray,
        data: acme_types.Transition,
    ):
        encoded = encoder_apply(params_critic["encoder"], data.observation)
        action_dist = actor_apply(params_actor, encoded)
        if hasattr(action_dist, "sample_and_log_prob"):
            # Support Distrax sample_and_log_prob shortcut
            actions, log_probs = action_dist.sample_and_log_prob(seed=key)
        else:
            # Fall back to calling sample and log_prob separately
            actions = action_dist.sample(seed=key)
            log_probs = action_dist.log_prob(actions)
        q1, q2 = critic_apply(params_critic["critic"], encoded, actions)
        chex.assert_rank((q1, q2, log_probs), [1, 1, 1])
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * jnp.exp(log_alpha) - q).mean(axis=0)
        entropy = -log_probs.mean()
        return actor_loss, {"entropy": entropy}

    return _loss_actor


def _loss_alpha(log_alpha: jnp.ndarray, entropy, target_entropy) -> jnp.ndarray:
    temperature = jnp.exp(log_alpha)
    return temperature * (entropy - target_entropy), {"alpha": temperature}


class TrainingState(NamedTuple):
    """Holds training state for the DrQ learner."""

    actor_params: hk.Params
    actor_opt_state: optax.OptState

    encoder_params: hk.Params
    encoder_target_params: hk.Params
    critic_params: hk.Params
    critic_target_params: hk.Params
    encoder_critic_opt_state: optax.OptState

    log_alpha: jnp.ndarray
    alpha_opt_state: jnp.ndarray
    key: jax_types.PRNGKey

    @property
    def encoder_critic_params(self):
        return {
            "encoder": self.encoder_params,
            "critic": self.critic_params,
        }

    @property
    def encoder_critic_target_params(self):
        return {
            "encoder": self.encoder_target_params,
            "critic": self.critic_target_params,
        }


class DrQLearner(core.Learner):
    """Learner for Data-regularized Q"""

    def __init__(
        self,
        random_key: jnp.ndarray,
        dataset: Iterator[reverb.ReplaySample],
        encoder_network: hk.Transformed,
        policy_network: hk.Transformed,
        critic_network: hk.Transformed,
        target_entropy: float,
        augmentation: Optional[augmentations.DataAugmentation] = None,
        policy_optimizer: Optional[optax.GradientTransformation] = None,
        critic_optimizer: Optional[optax.GradientTransformation] = None,
        temperature_optimizer: Optional[optax.GradientTransformation] = None,
        init_temperature: float = 0.1,
        actor_update_frequency: int = 1,
        critic_target_update_frequency: int = 1,
        critic_soft_update_rate: float = 0.005,
        discount: float = 0.99,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
    ):
        self._iterator = dataset

        self._num_learning_steps = 0
        # Other parameters.
        self._actor_update_frequency = actor_update_frequency
        self._critic_target_update_frequency = critic_target_update_frequency
        self._target_entropy = target_entropy
        self._augmentation = augmentation or augmentations.batched_random_crop
        self._augmentation = jax.jit(augmentation)
        self._counter = counter if counter is not None else counting.Counter()
        self._logger = (
            logger
            if logger is not None
            else loggers.make_default_logger(label="learner", save_data=False)
        )

        # Setup parameters and pure functions
        # Encoder.
        self._encoder = encoder_network
        self._critic = critic_network
        self._actor = policy_network
        self._opt_actor = policy_optimizer or optax.adam(1e-4)
        self._opt_alpha = temperature_optimizer or optax.adam(1e-4)
        self._opt_critic = critic_optimizer or optax.adam(1e-4)

        # Initialize training state
        key1, key2, key3, key = jax.random.split(random_key, 4)
        encoder_init_params = self._encoder.init(key1)
        critic_init_params = self._critic.init(key2)
        encoder_init_target_params = jax.tree_map(
            lambda x: x.copy(), encoder_init_params
        )
        critic_init_target_params = jax.tree_map(lambda x: x.copy(), critic_init_params)

        # Actor.
        actor_init_params = self._actor.init(key3)

        # Entropy coefficient.
        init_log_alpha = jnp.array(np.log(init_temperature), dtype=jnp.float32)

        actor_init_opt_state = self._opt_actor.init(actor_init_params)
        encoder_init_critic_opt_state = self._opt_critic.init(
            {"encoder": encoder_init_params, "critic": critic_init_params}
        )
        alpha_init_opt_state = self._opt_alpha.init(init_log_alpha)

        self._state: TrainingState = TrainingState(
            actor_params=actor_init_params,
            actor_opt_state=actor_init_opt_state,
            encoder_params=encoder_init_params,
            encoder_target_params=encoder_init_target_params,
            critic_params=critic_init_params,
            critic_target_params=critic_init_target_params,
            encoder_critic_opt_state=encoder_init_critic_opt_state,
            log_alpha=init_log_alpha,
            alpha_opt_state=alpha_init_opt_state,
            key=key,
        )

        # Setup losses
        critic_loss_fn = make_critic_loss_fn(
            self._encoder.apply, self._actor.apply, self._critic.apply, discount
        )
        actor_loss_fn = make_actor_loss_fn(
            self._encoder.apply, self._actor.apply, self._critic.apply
        )
        alpha_loss_fn = partial(_loss_alpha, target_entropy=self._target_entropy)

        @jax.jit
        def _update_critic(
            state: TrainingState, key, transitions: acme_types.Transition
        ):
            loss_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
            (loss, aux), grad = loss_grad_fn(
                state.encoder_critic_params,
                key,
                state.encoder_critic_target_params,
                state.actor_params,
                state.log_alpha,
                transitions,
            )
            update, new_opt_state = self._opt_critic.update(
                grad, state.encoder_critic_opt_state
            )
            new_params = optax.apply_updates(state.encoder_critic_params, update)
            return (
                state._replace(
                    critic_params=new_params["critic"],
                    encoder_params=new_params["encoder"],
                    encoder_critic_opt_state=new_opt_state,
                ),
                loss,
                aux,
            )

        @jax.jit
        def _update_actor(
            state: TrainingState, key, transitions: acme_types.Transition
        ):
            loss_grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
            (loss, aux), grad = loss_grad_fn(
                state.actor_params,
                key,
                state.encoder_critic_params,
                state.log_alpha,
                transitions,
            )
            update, new_opt_state = self._opt_actor.update(grad, state.actor_opt_state)
            new_params = optax.apply_updates(state.actor_params, update)
            return (
                state._replace(actor_params=new_params, actor_opt_state=new_opt_state),
                loss,
                aux,
            )

        @jax.jit
        def _update_alpha(state: TrainingState, entropy: jnp.ndarray):
            loss_grad_fn = jax.value_and_grad(alpha_loss_fn, has_aux=True)
            (loss, aux), grad = loss_grad_fn(state.log_alpha, entropy)
            update, new_opt_state = self._opt_alpha.update(grad, state.alpha_opt_state)
            new_log_alpha = optax.apply_updates(state.log_alpha, update)
            return (
                state._replace(log_alpha=new_log_alpha, alpha_opt_state=new_opt_state),
                loss,
                aux,
            )

        @jax.jit
        def _update_target(state: TrainingState):
            update_fn = partial(soft_update, tau=critic_soft_update_rate)
            return state._replace(
                encoder_target_params=update_fn(
                    state.encoder_target_params, state.encoder_params
                ),
                critic_target_params=update_fn(
                    state.critic_target_params, state.critic_params
                ),
            )

        def sgd_step(
            state: TrainingState,
            transitions: acme_types.Transition,
            step: int,
        ):
            key1, key2, key3, key4, key = jax.random.split(state.key, 5)
            # Perform data augmentation on o_tm1 and o_t
            observation = self._augmentation(key1, transitions.observation)
            next_observation = self._augmentation(key2, transitions.next_observation)
            transitions = transitions._replace(
                observation=observation,
                next_observation=next_observation,
            )

            state, loss, critic_metrics = _update_critic(state, key3, transitions)
            metrics = {"critic_loss": loss, **critic_metrics}

            # Update actor and alpha.
            if step % actor_update_frequency == 0:
                state, actor_loss, actor_stats = _update_actor(state, key4, transitions)
                state, alpha_loss, alpha_stats = _update_alpha(
                    state, actor_stats["entropy"]
                )
                metrics["alpha_loss"] = alpha_loss
                metrics["actor_loss"] = actor_loss
                metrics["entropy"] = actor_stats["entropy"]
                metrics["alpha"] = alpha_stats["alpha"]

            # Update target network.
            if step % critic_target_update_frequency == 0:
                state = _update_target(state)

            state = state._replace(key=key)
            return state, metrics

        self._timestamp = None
        self._sgd_step = sgd_step

    def step(self):
        # Get the next batch from the replay iterator
        sample = next(self._iterator)
        transitions: acme_types.Transition = sample.data

        # Perform a single learner step
        self._state, metrics = self._sgd_step(
            self._state, transitions, self._num_learning_steps
        )

        # Compute elapsed time
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp
        self._num_learning_steps += 1
        # Increment counts and record the current time
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        # Attempts to write the logs.
        self._logger.write({**metrics, **counts})

    def get_variables(self, names):
        variables = {
            "policy": {
                "encoder": self._state.encoder_params,
                "actor": self._state.actor_params,
            },
        }
        return [variables[name] for name in names]

    def save(self):
        return self._state

    def restore(self, state):
        self._state = state
