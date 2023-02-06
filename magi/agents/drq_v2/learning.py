"""Learner component for DrQV2."""
import time
from functools import partial
from typing import Iterator, NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax
import reverb
from acme import core
from acme import types as acme_types
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers

from magi.agents.drq import augmentations
from magi.agents.drq_v2 import networks as drq_v2_networks


def _soft_update(
    target_params: networks_lib.Params,
    online_params: networks_lib.Params,
    tau: float,
) -> networks_lib.Params:
    """
    Update target network using Polyak-Ruppert Averaging.
    """
    return jax.tree_map(
        lambda t, s: (1 - tau) * t + tau * s, target_params, online_params
    )


class TrainingState(NamedTuple):
    """Holds training state for the DrQ learner."""

    policy_params: networks_lib.Params
    policy_opt_state: optax.OptState

    encoder_params: networks_lib.Params
    # There is not target encoder parameters in v2.
    encoder_opt_state: optax.OptState

    critic_params: networks_lib.Params
    critic_target_params: networks_lib.Params
    critic_opt_state: optax.OptState

    key: jax_types.PRNGKey
    steps: int


class DrQV2Learner(core.Learner):
    """Learner for DrQ-v2"""

    def __init__(
        self,
        random_key: jax_types.PRNGKey,
        dataset: Iterator[reverb.ReplaySample],
        networks: drq_v2_networks.DrQV2Networks,
        sigma_schedule: optax.Schedule,
        augmentation: augmentations.DataAugmentation,
        policy_optimizer: optax.GradientTransformation,
        critic_optimizer: optax.GradientTransformation,
        encoder_optimizer: optax.GradientTransformation,
        noise_clip: float = 0.3,
        critic_soft_update_rate: float = 0.005,
        discount: float = 0.99,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
    ):
        def critic_loss_fn(
            critic_params: networks_lib.Params,
            encoder_params: networks_lib.Params,
            critic_target_params: networks_lib.Params,
            policy_params: networks_lib.Params,
            transitions: acme_types.Transition,
            key: jax_types.PRNGKey,
            sigma: jnp.ndarray,
        ):
            next_encoded = networks.encoder_network.apply(
                encoder_params, transitions.next_observation
            )
            next_action = networks.policy_network.apply(policy_params, next_encoded)
            next_action = networks.add_policy_noise(next_action, key, sigma, noise_clip)
            next_q1, next_q2 = networks.critic_network.apply(
                critic_target_params, next_encoded, next_action
            )
            # Calculate q target values
            next_q = jnp.minimum(next_q1, next_q2)
            target_q = transitions.reward + transitions.discount * discount * next_q
            target_q = jax.lax.stop_gradient(target_q)
            # Calculate predicted Q
            encoded = networks.encoder_network.apply(
                encoder_params, transitions.observation
            )
            q1, q2 = networks.critic_network.apply(
                critic_params, encoded, transitions.action
            )
            loss_critic = (jnp.square(target_q - q1) + jnp.square(target_q - q2)).mean(
                axis=0
            )
            return loss_critic, {"q1": q1.mean(), "q2": q2.mean()}

        def policy_loss_fn(
            policy_params: networks_lib.Params,
            critic_params: networks_lib.Params,
            encoder_params: networks_lib.Params,
            observation: acme_types.Transition,
            sigma: jnp.ndarray,
            key,
        ):
            encoded = networks.encoder_network.apply(encoder_params, observation)
            action = networks.policy_network.apply(policy_params, encoded)
            action = networks.add_policy_noise(action, key, sigma, noise_clip)
            q1, q2 = networks.critic_network.apply(critic_params, encoded, action)
            q = jnp.minimum(q1, q2)
            policy_loss = -q.mean()
            return policy_loss, {}

        policy_grad_fn = jax.value_and_grad(policy_loss_fn, has_aux=True)
        critic_grad_fn = jax.value_and_grad(
            critic_loss_fn, argnums=(0, 1), has_aux=True
        )

        def update_step(
            state: TrainingState,
            transitions: acme_types.Transition,
        ):
            key_aug1, key_aug2, key_policy, key_critic, key = jax.random.split(
                state.key, 5
            )
            sigma = sigma_schedule(state.steps)
            # Perform data augmentation on o_tm1 and o_t
            observation_aug = augmentation(key_aug1, transitions.observation)
            next_observation_aug = augmentation(key_aug2, transitions.next_observation)
            transitions = transitions._replace(
                observation=observation_aug,
                next_observation=next_observation_aug,
            )
            # Update critic
            (critic_loss, critic_aux), (critic_grad, encoder_grad) = critic_grad_fn(
                state.critic_params,
                state.encoder_params,
                state.critic_target_params,
                state.policy_params,
                transitions,
                key_critic,
                sigma,
            )
            encoder_update, encoder_opt_state = encoder_optimizer.update(
                encoder_grad, state.encoder_opt_state
            )
            critic_update, critic_opt_state = critic_optimizer.update(
                critic_grad, state.critic_opt_state
            )
            encoder_params = optax.apply_updates(state.encoder_params, encoder_update)
            critic_params = optax.apply_updates(state.critic_params, critic_update)
            # Update policy
            (policy_loss, policy_aux), actor_grad = policy_grad_fn(
                state.policy_params,
                critic_params,
                encoder_params,
                observation_aug,
                sigma,
                key_policy,
            )
            policy_update, policy_opt_state = policy_optimizer.update(
                actor_grad, state.policy_opt_state
            )
            policy_params = optax.apply_updates(state.policy_params, policy_update)

            # Update target parameters
            polyak_update_fn = partial(_soft_update, tau=critic_soft_update_rate)

            critic_target_params = polyak_update_fn(
                state.critic_target_params,
                critic_params,
            )
            metrics = {
                "policy_loss": policy_loss,
                "critic_loss": critic_loss,
                "sigma": sigma,
                **critic_aux,
                **policy_aux,
            }
            new_state = TrainingState(
                policy_params=policy_params,
                policy_opt_state=policy_opt_state,
                encoder_params=encoder_params,
                encoder_opt_state=encoder_opt_state,
                critic_params=critic_params,
                critic_target_params=critic_target_params,
                critic_opt_state=critic_opt_state,
                key=key,
                steps=state.steps + 1,
            )
            return new_state, metrics

        self._iterator = dataset
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger(
            label="learner",
            save_data=False,
            asynchronous=True,
            serialize_fn=utils.fetch_devicearray,
        )
        self._update_step = jax.jit(update_step)

        # Initialize training state
        def make_initial_state(key: jax_types.PRNGKey):
            key_encoder, key_critic, key_policy, key = jax.random.split(key, 4)
            encoder_init_params = networks.encoder_network.init(key_encoder)
            encoder_init_opt_state = encoder_optimizer.init(encoder_init_params)

            critic_init_params = networks.critic_network.init(key_critic)
            critic_init_opt_state = critic_optimizer.init(critic_init_params)

            policy_init_params = networks.policy_network.init(key_policy)
            policy_init_opt_state = policy_optimizer.init(policy_init_params)

            return TrainingState(
                policy_params=policy_init_params,
                policy_opt_state=policy_init_opt_state,
                encoder_params=encoder_init_params,
                critic_params=critic_init_params,
                critic_target_params=critic_init_params,
                encoder_opt_state=encoder_init_opt_state,
                critic_opt_state=critic_init_opt_state,
                key=key,
                steps=0,
            )

        # Create initial state.
        self._state = make_initial_state(random_key)

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp = None

    def step(self):
        # Get the next batch from the replay iterator
        sample = next(self._iterator)
        transitions: acme_types.Transition = sample.data

        # Perform a single learner step
        self._state, metrics = self._update_step(self._state, transitions)

        # Compute elapsed time
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        # Increment counts and record the current time
        counts = self._counter.increment(steps=1, walltime=elapsed_time)

        # Attempts to write the logs.
        self._logger.write({**metrics, **counts})

    def get_variables(self, names):
        variables = {
            "policy": {
                "encoder": self._state.encoder_params,
                "policy": self._state.policy_params,
            },
        }
        return [variables[name] for name in names]

    def save(self) -> TrainingState:
        return self._state

    def restore(self, state: TrainingState) -> None:
        self._state = state
