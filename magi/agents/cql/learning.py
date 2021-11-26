"""Learner component for CQL."""
import functools
import time
from typing import Iterator, NamedTuple, Optional

from acme import core
from acme import types
from acme.jax import networks as networks_lib
from acme.utils import counting
from acme.utils import loggers
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optax


class TrainingState(NamedTuple):
    """Training state for CQL Learner."""

    policy_params: networks_lib.Params
    critic_params: networks_lib.Params
    critic_target_params: networks_lib.Params
    policy_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    alpha_optimizer_state: optax.OptState
    log_alpha: jnp.ndarray
    alpha_prime_optimizer_state: Optional[optax.OptState]
    log_alpha_prime: Optional[jnp.ndarray]
    key: networks_lib.PRNGKey
    steps: int


class CQLLearner(core.Learner):
    """Conservative Q Learning (CQL) learner component.

    This corresponds to CQL(H) agent from [1], with importance sampling
    (min_q_version == 3) according to Appendix F in [1].
    The implementation is based on

        https://github.com/aviralkumar2907/CQL/blob/master/d4rl/rlkit/torch/sac/cql.py

    References:
        [1]: Aviral Kumar and Aurick Zhou and George Tucker and Sergey Levine,
             Conservative Q-Learning for Offline Reinforcement Learning,
             arXiv Pre-print, https://arxiv.org/abs/2006.04779
    """

    def __init__(
        self,
        policy_network: networks_lib.FeedForwardNetwork,
        critic_network: networks_lib.FeedForwardNetwork,
        random_key: networks_lib.PRNGKey,
        dataset: Iterator[types.Transition],
        policy_optimizer: optax.GradientTransformation,
        critic_optimizer: optax.GradientTransformation,
        alpha_optimizer: optax.GradientTransformation,
        target_entropy: float,
        discount: float = 0.99,
        tau: float = 5e-3,
        init_alpha: float = 1.0,
        num_bc_steps: int = 0,
        softmax_temperature: float = 1.0,
        cql_alpha: float = 5.0,
        max_q_backup: bool = False,
        deterministic_backup: bool = True,
        num_cql_samples: int = 10,
        with_lagrange: bool = False,
        target_action_gap: float = 10.0,
        logger: Optional[loggers.Logger] = None,
        counter: Optional[counting.Counter] = None,
    ):
        """Initialize the CQL Learner.

        Args:
            policy_network: policy network
            critic_network: critic network
            random_key: key for random number generation
            dataset: iterator for the training data
            policy_optimizer: optimizer for policy network
            critic_optimizer: optimizer for critic network
            alpha_optimizer: optimizer for SAC alpha "temperature"
            target_entropy: target entropy for automatic entropy tuning
            discount: discount for TD updates.
            tau: coefficient for smoothing target network update.
            init_alpha: Initial alpha.
            num_bc_steps: Number of steps to perform BC on policy update.
            softmax_temperature: temperature for the logsumexp.
            min_q_weight: the value of alpha, set to 5.0 or 10.0 if not using lagrange.
                When adaptive cql weight is used, this determines the minimum
                weight for the cql loss.
            max_q_backup: set this to true to use max_{a} backup.
            deterministic_backup: set this to true to use deterministic backup, i.e.,
                it will not backup the entropy in the Q function.
            num_cql_samples: number of random samples to use for max backup and
                importance sampling.
            with_lagrange: with to use the lagrangian formulation of CQL.
            target_action_gap: Threshold for the lagrangian.
            logger: logger object to write the metrics to.
            counter: counter used for keeping track of the number of steps.

        References:
            Aviral Kumar, Aurick Zhou, George Tucker, Sergey Levine,
            Conservative Q-Learning for Offline Reinforcement Learning
            https://arxiv.org/abs/2006.04779

        """
        if with_lagrange:
            # For now, use the alpha optimizer hyperparams
            alpha_prime_optimizer = optax.adam(3e-4)
        else:
            alpha_prime_optimizer = None

        polyak_average = functools.partial(optax.incremental_update, step_size=tau)

        def sample_action_and_log_prob(
            policy_params: networks_lib.Params,
            key: networks_lib.PRNGKey,
            observation: networks_lib.Observation,
            sample_shape=(),
        ):
            action_dist = policy_network.apply(policy_params, observation)
            action = action_dist.sample(sample_shape, seed=key)
            log_prob = action_dist.log_prob(action)
            return action, log_prob

        def critic_loss_fn(
            critic_params: networks_lib.Params,
            log_alpha_prime: jnp.ndarray,
            critic_target_params: networks_lib.Params,
            policy_params: networks_lib.Params,
            key: networks_lib.PRNGKey,
            log_alpha: jnp.ndarray,
            transitions: types.Transition,
        ):
            # For CQL(H), the loss is
            # min_Q alpha' * [logsumexp(Q(s,a')) - Q(s,a)] + (Q(s, a) - Q(s', a''))^2
            #     = alpha' * cql_loss + critic_loss
            # First compute the SAC critic loss
            alpha = jnp.exp(log_alpha)
            q1_pred, q2_pred = critic_network.apply(
                critic_params, transitions.observation, transitions.action
            )

            if not max_q_backup:
                next_action_key, key = jax.random.split(key)
                new_next_actions, next_log_pi = sample_action_and_log_prob(
                    policy_params, next_action_key, transitions.next_observation
                )
                target_q1, target_q2 = critic_network.apply(
                    critic_target_params,
                    transitions.next_observation,
                    new_next_actions,
                )
                target_q_values = jnp.minimum(target_q1, target_q2)
                if not deterministic_backup:
                    target_q_values = target_q_values - alpha * next_log_pi
            else:
                next_action_key, key = jax.random.split(key)
                # TODO(yl): allow configuting number of actions
                sampled_next_actions, next_log_pi = sample_action_and_log_prob(
                    policy_params,
                    next_action_key,
                    transitions.next_observation,
                    sample_shape=(num_cql_samples,),
                )
                target_q1, target_q2 = jax.vmap(critic_network.apply, (None, None, 0))(
                    critic_target_params,
                    transitions.next_observation,
                    sampled_next_actions,
                )
                target_q1 = jnp.max(target_q1, axis=0)
                target_q2 = jnp.max(target_q2, axis=0)
                target_q_values = jnp.min(target_q1, target_q2)

            q_target = (
                transitions.reward + transitions.discount * discount * target_q_values
            )
            q_target = jax.lax.stop_gradient(q_target)
            qf1_loss = jnp.mean(jnp.square(q1_pred - q_target))
            qf2_loss = jnp.mean(jnp.square(q2_pred - q_target))
            qf_loss = qf1_loss + qf2_loss

            # Next compute the cql_loss
            batch_size = transitions.action.shape[0]
            action_size = transitions.action.shape[-1]
            vmapped_critic_apply = jax.vmap(
                critic_network.apply, (None, None, 0), out_axes=0
            )
            # Compute the logsumexp(Q(s,a')) according to Appendix F
            # for the importance sampled version
            # Sample actions from uniform-at-random distribution
            # (N, B, A)
            uniform_key, policy_key, key = jax.random.split(key, 3)
            cql_random_actions = jax.random.uniform(
                uniform_key,
                shape=(num_cql_samples, batch_size, action_size),
                dtype=transitions.action.dtype,
                maxval=1.0,
                minval=-1.0,
            )
            cql_random_log_probs = jnp.log(0.5 ** action_size)
            # Compute the q values for the uniform actions
            # Sample actions from the policy
            cql_current_actions, cql_current_log_probs = sample_action_and_log_prob(
                policy_params, policy_key, transitions.observation, (num_cql_samples,)
            )
            cql_q1_random, cql_q2_random = vmapped_critic_apply(
                critic_params, transitions.observation, cql_random_actions
            )
            cql_q1_current_actions, cql_q2_current_actions = vmapped_critic_apply(
                critic_params, transitions.observation, cql_current_actions
            )

            # Perform importance sampling
            cat_q1 = jnp.concatenate(
                [
                    cql_q1_random - cql_random_log_probs,
                    cql_q1_current_actions - cql_current_log_probs,
                ],
                axis=0,
            )
            cat_q2 = jnp.concatenate(
                [
                    cql_q2_random - cql_random_log_probs,
                    cql_q2_current_actions - cql_current_log_probs,
                ],
                axis=0,
            )

            logsumexp = jsp.special.logsumexp

            logsumexp_q1 = (
                logsumexp(cat_q1 / softmax_temperature, axis=0) * softmax_temperature
            )
            logsumexp_q2 = (
                logsumexp(cat_q2 / softmax_temperature, axis=0) * softmax_temperature
            )
            cql_q1_loss = jnp.mean(logsumexp_q1 - q1_pred)
            cql_q2_loss = jnp.mean(logsumexp_q2 - q2_pred)
            cql_loss = cql_q1_loss + cql_q2_loss
            alpha_prime = jnp.clip(jnp.exp(log_alpha_prime), 0.0, 10000.0)
            metrics = {
                "qf_loss": qf_loss,
                "cql_loss": cql_loss,
                "q1": jnp.mean(q1_pred),
                "q2": jnp.mean(q2_pred),
                "q1_random": jnp.mean(cql_q1_random),
                "q2_random": jnp.mean(cql_q2_random),
            }
            return qf_loss + alpha_prime * cql_loss, metrics

        def actor_loss_fn(
            policy_params: networks_lib.Params,
            critic_params: networks_lib.Params,
            key: networks_lib.PRNGKey,
            log_alpha: jnp.ndarray,
            observation: jnp.ndarray,
        ):
            alpha = jnp.exp(log_alpha)
            action_dist = policy_network.apply(policy_params, observation)
            new_actions = action_dist.sample(seed=key)
            log_probs = action_dist.log_prob(new_actions)
            q1, q2 = critic_network.apply(critic_params, observation, new_actions)
            q_new_actions = jnp.minimum(q1, q2)
            entropy = -log_probs.mean()
            actor_loss = alpha * log_probs - q_new_actions
            return jnp.mean(actor_loss), {"entropy": entropy}

        def bc_actor_loss_fn(
            policy_params: networks_lib.Params,
            key: networks_lib.PRNGKey,
            log_alpha: jnp.ndarray,
            observations: jnp.ndarray,
            actions: jnp.ndarray,
        ):
            # This is the loss function for pre-training the policy
            action_dist = policy_network.apply(policy_params, observations)
            policy_log_prob = action_dist.log_prob(actions)
            new_actions = action_dist.sample(seed=key)
            log_pi = action_dist.log_prob(new_actions)
            policy_loss = (jnp.exp(log_alpha) * log_pi - policy_log_prob).mean()
            return policy_loss, {"entropy": -log_pi.mean()}

        def alpha_loss_fn(log_alpha: jnp.ndarray, entropy: jnp.ndarray):
            # Use log_alpha here for numerical stability
            return log_alpha * (entropy - target_entropy)

        def alpha_prime_loss_fn(log_alpha_prime: jnp.ndarray, cql_loss: jnp.ndarray):
            alpha_prime = jnp.clip(jnp.exp(log_alpha_prime), 0.0, 10000.0)
            return -alpha_prime * (cql_loss - target_action_gap)

        bc_policy_grad_fn = jax.value_and_grad(bc_actor_loss_fn, has_aux=True)
        policy_grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)

        @jax.jit
        def sgd_step(state: TrainingState, transitions: types.Transition):
            metrics = {}
            # Update critic
            critic_loss_key, key = jax.random.split(state.key)
            (critic_loss, critic_metrics), critic_grads = jax.value_and_grad(
                critic_loss_fn, has_aux=True
            )(
                state.critic_params,
                state.log_alpha_prime,
                state.critic_target_params,
                state.policy_params,
                critic_loss_key,
                state.log_alpha,
                transitions,
            )
            metrics.update({"critic_loss": critic_loss, **critic_metrics})
            critic_updates, critic_optimizer_state = critic_optimizer.update(
                critic_grads, state.critic_optimizer_state
            )
            critic_params = optax.apply_updates(state.critic_params, critic_updates)
            # Update policy
            actor_key, key = jax.random.split(key)
            (policy_loss, actor_metrics), policy_grads = jax.lax.cond(
                state.steps < num_bc_steps,
                lambda _: bc_policy_grad_fn(
                    state.policy_params,
                    actor_key,
                    state.log_alpha,
                    transitions.observation,
                    transitions.action,
                ),
                lambda _: policy_grad_fn(
                    state.policy_params,
                    state.critic_params,
                    actor_key,
                    state.log_alpha,
                    transitions.observation,
                ),
                operand=None,
            )
            policy_updates, policy_optimizer_state = policy_optimizer.update(
                policy_grads, state.policy_optimizer_state
            )
            policy_params = optax.apply_updates(state.policy_params, policy_updates)
            metrics.update({"actor_loss": policy_loss, **actor_metrics})

            # Update entropy alpha
            alpha_loss, grad = jax.value_and_grad(alpha_loss_fn)(
                state.log_alpha, actor_metrics["entropy"]
            )
            alpha_update, alpha_optimizer_state = alpha_optimizer.update(
                grad, state.alpha_optimizer_state
            )
            log_alpha = optax.apply_updates(state.log_alpha, alpha_update)
            metrics.update({"alpha_loss": alpha_loss, "alpha": jnp.exp(log_alpha)})

            # Update adaptive alpha_prime
            if with_lagrange:
                alpha_prime_loss, alpha_prime_grads = jax.value_and_grad(
                    alpha_prime_loss_fn
                )(state.log_alpha_prime, critic_metrics["cql_loss"])
                # pytype: disable=attribute-error
                (
                    alpha_prime_updates,
                    alpha_prime_optimizer_state,
                ) = alpha_prime_optimizer.update(
                    alpha_prime_grads, state.alpha_prime_optimizer_state
                )
                # pytype: enable=attribute-error
                log_alpha_prime = optax.apply_updates(
                    state.log_alpha_prime, alpha_prime_updates
                )
                metrics.update(
                    {
                        "alpha_prime_loss": alpha_prime_loss,
                        "alpha_prime": jnp.exp(log_alpha_prime),
                    }
                )
            else:
                log_alpha_prime = state.log_alpha_prime
                alpha_prime_optimizer_state = None

            # Update target network params
            critic_target_params = polyak_average(
                critic_params, state.critic_target_params
            )
            steps = state.steps + 1
            state = TrainingState(
                policy_params=policy_params,
                critic_params=critic_params,
                critic_target_params=critic_target_params,
                policy_optimizer_state=policy_optimizer_state,
                critic_optimizer_state=critic_optimizer_state,
                alpha_optimizer_state=alpha_optimizer_state,
                log_alpha=log_alpha,
                alpha_prime_optimizer_state=alpha_prime_optimizer_state,
                log_alpha_prime=log_alpha_prime,
                key=key,
                steps=steps,
            )
            return state, metrics

        self._iterator = dataset
        self._logger = logger or loggers.make_default_logger(
            label="learner", save_data=False
        )
        self._counter = counter or counting.Counter()

        self._sgd_step = sgd_step

        def make_initial_state(key):
            init_policy_key, init_critic_key, key = jax.random.split(random_key, 3)
            init_policy_params = policy_network.init(init_policy_key)
            init_critic_params = critic_network.init(init_critic_key)
            init_policy_optimizer_state = policy_optimizer.init(init_policy_params)
            init_critic_optimizer_state = critic_optimizer.init(init_critic_params)
            init_log_alpha = jnp.array(np.log(init_alpha), dtype=jnp.float32)
            init_alpha_optimizer_state = alpha_optimizer.init(init_log_alpha)

            init_log_alpha_prime = jnp.asarray(jnp.log(cql_alpha), dtype=jnp.float32)
            if alpha_prime_optimizer is not None:
                init_alpha_prime_optimizer_state = alpha_prime_optimizer.init(
                    init_log_alpha_prime
                )
            else:
                init_alpha_prime_optimizer_state = None

            return TrainingState(
                policy_params=init_policy_params,
                critic_params=init_critic_params,
                critic_target_params=init_critic_params,
                policy_optimizer_state=init_policy_optimizer_state,
                critic_optimizer_state=init_critic_optimizer_state,
                alpha_optimizer_state=init_alpha_optimizer_state,
                alpha_prime_optimizer_state=init_alpha_prime_optimizer_state,
                log_alpha=init_log_alpha,
                log_alpha_prime=init_log_alpha_prime,
                key=key,
                steps=0,
            )

        self._state = make_initial_state(random_key)

        self._timestamp = None

    def step(self):
        # Get data from replay
        transitions = next(self._iterator)
        # Perform a single learner step
        self._state, metrics = self._sgd_step(self._state, transitions)

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
            "policy": self._state.policy_params,
            "critic": self._state.critic_params,
        }
        return [variables[name] for name in names]

    def save(self) -> TrainingState:
        return self._state

    def restore(self, state: TrainingState):
        self._state = state
