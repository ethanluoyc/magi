"""JAX feed-forward Critic-Regularized Regression (CRR) learner implementation."""
import copy
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

import acme
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import reverb
import rlax
from acme import types
from acme.jax import networks as networks_lib
from acme.utils import counting
from acme.utils import loggers


class TrainingState(NamedTuple):
    """Training state for CRR agent."""

    policy_params: hk.Params
    critic_params: hk.Params
    policy_opt_state: hk.Params
    critic_opt_state: hk.Params
    policy_target_params: hk.Params
    critic_target_params: hk.Params
    steps: int
    random_key: networks_lib.PRNGKey


class CRRLearner(acme.Learner):
    """CRR Learner component."""

    _state: TrainingState

    def __init__(
        self,
        policy_network: networks_lib.FeedForwardNetwork,
        critic_network: networks_lib.FeedForwardNetwork,
        dataset: Iterator[reverb.ReplaySample],
        random_key: jnp.ndarray,
        policy_optimizer: Optional[optax.GradientTransformation] = None,
        critic_optimizer: Optional[optax.GradientTransformation] = None,
        discount: float = 0.99,
        target_update_period: int = 100,
        num_action_samples_td_learning: int = 1,
        num_action_samples_policy_weight: int = 4,
        baseline_reduce_function: str = "mean",
        policy_improvement_modes: str = "exp",
        ratio_upper_bound: float = 20.0,
        beta: float = 1.0,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
    ):
        if policy_improvement_modes not in {"exp", "binary", "all"}:
            raise ValueError(
                "policy_improvement_modes must be one of ['exp', 'binary', 'all'], "
                f"got {policy_improvement_modes}"
            )
        if baseline_reduce_function not in {"mean", "max", "min"}:
            raise ValueError(
                "baseline_reduce_function must be one of ['mean', 'max', 'min'], "
                f"got {baseline_reduce_function}"
            )
        if policy_optimizer is None:
            policy_optimizer = optax.chain(
                optax.clip_by_global_norm(40.0),
                optax.adam(1e-4),
            )
        if critic_optimizer is None:
            critic_optimizer = optax.chain(
                optax.clip_by_global_norm(40.0),
                optax.adam(1e-4),
            )

        def critic_mean(logits, atoms):
            """Compute the expected value from distributional critic"""
            chex.assert_rank(logits, 2)
            chex.assert_rank(atoms, 1)
            probabilities = jax.nn.softmax(logits)
            return jnp.sum(probabilities * atoms, axis=-1)

        def loss_fn(
            online_params: Tuple[networks_lib.Params, networks_lib.Params],
            target_params: Tuple[networks_lib.Params, networks_lib.Params],
            batch: types.Transition,
            random_key: networks_lib.PRNGKey,
        ):
            """Loss function for CRR."""
            a_t_key, a_tm1_key = jax.random.split(random_key)
            transitions: types.Transition = batch.data
            policy_params, critic_params = online_params
            policy_target_params, critic_target_params = target_params
            o_tm1 = transitions.observation
            a_tm1 = transitions.action
            r_t = transitions.reward
            d_t = discount * transitions.discount
            o_t = transitions.next_observation
            # ========================= Critic learning =============================
            q_tm1_logits, q_tm1_atoms = critic_network.apply(
                critic_params, o_tm1, a_tm1
            )
            target_action_distribution = policy_network.apply(policy_target_params, o_t)
            sampled_actions_t = target_action_distribution.sample(
                num_action_samples_td_learning, seed=a_t_key
            )
            # [N, B, ...]
            # Compute the target critic's Q-value of the sampled actions.
            sampled_q_t_logits, sampled_q_t_atoms = jax.vmap(
                critic_network.apply, (None, None, 0)
            )(critic_target_params, o_t, sampled_actions_t)

            sampled_logits = sampled_q_t_logits
            sampled_logprobs = jax.nn.log_softmax(sampled_logits, axis=-1)
            q_t_logits = jax.scipy.special.logsumexp(sampled_logprobs, axis=0)
            # We take sampled_q_t_atoms[0] here as the vmap returns a batch
            # of same values
            q_t_atoms = sampled_q_t_atoms[0]

            batch_td_learning = jax.vmap(
                rlax.categorical_td_learning, in_axes=(None, 0, 0, 0, None, 0)
            )
            # Construct the expected distributional value for bootstrapping.
            critic_loss_t = batch_td_learning(
                q_tm1_atoms, q_tm1_logits, r_t, d_t, q_t_atoms, q_t_logits
            )
            chex.assert_rank(critic_loss_t, 1)
            critic_loss_t = jnp.mean(critic_loss_t)

            # ========================= Actor learning =============================
            action_distribution_tm1 = policy_network.apply(policy_params, o_tm1)
            # Compute expected q_tm1
            q_tm1_mean = critic_mean(q_tm1_logits, q_tm1_atoms)

            # Compute the estimate of the value function based on
            # self._num_action_samples_policy_weight samples from the policy.
            action_tm1 = action_distribution_tm1.sample(
                num_action_samples_policy_weight, seed=a_tm1_key
            )
            tiled_z_tm1_logits, tiled_z_tm1_values = jax.vmap(
                critic_network.apply, (None, None, 0)
            )(critic_params, o_tm1, action_tm1)
            tiled_v_tm1 = jax.vmap(critic_mean)(tiled_z_tm1_logits, tiled_z_tm1_values)

            # Use mean, min, or max to aggregate Q(s, a_i), a_i ~ pi(s) into the
            # final estimate of the value function.
            if baseline_reduce_function == "mean":
                v_tm1_estimate = jnp.mean(tiled_v_tm1, axis=0)
            elif baseline_reduce_function == "max":
                v_tm1_estimate = jnp.max(tiled_v_tm1, axis=0)
            elif baseline_reduce_function == "min":
                v_tm1_estimate = jnp.min(tiled_v_tm1, axis=0)

            # Assert that action_distribution_tm1 is a batch of multivariate
            # distributions (in contrast to e.g. a [batch, action_size] collection
            # of 1d distributions).
            assert len(action_distribution_tm1.batch_shape) == 1
            policy_loss_batch = -action_distribution_tm1.log_prob(a_tm1)

            advantage = q_tm1_mean - v_tm1_estimate
            if policy_improvement_modes == "exp":
                policy_loss_coef_t = jnp.minimum(
                    jnp.exp(advantage / beta), ratio_upper_bound
                )
            elif policy_improvement_modes == "binary":
                policy_loss_coef_t = (advantage > 0).astype(r_t.dtype)
            elif policy_improvement_modes == "all":
                # Regress against all actions (effectively pure BC).
                policy_loss_coef_t = 1.0
            policy_loss_coef_t = jax.lax.stop_gradient(policy_loss_coef_t)

            policy_loss_batch *= policy_loss_coef_t
            chex.assert_rank(policy_loss_batch, 1)
            policy_loss_t = jnp.mean(policy_loss_batch)

            policy_loss_coef = jnp.mean(policy_loss_coef_t)  # For logging.
            critic_loss = critic_loss_t
            policy_loss = policy_loss_t
            loss = critic_loss + policy_loss
            metrics = {
                "policy_loss": policy_loss,
                "critic_loss": critic_loss,
                "policy_loss_coef": policy_loss_coef,
                "advantages": jnp.mean(advantage),
                "q": jnp.mean(q_tm1_mean),
            }
            return loss, metrics

        def sgd_step(
            state: TrainingState,
            batch: reverb.ReplaySample,
        ) -> Tuple[TrainingState, Dict[str, types.NestedArray]]:
            """Perform a single SGD step."""
            online_params = (state.policy_params, state.critic_params)
            target_params = (state.policy_target_params, state.critic_target_params)
            step_key, random_key = jax.random.split(state.random_key)
            (policy_grads, critic_grads), metrics = jax.grad(loss_fn, has_aux=True)(
                online_params, target_params, batch, step_key
            )
            metrics.update(
                {
                    "policy_g_norm": optax.global_norm(policy_grads),
                    "critic_g_norm": optax.global_norm(critic_grads),
                }
            )
            policy_updates, new_policy_opt_state = policy_optimizer.update(
                policy_grads, state.policy_opt_state
            )
            critic_updates, new_critic_opt_state = critic_optimizer.update(
                critic_grads, state.critic_opt_state
            )
            new_critic_params = optax.apply_updates(state.critic_params, critic_updates)
            new_policy_params = optax.apply_updates(state.policy_params, policy_updates)
            new_policy_target_params, new_critic_target_params = optax.periodic_update(
                (new_policy_params, new_critic_params),
                target_params,
                steps=state.steps,
                update_period=target_update_period,
            )
            new_state = state._replace(
                steps=state.steps + 1,
                random_key=random_key,
                critic_params=new_critic_params,
                policy_params=new_policy_params,
                policy_target_params=new_policy_target_params,
                critic_target_params=new_critic_target_params,
                critic_opt_state=new_critic_opt_state,
                policy_opt_state=new_policy_opt_state,
            )
            return new_state, metrics

        def make_initial_state(key):
            policy_init_key, critic_init_key, random_key = jax.random.split(key, 3)
            initial_policy_params = policy_network.init(policy_init_key)
            initial_policy_opt_state = policy_optimizer.init(initial_policy_params)
            initial_critic_params = critic_network.init(critic_init_key)
            initial_critic_opt_state = critic_optimizer.init(initial_critic_params)
            return TrainingState(
                steps=0,
                random_key=random_key,
                policy_params=initial_policy_params,
                critic_params=initial_critic_params,
                policy_opt_state=initial_policy_opt_state,
                critic_opt_state=initial_critic_opt_state,
                policy_target_params=copy.deepcopy(initial_policy_params),
                critic_target_params=copy.deepcopy(initial_critic_params),
            )

        self._sgd_step = jax.jit(sgd_step)
        self._data_iterator = dataset
        # Initialize state
        self._state = make_initial_state(random_key)
        # Set up logger and counter
        self._logger = logger or loggers.make_default_logger("learner", save_data=False)
        self._counter = counter or counting.Counter()

    def step(self):
        # Sample replay buffer
        batch = next(self._data_iterator)
        self._state, metrics = self._sgd_step(self._state, batch)
        counts = self._counter.increment(steps=1)
        metrics.update(counts)

        self._logger.write(metrics)

    def get_variables(self, names: Sequence[str]) -> List[networks_lib.Params]:
        variables = {
            "policy": self._state.policy_params,
            "critic": self._state.critic_params,
        }
        return [variables[name] for name in names]

    def save(self) -> TrainingState:
        return self._state

    def restore(self, state: TrainingState):
        self._state = state
