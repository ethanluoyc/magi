"""JAX feed-forward Critic-Regularized Regression (CRR) learner implementation."""
import copy
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

import acme
from acme import specs
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils as jax_utils
from acme.utils import counting
from acme.utils import loggers
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import reverb

from magi.agents.crr import distributions
from magi.agents.crr import losses


class TrainingState(NamedTuple):
    policy_params: hk.Params
    critic_params: hk.Params
    policy_opt_state: hk.Params
    critic_opt_state: hk.Params
    policy_target_params: hk.Params
    critic_target_params: hk.Params
    steps: int
    random_key: networks_lib.PRNGKey


class CRRLearner(acme.Learner):
    def __init__(
        self,
        environment_spec: specs.EnvironmentSpec,
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
        self._policy_optimizer = (
            policy_optimizer
            if policy_optimizer
            else optax.chain(optax.clip_by_global_norm(40.0), optax.adam(1e-4))
        )
        self._critic_optimizer = (
            critic_optimizer
            if critic_optimizer
            else optax.chain(optax.clip_by_global_norm(40.0), optax.adam(1e-4))
        )
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
        # Internalize agent parameters
        self._environment_spec = environment_spec
        self._policy_network = policy_network
        self._critic_network = critic_network
        self._data_iterator = dataset
        self._discount = discount
        self._target_update_period = target_update_period
        self._beta = beta
        self._num_action_samples_td_learning = num_action_samples_td_learning
        self._num_action_samples_policy_weight = num_action_samples_policy_weight
        self._baseline_reduce_function = baseline_reduce_function
        self._policy_improvement_modes = policy_improvement_modes
        self._ratio_upper_bound = ratio_upper_bound
        self._logger = logger or loggers.make_default_logger("learner", save_data=False)
        self._counter = counter or counting.Counter()

        # Initialize state
        policy_init_key, critic_init_key, random_key = jax.random.split(random_key, 3)
        obs_spec = jax_utils.add_batch_dim(
            jax_utils.zeros_like(environment_spec.observations)
        )
        action_spec = jax_utils.add_batch_dim(
            jax_utils.zeros_like(environment_spec.actions)
        )
        initial_policy_params = policy_network.init(policy_init_key, obs_spec)
        initial_policy_opt_state = self._policy_optimizer.init(initial_policy_params)
        initial_critic_params = critic_network.init(
            critic_init_key, obs_spec, action_spec
        )
        initial_critic_opt_state = self._critic_optimizer.init(initial_critic_params)
        self._state = TrainingState(
            steps=0,
            random_key=random_key,
            policy_params=initial_policy_params,
            critic_params=initial_critic_params,
            policy_opt_state=initial_policy_opt_state,
            critic_opt_state=initial_critic_opt_state,
            policy_target_params=copy.deepcopy(initial_policy_params),
            critic_target_params=copy.deepcopy(initial_critic_params),
        )

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
            d_t = transitions.discount
            o_t = transitions.next_observation
            # ========================= Critic learning =============================
            q_tm1 = self._critic_network.apply(critic_params, o_tm1, a_tm1)
            q_tm1 = distributions.DiscreteValuedDistribution(
                values=q_tm1.values, logits=q_tm1.logits
            )
            target_action_distribution = self._policy_network.apply(
                policy_target_params, o_t
            )
            sampled_actions_t = target_action_distribution.sample(
                self._num_action_samples_td_learning, seed=a_t_key
            )
            # [N, B, ...]
            tiled_o_t = jax_utils.tile_nested(o_t, self._num_action_samples_td_learning)
            # Compute the target critic's Q-value of the sampled actions.
            sampled_q_t = jax.vmap(self._critic_network.apply, (None, 0, 0))(
                critic_target_params, tiled_o_t, sampled_actions_t
            )
            # Compute average logits by first reshaping them to [N, B, A] and then
            # normalizing them across atoms.
            new_shape = [self._num_action_samples_td_learning, r_t.shape[0], -1]
            sampled_logits = jnp.reshape(sampled_q_t.logits, new_shape)
            sampled_logprobs = jax.nn.log_softmax(sampled_logits, axis=-1)
            averaged_logits = jax.scipy.special.logsumexp(sampled_logprobs, axis=0)

            # Construct the expected distributional value for bootstrapping.
            # We take sampled_q_t.values[0] here as the vmap returns a batch
            # of same values while losses.categorical expects 1D support
            q_t = distributions.DiscreteValuedDistribution(
                values=sampled_q_t.values[0], logits=averaged_logits
            )
            critic_loss_t = losses.categorical(q_tm1, r_t, discount * d_t, q_t)
            assert len(critic_loss_t.shape) == 1
            critic_loss_t = jnp.mean(critic_loss_t)

            # ========================= Actor learning =============================
            action_distribution_tm1 = self._policy_network.apply(policy_params, o_tm1)
            q_tm1_mean = q_tm1.mean()

            # Compute the estimate of the value function based on
            # self._num_action_samples_policy_weight samples from the policy.
            tiled_o_tm1 = jax_utils.tile_nested(
                o_tm1, self._num_action_samples_policy_weight
            )
            action_tm1 = action_distribution_tm1.sample(
                self._num_action_samples_policy_weight, seed=a_tm1_key
            )
            tiled_z_tm1_logits, tiled_z_tm1_values = jax.vmap(
                self._critic_network.apply, (None, 0, 0)
            )(critic_params, tiled_o_tm1, action_tm1)
            tiled_z_tm1 = distributions.DiscreteValuedDistribution(
                values=tiled_z_tm1_values[0],
                logits=tiled_z_tm1_logits,
            )
            tiled_v_tm1 = jnp.reshape(
                tiled_z_tm1.mean(),
                [self._num_action_samples_policy_weight, -1],
            )

            # Use mean, min, or max to aggregate Q(s, a_i), a_i ~ pi(s) into the
            # final estimate of the value function.
            if self._baseline_reduce_function == "mean":
                v_tm1_estimate = jnp.mean(tiled_v_tm1, axis=0)
            elif self._baseline_reduce_function == "max":
                v_tm1_estimate = jnp.max(tiled_v_tm1, axis=0)
            elif self._baseline_reduce_function == "min":
                v_tm1_estimate = jnp.min(tiled_v_tm1, axis=0)

            # Assert that action_distribution_tm1 is a batch of multivariate
            # distributions (in contrast to e.g. a [batch, action_size] collection
            # of 1d distributions).
            assert len(action_distribution_tm1.batch_shape) == 1
            policy_loss_batch = -action_distribution_tm1.log_prob(a_tm1)

            advantage = q_tm1_mean - v_tm1_estimate
            if self._policy_improvement_modes == "exp":
                policy_loss_coef_t = jnp.minimum(
                    jnp.exp(advantage / self._beta), self._ratio_upper_bound
                )
            elif self._policy_improvement_modes == "binary":
                policy_loss_coef_t = (advantage > 0).astype(r_t.dtype)
            elif self._policy_improvement_modes == "all":
                # Regress against all actions (effectively pure BC).
                policy_loss_coef_t = 1.0
            policy_loss_coef_t = jax.lax.stop_gradient(policy_loss_coef_t)

            policy_loss_batch *= policy_loss_coef_t
            assert len(policy_loss_batch.shape) == 1
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
            policy_updates, new_policy_opt_state = self._policy_optimizer.update(
                policy_grads, state.policy_opt_state
            )
            critic_updates, new_critic_opt_state = self._critic_optimizer.update(
                critic_grads, state.critic_opt_state
            )
            new_critic_params = optax.apply_updates(state.critic_params, critic_updates)
            new_policy_params = optax.apply_updates(state.policy_params, policy_updates)
            new_policy_target_params, new_critic_target_params = optax.periodic_update(
                (new_policy_params, new_critic_params),
                target_params,
                steps=state.steps,
                update_period=self._target_update_period,
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

        self._sgd_step = jax.jit(sgd_step)

    def step(self):
        # Sample replay buffer
        batch = next(self._data_iterator)
        self._state, metrics = self._sgd_step(self._state, batch)
        counts = self._counter.increment(steps=1)
        metrics.update(counts)

        self._logger.write(metrics)

    def get_variables(self, names: Sequence[str]) -> List[networks_lib.Params]:
        del names
        return [self._state.policy_params]
