"""SAC-AE agent."""
import dataclasses
from functools import partial
import time
from typing import Any, Mapping, Optional

from acme import core
from acme import datasets
from acme import specs
from acme import types
from acme.adders import reverb as adders
from acme.agents.jax import actor_core
from acme.agents.jax import actors as acme_actors
from acme.jax import utils
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import reverb
from reverb import rate_limiters

from magi.agents import actors
from magi.agents.sac import losses


def apply_policy_sample(encoder_network, actor_network, eval_mode: bool):
    def forward_fn(params, key, observation):
        feature_map = encoder_network.apply(params["encoder"], observation)
        action_dist = actor_network.apply(params["actor"], feature_map)
        return action_dist.mode() if eval_mode else action_dist.sample(seed=key)

    return forward_fn


@jax.jit
def preprocess_observation(
    observation: np.ndarray,
    key: jnp.ndarray,
) -> jnp.ndarray:
    """
    Preprocess pixel states to fit into [-0.5, 0.5].
    """
    observation = observation.astype(jnp.float32)
    observation = jnp.floor(observation / 8)
    observation = observation / 32
    observation = observation + jax.random.uniform(key, observation.shape) / 32
    observation = observation - 0.5
    return observation


@jax.jit
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


@jax.jit
def weight_decay(params: hk.Params) -> jnp.ndarray:
    """
    Calculate the sum of L2 norms of parameters.
    """
    leaves, _ = jax.tree_flatten(params)
    return 0.5 * sum(jnp.vdot(x, x) for x in leaves)


# Loss functions
def make_critic_loss_fn(encoder_apply, actor_apply, linear_apply, critic_apply, gamma):
    def _loss_critic(
        params_critic: hk.Params,
        key,
        params_critic_target: hk.Params,
        params_actor: hk.Params,
        log_alpha: jnp.ndarray,
        batch,
    ):
        data: types.Transition = batch.data
        last_conv = encoder_apply(params_critic["encoder"], data.observation)
        next_last_conv = jax.lax.stop_gradient(
            encoder_apply(params_critic["encoder"], data.next_observation)
        )
        next_action, next_log_pi = actor_apply(
            params_actor, next_last_conv
        ).sample_and_log_prob(key)
        # Compute target
        next_last_conv_target = encoder_apply(
            params_critic_target["encoder"], data.next_observation
        )
        next_feature_target = linear_apply(
            params_critic_target["linear"], next_last_conv_target
        )
        next_q1, next_q2 = critic_apply(
            params_critic_target["critic"], next_feature_target, next_action
        )
        next_q = jnp.minimum(next_q1, next_q2)
        next_q -= jnp.exp(log_alpha) * next_log_pi
        target = jax.lax.stop_gradient(data.reward + data.discount * gamma * next_q)
        feature = linear_apply(params_critic["linear"], last_conv)
        q1, q2 = critic_apply(params_critic["critic"], feature, data.action)
        return (jnp.square(target - q1) + jnp.square(target - q2)).mean(), {
            "q1": q1.mean(),
            "q2": q2.mean(),
        }

    return _loss_critic


def make_actor_loss_fn(encoder_apply, actor_apply, linear_apply, critic_apply):
    @jax.jit
    def _loss_actor(
        params_actor: hk.Params,
        key,
        params_critic: hk.Params,
        log_alpha: jnp.ndarray,
        observation: np.ndarray,
    ):
        last_conv = jax.lax.stop_gradient(
            encoder_apply(params_critic["encoder"], observation)
        )
        action, log_pi = actor_apply(params_actor, last_conv).sample_and_log_prob(key)

        feature = linear_apply(params_critic["linear"], last_conv)
        q1, q2 = critic_apply(params_critic["critic"], feature, action)
        q = jnp.minimum(q1, q2)
        entropy = -log_pi.mean()
        return (jnp.exp(log_alpha) * log_pi - q).mean(), entropy

    return _loss_actor


def make_ae_loss_fn(
    encoder_apply,
    linear_apply,
    decoder_apply,
    lambda_latent,
    lambda_weight,
    preprocess_target_fn,
):
    def _loss_ae(params_ae: hk.Params, key: jnp.ndarray, observation: np.ndarray):
        # Preprocess states.
        target = preprocess_target_fn(observation, key)
        # Reconstruct states.
        last_conv = encoder_apply(params_ae["encoder"], observation)
        feature = linear_apply(params_ae["linear"], last_conv)
        reconst = decoder_apply(params_ae["decoder"], feature)
        # MSE for reconstruction errors.
        loss_reconst = jnp.square(target - reconst).mean()
        # L2 penalty of latent representations following RAE.
        loss_latent = 0.5 * jnp.square(feature).sum(axis=1).mean()
        # Weight decay for the decoder.
        loss_weight = weight_decay(params_ae["decoder"])
        return (
            loss_reconst + lambda_latent * loss_latent + lambda_weight * loss_weight,
            (),
        )

    return _loss_ae


@dataclasses.dataclass
class SACAEConfig:
    """Configuration parameters for SAC-AE.

    Notes:
      These parameters are taken from [1].
      Note that hyper-parameters such as log-stddev bounds on the policy should
      be configured in the network builder.

    [1]: Improving Sample Efficiency in Model-Free Reinforcement Learning from Images,
      https://arxiv.org/abs/1910.01741
    """

    min_replay_size: int = 1
    max_replay_size: int = 1_000_000
    replay_table_name: str = adders.DEFAULT_PRIORITY_TABLE

    discount: float = 0.99
    batch_size: int = 128
    initial_num_steps: int = 1000

    critic_learning_rate: float = 1e-3
    critic_target_update_frequency: int = 2
    critic_q_soft_update_rate: float = 0.01
    critic_encoder_soft_update_rate: float = 0.05

    actor_learning_rate: float = 1e-3
    actor_update_frequency: int = 2
    # log_std_bounds: Tuple[float, float] = [-10, 2]

    # TODO(yl): Handle maximum gradient norm
    max_grad_norm: float = np.inf

    autoencoder_learning_rate: float = 1e-3
    encoder_update_frequency: int = 1

    temperature_learning_rate: float = 1e-4
    temperature_adam_b1: float = 0.5
    init_temperature: float = 0.1

    lambda_latent: float = 1e-6
    lambda_weight: float = 1e-7


class SACAEAgent(core.Actor, core.VariableSource):
    """SAC-AE agent."""

    def __init__(
        self,
        environment_spec: specs.EnvironmentSpec,
        networks: Mapping[str, Any],
        seed: int,
        config: Optional[SACAEConfig] = None,
        target_processor=preprocess_observation,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
    ):
        # Setup reverb
        if config is None:
            config = SACAEConfig()
        replay_table = reverb.Table(
            name=adders.DEFAULT_PRIORITY_TABLE,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=config.max_replay_size,
            rate_limiter=rate_limiters.MinSize(config.min_replay_size),
            signature=adders.NStepTransitionAdder.signature(
                environment_spec=environment_spec
            ),
        )

        # Hold a reference to server to prevent from being gc'ed.
        self._server = reverb.Server([replay_table], port=None)

        address = f"localhost:{self._server.port}"
        # The dataset provides an interface to sample from replay.
        dataset = datasets.make_reverb_dataset(
            server_address=address, batch_size=config.batch_size, transition_adder=True
        )

        self._iterator = dataset.as_numpy_iterator()

        self._rng = hk.PRNGSequence(seed)
        self._initial_num_steps = config.initial_num_steps
        self._num_observations = 0
        self._num_learning_steps = 0
        self._max_grad_norm = config.max_grad_norm
        # Other parameters.
        self._actor_update_frequency = config.actor_update_frequency
        self._encoder_update_frequency = config.encoder_update_frequency
        self._critic_target_update_frequency = config.critic_target_update_frequency
        self._counter = counter if counter is not None else counting.Counter()
        self._logger = (
            logger
            if logger is not None
            else loggers.make_default_logger(label="learner", save_data=False)
        )

        example_obs = utils.add_batch_dim(
            utils.zeros_like(environment_spec.observations)
        )
        example_action = utils.add_batch_dim(utils.zeros_like(environment_spec.actions))
        # Setup parameters and pure functions
        # Encoder.
        self._encoder = hk.without_apply_rng(hk.transform(networks["encoder"]))
        self._encoder_params = self._encoder_target_params = self._encoder.init(
            next(self._rng), example_obs
        )
        example_last_conv = self._encoder.apply(self._encoder_params, example_obs)

        # Linear layer for critic and decoder.
        self._linear = hk.without_apply_rng(hk.transform(networks["linear"]))
        self._linear_params = self._linear_target_params = self._linear.init(
            next(self._rng), example_last_conv
        )
        example_latent = self._linear.apply(self._linear_params, example_last_conv)

        # Critic from latent to Q values
        self._critic = hk.without_apply_rng(hk.transform(networks["critic"]))
        self._critic_params = self._critic_target_params = self._critic.init(
            next(self._rng), example_latent, example_action
        )

        # Actor.
        self._actor = hk.without_apply_rng(hk.transform(networks["actor"]))
        self._actor_params = self._actor.init(next(self._rng), example_last_conv)
        self._opt_actor = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.actor_learning_rate),
        )
        self._opt_state_actor = self._opt_actor.init(self._actor_params)

        # Entropy coefficient.
        self._log_alpha = jnp.array(np.log(config.init_temperature), dtype=jnp.float32)
        self._opt_alpha = optax.adam(
            config.temperature_learning_rate, b1=config.temperature_adam_b1
        )
        self._opt_state_alpha = self._opt_alpha.init(self._log_alpha)

        # Decoder.
        self._decoder = hk.without_apply_rng(hk.transform(networks["decoder"]))
        self._decoder_params = self._decoder.init(next(self._rng), example_latent)
        self._opt_ae = optax.adam(config.autoencoder_learning_rate)
        self._opt_state_ae = self._opt_ae.init(self._params_ae)

        # Critic
        self._opt_critic = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.critic_learning_rate),
        )
        self._opt_state_critic = self._opt_critic.init(self._params_entire_critic)

        # Setup losses
        self._critic_loss_fn = make_critic_loss_fn(
            self._encoder.apply,
            self._actor.apply,
            self._linear.apply,
            self._critic.apply,
            config.discount,
        )
        self._actor_loss_fn = make_actor_loss_fn(
            self._encoder.apply,
            self._actor.apply,
            self._linear.apply,
            self._critic.apply,
        )
        self._ae_loss_fn = make_ae_loss_fn(
            self._encoder.apply,
            self._linear.apply,
            self._decoder.apply,
            config.lambda_latent,
            config.lambda_weight,
            target_processor,
        )

        @jax.jit
        def _update_actor(
            params_actor, opt_state, key, params_critic, log_alpha, observation
        ):
            def loss_fn(actor_params):
                return self._actor_loss_fn(
                    actor_params, key, params_critic, log_alpha, observation
                )

            (loss, aux), grad = jax.value_and_grad(loss_fn, has_aux=True)(params_actor)
            update, opt_state = self._opt_actor.update(grad, opt_state)
            params_actor = optax.apply_updates(params_actor, update)
            return params_actor, opt_state, loss, aux

        @jax.jit
        def _update_critic(
            params_critic,
            opt_state,
            key,
            critic_target_params,
            actor_params,
            log_alpha,
            batch,
        ):
            def loss_fn(critic_params):
                return self._critic_loss_fn(
                    critic_params,
                    key,
                    critic_target_params,
                    actor_params,
                    log_alpha,
                    batch,
                )

            (loss, aux), grad = jax.value_and_grad(loss_fn, has_aux=True)(params_critic)
            update, opt_state = self._opt_critic.update(grad, opt_state)
            params_critic = optax.apply_updates(params_critic, update)
            return params_critic, opt_state, loss, aux

        @jax.jit
        def _update_ae(ae_params, opt_state, key, observation):
            def loss_fn(ae_params):
                return self._ae_loss_fn(ae_params, key, observation)

            (loss, aux), grad = jax.value_and_grad(loss_fn, has_aux=True)(ae_params)
            update, opt_state = self._opt_ae.update(grad, opt_state)
            new_params = optax.apply_updates(ae_params, update)
            return new_params, opt_state, loss, aux

        target_entropy = -float(np.prod(environment_spec.actions.shape))

        @jax.jit
        def _update_alpha(log_alpha, opt_state, entropy):
            def loss_fn(log_alpha):
                return losses.alpha_loss_fn(
                    log_alpha, entropy, target_entropy=target_entropy
                )

            (loss, aux), grad = jax.value_and_grad(loss_fn, has_aux=True)(log_alpha)
            update, opt_state = self._opt_alpha.update(grad, opt_state)
            log_alpha = optax.apply_updates(log_alpha, update)
            return log_alpha, opt_state, loss, aux

        self._update_actor = _update_actor
        self._update_critic = _update_critic
        self._update_alpha = _update_alpha
        self._update_ae = _update_ae

        # Update functions for target networks.
        self._update_encoder_target = jax.jit(
            partial(soft_update, tau=config.critic_encoder_soft_update_rate)
        )
        self._update_critic_target = jax.jit(
            partial(soft_update, tau=config.critic_q_soft_update_rate)
        )

        # Setup actors
        self._client = variable_utils.VariableClient(self, "policy")
        # The adder is used to insert observations into replay.
        # discount is 1.0 as we are multiplying gamma during learner step
        adder = adders.NStepTransitionAdder(
            client=reverb.Client(address), n_step=1, discount=1.0
        )

        self._policy_actor = acme_actors.GenericActor(
            actor_core.batched_feed_forward_to_actor_core(
                apply_policy_sample(self._encoder, self._actor, eval_mode=False)
            ),
            random_key=next(self._rng),
            variable_client=self._client,
            adder=adder,
        )
        self._random_actor = actors.RandomActor(
            environment_spec.actions, next(self._rng), adder=adder
        )
        # Set up actor for running evaluation loop

    def select_action(self, observation):
        return self._get_active_actor().select_action(observation)

    def observe_first(self, timestep: dm_env.TimeStep):
        return self._get_active_actor().observe_first(timestep)

    def observe(self, action, next_timestep: dm_env.TimeStep):
        self._num_observations += 1
        self._get_active_actor().observe(action, next_timestep)

    def _get_active_actor(self):
        if self._num_observations < self._initial_num_steps:
            return self._random_actor
        else:
            return self._policy_actor

    def _should_update(self):
        return self._num_observations >= self._initial_num_steps

    def update(self, wait: bool = True):
        if not self._should_update():
            return
        batch = next(self._iterator)
        start = time.time()

        metrics = {}
        # Update critic.
        (
            params_entire_critic,
            self._opt_state_critic,
            loss_critic,
            critic_metrics,
        ) = self._update_critic(
            self._params_entire_critic,
            self._opt_state_critic,
            key=next(self._rng),
            critic_target_params=self._params_entire_critic_target,
            actor_params=self._actor_params,
            log_alpha=self._log_alpha,
            batch=batch,
        )
        self._encoder_params = params_entire_critic["encoder"]
        self._linear_params = params_entire_critic["linear"]
        self._critic_params = params_entire_critic["critic"]
        metrics["critic_loss"] = loss_critic
        metrics["q1"] = critic_metrics["q1"]
        metrics["q2"] = critic_metrics["q2"]

        # Update actor and alpha.
        if self._num_learning_steps % self._actor_update_frequency == 0:
            (
                self._actor_params,
                self._opt_state_actor,
                loss_actor,
                entropy,
            ) = self._update_actor(
                self._actor_params,
                self._opt_state_actor,
                key=next(self._rng),
                params_critic=self._params_entire_critic,
                log_alpha=self._log_alpha,
                observation=batch.data.observation,
            )
            self._log_alpha, self._opt_state_alpha, loss_alpha, _ = self._update_alpha(
                self._log_alpha, self._opt_state_alpha, entropy=entropy
            )
            metrics["entropy"] = entropy
            metrics["alpha_loss"] = loss_alpha
            metrics["actor_loss"] = loss_actor
            metrics["alpha"] = jnp.exp(self._log_alpha)

        # Update autoencoder.
        if self._num_learning_steps % self._encoder_update_frequency == 0:
            params_ae, self._opt_state_ae, loss_ae, _ = self._update_ae(
                self._params_ae,
                self._opt_state_ae,
                next(self._rng),
                batch.data.observation,
            )
            self._encoder_params = params_ae["encoder"]
            self._linear_params = params_ae["linear"]
            self._decoder_params = params_ae["decoder"]
            metrics["autoencoder_loss"] = loss_ae

        # Update target network.
        if self._num_learning_steps % self._critic_target_update_frequency == 0:
            self._encoder_target_params = self._update_encoder_target(
                self._encoder_target_params, self._encoder_params
            )
            self._linear_target_params = self._update_encoder_target(
                self._linear_target_params, self._linear_params
            )
            self._critic_target_params = self._update_critic_target(
                self._critic_target_params, self._critic_params
            )
        self._policy_actor.update(wait=wait)

        self._num_learning_steps += 1
        metrics = utils.to_numpy(metrics)
        counts = self._counter.increment(steps=1, time_elapsed=time.time() - start)

        self._logger.write({**counts, **metrics})

    @property
    def _params_ae(self):
        return {
            "encoder": self._encoder_params,
            "linear": self._linear_params,
            "decoder": self._decoder_params,
        }

    @property
    def _params_entire_critic(self):
        return {
            "encoder": self._encoder_params,
            "linear": self._linear_params,
            "critic": self._critic_params,
        }

    @property
    def _params_entire_critic_target(self):
        return {
            "encoder": self._encoder_target_params,
            "linear": self._linear_target_params,
            "critic": self._critic_target_params,
        }

    def get_variables(self, names):
        del names
        return [{"encoder": self._encoder_params, "actor": self._actor_params}]
