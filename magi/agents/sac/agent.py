"""Soft Actor-Critic implementation"""
from typing import Optional, Sequence

from acme import core
from acme import datasets
from acme import specs
from acme.adders import reverb as adders
from acme.jax import networks as network_lib
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import dm_env
import haiku as hk
import jax
import optax
import reverb
from reverb import rate_limiters

from magi.agents import actors
from magi.agents.sac import acting as acting_lib
from magi.agents.sac import config as sac_config
from magi.agents.sac import learning as learning_lib


class SACAgentFromConfig(core.Actor):
    def __init__(
        self,
        environment_spec: specs.EnvironmentSpec,
        policy: hk.Transformed,
        critic: hk.Transformed,
        seed: int,
        config: sac_config.SACConfig,
        logger: Optional[loggers.Logger] = None,
        counter: Optional[counting.Counter] = None,
    ):
        learner_key, actor_key, random_key = jax.random.split(
            jax.random.PRNGKey(seed), 3
        )
        self._num_observations = 0
        self._initial_num_steps = config.initial_num_steps

        replay_table = reverb.Table(
            name=config.replay_table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=config.max_replay_size,
            rate_limiter=rate_limiters.MinSize(config.min_replay_size),
            signature=adders.NStepTransitionAdder.signature(
                environment_spec=environment_spec
            ),
        )
        self._server = reverb.Server([replay_table], port=None)

        # The adder is used to insert observations into replay.
        # discount is 1.0 as we are multiplying gamma during learner step
        address = f"localhost:{self._server.port}"
        # The dataset provides an interface to sample from replay.
        dataset = datasets.make_reverb_dataset(
            server_address=address,
            batch_size=config.batch_size,
            prefetch_size=config.prefetch_size,
            transition_adder=True,
        )
        critic_opt = optax.chain(
            optax.clip_by_global_norm(config.max_gradient_norm),
            optax.adam(config.critic_learning_rate),
        )
        actor_opt = optax.chain(
            optax.clip_by_global_norm(config.max_gradient_norm),
            optax.adam(config.actor_learning_rate),
        )
        alpha_opt = optax.chain(
            optax.clip_by_global_norm(config.max_gradient_norm),
            optax.adam(config.temperature_learning_rate, b1=config.temperature_adam_b1),
        )

        self._learner = learning_lib.SACLearner(
            environment_spec,
            policy,
            critic,
            key=learner_key,
            dataset=dataset.as_numpy_iterator(),
            actor_optimizer=actor_opt,
            critic_optimizer=critic_opt,
            alpha_optimizer=alpha_opt,
            gamma=config.discount,
            tau=config.critic_soft_update_rate,
            init_alpha=config.init_temperature,
            logger=logger,
            counter=counter,
        )

        adder = adders.NStepTransitionAdder(
            client=reverb.Client(address), n_step=1, discount=1.0
        )

        client = variable_utils.VariableClient(self._learner, "")
        self.policy = policy
        self._actor = acting_lib.SACActor(
            policy.apply, actor_key, is_eval=False, variable_client=client, adder=adder
        )
        self._random_actor = actors.RandomActor(
            environment_spec.actions, random_key, adder=adder
        )

    def select_action(self, observation: network_lib.Observation) -> network_lib.Action:
        return self._get_active_actor().select_action(observation)

    def observe_first(self, timestep: dm_env.TimeStep):
        return self._get_active_actor().observe_first(timestep)

    def observe(self, action: network_lib.Action, next_timestep: dm_env.TimeStep):
        self._num_observations += 1
        self._get_active_actor().observe(action, next_timestep)

    def _get_active_actor(self):
        if self._num_observations < self._initial_num_steps:
            return self._random_actor
        else:
            return self._actor

    def update(self):
        if self._num_observations < self._initial_num_steps:
            return
        self._learner.step()
        self._actor.update(wait=True)

    def get_variables(self, names: Sequence[str]):
        return self._learner.get_variables(names)

    def make_actor(self, key, is_eval=True):
        client = variable_utils.VariableClient(self, "")
        return acting_lib.SACActor(
            self.policy.apply, key, is_eval=is_eval, variable_client=client
        )


class SACAgent(SACAgentFromConfig):
    def __init__(
        self,
        environment_spec: specs.EnvironmentSpec,
        policy: hk.Transformed,
        critic: hk.Transformed,
        seed: int,
        discount: float = 0.99,
        max_replay_size: int = int(10 ** 6),
        batch_size: int = 256,
        initial_num_steps: int = 10000,
        tau: float = 5e-3,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        temperature_learning_rate: float = 3e-4,
        init_temperature: float = 1.0,
        temperature_adam_b1: float = 0.9,
        logger: Optional[loggers.Logger] = None,
        counter: Optional[counting.Counter] = None,
    ):
        config = sac_config.SACConfig(
            discount=discount,
            max_replay_size=max_replay_size,
            batch_size=batch_size,
            initial_num_steps=initial_num_steps,
            critic_soft_update_rate=tau,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            temperature_learning_rate=temperature_learning_rate,
            init_temperature=init_temperature,
            temperature_adam_b1=temperature_adam_b1,
        )
        super().__init__(
            environment_spec,
            policy,
            critic,
            seed,
            config,
            logger=logger,
            counter=counter,
        )
