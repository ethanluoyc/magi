"""Data-regularized Q (DrQ) agent."""
import dataclasses
from typing import Any, Mapping, Optional

from acme import core
from acme import datasets
from acme import specs
from acme.adders import reverb as adders
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
from magi.agents.drq import augmentations
from magi.agents.drq import types
from magi.agents.drq.learning import DrQLearner
from magi.agents.sac import acting

batched_random_crop = jax.jit(augmentations.batched_random_crop)


@dataclasses.dataclass
class DrQConfig:
    """Configuration parameters for SAC-AE.

    Notes:
      These parameters are taken from [1].
      Note that hyper-parameters such as log-stddev bounds on the policy should
      be configured in the network builder.
    """

    min_replay_size: int = 1
    max_replay_size: int = 1_000_000
    replay_table_name: str = adders.DEFAULT_PRIORITY_TABLE

    discount: float = 0.99
    batch_size: int = 128
    initial_num_steps: int = 1000

    critic_learning_rate: float = 3e-4
    critic_target_update_frequency: int = 1
    critic_q_soft_update_rate: float = 0.005

    actor_learning_rate: float = 3e-4
    actor_update_frequency: int = 1

    temperature_learning_rate: float = 3e-4
    temperature_adam_b1: float = 0.5
    init_temperature: float = 0.1

    augmentation: types.DataAugmentation = batched_random_crop


class DrQAgent(core.Actor, core.VariableSource):
    """Data-regularized Q agent."""

    def __init__(
        self,
        environment_spec: specs.EnvironmentSpec,
        networks: Mapping[str, Any],
        seed: int,
        config: Optional[DrQConfig] = None,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
    ):
        # Setup reverb
        if config is None:
            config = DrQConfig()
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

        self._rng = hk.PRNGSequence(seed)
        self._initial_num_steps = config.initial_num_steps
        self._num_observations = 0

        self._encoder = hk.without_apply_rng(hk.transform(networks["encoder"]))
        self._actor = hk.without_apply_rng(hk.transform(networks["actor"]))
        self._critic = hk.without_apply_rng(hk.transform(networks["critic"]))

        # Set up learner

        self._learner = DrQLearner(
            environment_spec,
            dataset_iterator=dataset.as_numpy_iterator(),
            random_key=next(self._rng),
            encoder_network=self._encoder,
            policy_network=self._actor,
            critic_network=self._critic,
            policy_optimizer=optax.adam(config.actor_learning_rate),
            critic_optimizer=optax.adam(config.critic_learning_rate),
            temperature_optimizer=optax.adam(
                config.temperature_learning_rate, b1=config.temperature_adam_b1
            ),
            init_temperature=config.init_temperature,
            actor_update_frequency=config.actor_update_frequency,
            critic_target_update_frequency=config.critic_target_update_frequency,
            critic_soft_update_rate=config.critic_q_soft_update_rate,
            discount=config.discount,
            augmentation=config.augmentation,
            counter=counter,
            logger=logger,
        )

        # Setup actors
        # The adder is used to insert observations into replay.
        # discount is 1.0 as we are multiplying gamma during learner step
        adder = adders.NStepTransitionAdder(
            client=reverb.Client(address), n_step=1, discount=1.0
        )

        def forward_fn(params, observation):
            feature_map = self._encoder.apply(params["encoder"], observation)
            return self._actor.apply(params["actor"], feature_map)

        self._forward_fn = forward_fn

        client = variable_utils.VariableClient(self, "")
        self._policy_actor = acting.SACActor(
            self._forward_fn,
            next(self._rng),
            is_eval=False,
            variable_client=client,
            adder=adder,
        )
        self._random_actor = actors.RandomActor(
            environment_spec.actions, next(self._rng), adder=adder
        )

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
        if self._should_update():
            self._learner.step()
            self._policy_actor.update(wait=wait)

    def get_variables(self, names):
        return self._learner.get_variables(names)

    def make_actor(self, is_eval=True):
        client = variable_utils.VariableClient(self, "")
        return acting.SACActor(
            self._forward_fn, next(self._rng), is_eval=is_eval, variable_client=client
        )
