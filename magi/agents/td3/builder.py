"""TD3 builder"""
from typing import Dict, Iterator, List, Optional

import reverb
from acme import adders
from acme import core
from acme import datasets
from acme import specs
from acme.adders import reverb as adders_reverb
from acme.agents.jax import actor_core
from acme.agents.jax import actors as acting_lib
from acme.agents.jax import builders
from acme.jax import networks as networks_lib
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers

from magi.agents.td3 import config as td3_config
from magi.agents.td3 import learning as learning_lib

TD3Networks = Dict[str, networks_lib.FeedForwardNetwork]


class TD3Builder(builders.ActorLearnerBuilder):
    """Builder for creating TD3 agent."""

    def __init__(self, config: td3_config.TD3Config):
        self._config = config

    def make_replay_tables(
        self,
        environment_spec: specs.EnvironmentSpec,
        policy: actor_core.FeedForwardPolicy,
    ) -> List[reverb.Table]:
        del policy
        replay_table = reverb.Table(
            name=self._config.replay_table_name,
            # TODO(yl): support prioritized sampling in SAC
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=reverb.rate_limiters.MinSize(self._config.min_replay_size),
            signature=adders_reverb.NStepTransitionAdder.signature(
                environment_spec=environment_spec
            ),
        )
        return [replay_table]

    def make_dataset_iterator(
        self,
        replay_client: reverb.Client,
    ) -> Iterator[reverb.ReplaySample]:
        dataset = datasets.make_reverb_dataset(
            table=self._config.replay_table_name,
            server_address=replay_client.server_address,
            batch_size=self._config.batch_size,
            prefetch_size=self._config.prefetch_size,
            transition_adder=True,
        )
        return dataset.as_numpy_iterator()

    def make_adder(
        self,
        replay_client: reverb.Client,
        environment_spec: Optional[specs.EnvironmentSpec],
        policy: Optional[actor_core.FeedForwardPolicy],
    ) -> Optional[adders.Adder]:
        del environment_spec, policy
        # TODO(yl): support multi step transitions
        return adders_reverb.NStepTransitionAdder(
            client=replay_client, n_step=1, discount=self._config.discount
        )

    def make_actor(
        self,
        random_key: networks_lib.PRNGKey,
        policy: actor_core.FeedForwardPolicy,
        environment_spec: specs.EnvironmentSpec,
        variable_source: Optional[core.VariableSource] = None,
        adder: Optional[adders.Adder] = None,
    ) -> core.Actor:
        del environment_spec

        assert variable_source is not None
        variable_client = variable_utils.VariableClient(variable_source, "policy")
        variable_client.update_and_wait()
        return acting_lib.GenericActor(
            actor=actor_core.batched_feed_forward_to_actor_core(policy),
            random_key=random_key,
            variable_client=variable_client,
            adder=adder,
        )

    def make_learner(
        self,
        random_key: networks_lib.PRNGKey,
        networks: TD3Networks,
        dataset: Iterator[reverb.ReplaySample],
        logger_fn: loggers.LoggerFactory,
        environment_spec: specs.EnvironmentSpec,
        replay_client: Optional[reverb.Client] = None,
        counter: Optional[counting.Counter] = None,
    ) -> core.Learner:
        del environment_spec, replay_client
        return learning_lib.TD3Learner(
            policy_network=networks["policy"],
            critic_network=networks["critic"],
            iterator=dataset,
            random_key=random_key,
            policy_optimizer=self._config.policy_optimizer,
            critic_optimizer=self._config.critic_optimizer,
            discount=self._config.discount,
            soft_update_rate=self._config.soft_update_rate,
            policy_noise=self._config.policy_noise,
            policy_noise_clip=self._config.policy_noise_clip,
            policy_target_update_period=self._config.policy_update_period,
            logger=logger_fn("learner"),
            counter=counter,
        )
