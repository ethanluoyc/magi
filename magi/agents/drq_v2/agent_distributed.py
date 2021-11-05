"""Distributed DrQ-V2 agent implementation."""
from typing import Callable, Optional

from acme import specs
from acme.utils import loggers
import dm_env
import optax

from magi.agents.drq_v2 import builder as drq_v2_builder
from magi.agents.drq_v2 import config as drq_v2_config
from magi.agents.drq_v2 import networks as drq_v2_networks
from magi.layouts import distributed_layout


class DistributedDrQV2(distributed_layout.DistributedLayout):
    """Program definition for distributed DrQ-V2."""

    def __init__(
        self,
        seed: int,
        environment_factory: Callable[[int, bool], dm_env.Environment],
        network_factory: Callable[
            [specs.EnvironmentSpec], drq_v2_networks.DrQV2Networks
        ],
        config: drq_v2_config.DrQV2Config,
        num_actors: int = 1,
        environment_spec: specs.EnvironmentSpec = None,
        max_actor_steps: Optional[int] = None,
        log_every: float = 5.0,
        logger_fn=loggers.make_default_logger,
    ):

        if not logger_fn:
            logger_fn = loggers.make_default_logger

        if not environment_spec:
            environment_spec = specs.make_environment_spec(
                environment_factory(seed, True)
            )

        learner_logger_fn = lambda: logger_fn(  # noqa
            "learner",
            time_delta=self._log_every,
            save_data=True,
            steps_key="learner_steps",
        )

        builder = drq_v2_builder.DrQV2Builder(config, learner_logger_fn)
        self._builder = builder

        def policy_factory(networks):
            # TODO(yl): figure out a good strategy for stddev decay
            # in distributed setting.
            sigma_start, sigma_end, sigma_steps = config.sigma
            sigma_steps = sigma_steps // num_actors
            return drq_v2_networks.get_default_behavior_policy(
                networks,
                environment_spec.actions,
                optax.linear_schedule(sigma_start, sigma_end, sigma_steps),
            )

        def evaluator_policy_factory(networks):
            return drq_v2_networks.get_default_behavior_policy(
                networks,
                environment_spec.actions,
                optax.constant_schedule(0.0),
            )

        evaluator_factory = distributed_layout.default_evaluator(
            environment_factory,
            network_factory,
            builder,
            evaluator_policy_factory,
            logger_fn=logger_fn,
        )

        super().__init__(
            seed=seed,
            environment_factory=environment_factory,
            network_factory=network_factory,
            policy_factory=policy_factory,
            builder=builder,
            num_actors=num_actors,
            environment_spec=environment_spec,
            evaluator_factories=(evaluator_factory,),
            max_actor_steps=max_actor_steps,
            prefetch_size=config.prefetch_size,
            log_every=log_every,
            logger_fn=logger_fn,
        )
