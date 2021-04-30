"""PPO implementation."""
from acme import core
from acme import specs
from acme.adders import reverb as adders
from acme.jax import variable_utils
import haiku as hk
import numpy as np
import optax
import reverb
from reverb import item_selectors
from reverb import rate_limiters

from magi.agents.ppo import acting
from magi.agents.ppo import learning


def _make_queue(name, max_size, extensions=(), signature=None, max_times_sampled=1):
    return reverb.Table(
        name=name,
        sampler=item_selectors.Fifo(),
        remover=item_selectors.Fifo(),
        max_size=max_size,
        max_times_sampled=max_times_sampled,
        rate_limiter=rate_limiters.Queue(max_size),
        extensions=extensions,
        signature=signature,
    )


class PPO(core.Actor, core.VariableSource):
    """Proximal Policy Optimization"""

    def __init__(
        self,
        environment_spec: specs.EnvironmentSpec,
        network_fn,
        sequence_length: int,
        sequence_period: int,
        discount: float = 0.99,
        lr: float = 0.001,
        max_queue_size: int = 1,
        batch_size: int = 1,
        num_update_epochs: int = 10,
        clip_eps: float = 0.2,
        max_gradient_norm=0.5,
        entropy_cost=0.01,
        value_cost=0.5,
        normalize_rew=False,
        seed: int = 0,
        lambd=0.95,
        counter=None,
        logger=None,
    ):
        self.rng = hk.PRNGSequence(seed)
        self.discount = discount
        self.num_update_epochs = num_update_epochs
        self.clip_eps = clip_eps
        self.lambd = lambd
        extra_spec = {
            "logits": specs.Array(
                shape=(environment_spec.actions.num_values,),
                dtype=np.float32,
                name="logits",
            ),
            "value": specs.Array(shape=(), dtype=np.float32, name="value"),
        }
        # Set up Reverb replay
        signature = adders.SequenceAdder.signature(
            environment_spec,
            extra_spec,
        )
        queue = _make_queue(
            name=adders.DEFAULT_PRIORITY_TABLE,
            max_size=max_queue_size,
            signature=signature,
        )
        server = reverb.Server([queue], port=None)
        can_sample = lambda: queue.can_sample(batch_size)  # noqa: E731
        address = f"localhost:{server.port}"
        adder = adders.SequenceAdder(
            client=reverb.Client(address),
            period=sequence_period,
            sequence_length=sequence_length,
            pad_end_of_episode=False,
            break_end_of_episode=False,
        )

        # The dataset object to learn from.
        # We don't use datasets.make_reverb_dataset() here to avoid interleaving
        # and prefetching, that doesn't work well with can_sample() check on update.
        dataset = reverb.ReplayDataset.from_table_signature(
            server_address=address,
            table=adders.DEFAULT_PRIORITY_TABLE,
            max_in_flight_samples_per_worker=1,
            sequence_length=sequence_length,
            emit_timesteps=False,
        )
        dataset = dataset.batch(batch_size, drop_remainder=True)
        iterator = dataset.as_numpy_iterator()

        self._queue = queue
        self._server = server
        self._dataset = dataset
        self._iterator = iterator
        self._adder = adder
        self._can_sample = can_sample

        optimizer = optax.chain(
            optax.clip_by_global_norm(max_gradient_norm),
            optax.adam(lr),
        )

        self._learner = learning.PPOLearner(
            environment_spec.observations,
            network_fn,
            optimizer,
            iterator,
            next(self.rng),
            discount=discount,
            lambd=lambd,
            clip_eps=clip_eps,
            entropy_cost=entropy_cost,
            value_cost=value_cost,
            ppo_epochs=self.num_update_epochs,
            normalize_rew=normalize_rew,
            counter=counter,
            logger=logger,
        )
        variable_client = variable_utils.VariableClient(self._learner, key="")

        self._actor = acting.PPOActor(
            network_fn, variable_client, adder, next(self.rng)
        )

    def observe_first(self, timestep):
        # self._timestep = preprocess_step(timestep)
        self._actor.observe_first(timestep)

    def observe(self, action, next_timestep):
        self._actor.observe(action, next_timestep)

    def select_action(self, observation):
        return self._actor.select_action(observation)

    def update(self, wait: bool = True):
        # NOTE: Unlike IMPALA,
        # `wait` is set to be true since we are implementing synchronous PPO
        should_update_actor = False
        # Run a number of learner steps (usually gradient steps).
        while self._can_sample():
            self._learner.step()
            should_update_actor = True
        if should_update_actor:
            # Update actor weights after learner.
            self._actor.update(wait)

    def get_variables(self, names):
        return self._learner.get_variables(names)
