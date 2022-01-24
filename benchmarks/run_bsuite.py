"""Run SAC on bsuite."""

from typing import Optional
from absl import app
from absl import flags
import acme
from acme import specs
from acme import wrappers
import numpy as np
import tensorflow as tf

from magi.agents import impala
import bsuite

import haiku as hk
import jax.numpy as jnp
import numpy as np
from acme.jax import networks

from magi.agents import impala


FLAGS = flags.FLAGS
flags.DEFINE_integer("num_steps", int(1e6), "")
flags.DEFINE_integer("max_replay_size", 100000, "Maximum replay size")
flags.DEFINE_integer("min_replay_size", 1000, "Minimum replay size")
flags.DEFINE_integer("batch_size", 128, "Batch size")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("bsuite_id", "catch/0", "Bsuite id.")
flags.DEFINE_string("results_dir", "/tmp/bsuite", "CSV results directory.")
flags.DEFINE_boolean("overwrite", False, "Whether to overwrite csv results.")


class MyNetwork(hk.RNNCore):
    """A simple recurrent network for testing."""

    def __init__(self, num_actions: int):
        super().__init__(name="my_network")
        self._torso = hk.Sequential(
            [
                hk.Flatten(),
                hk.nets.MLP([50, 50]),
            ]
        )
        self._core = hk.LSTM(20)
        self._head = networks.PolicyValueHead(num_actions)

    def __call__(self, inputs, state):
        embeddings = self.embed(inputs)
        embeddings, new_state = self._core(embeddings, state)
        logits, value = self._head(embeddings)
        return (logits, value), new_state

    def initial_state(self, batch_size: int):
        return self._core.initial_state(batch_size)

    def embed(self, observation):
        if observation.ndim not in [2, 3]:
            raise ValueError(
                "Expects inputs to have rank 3 (unbatched) or 4 (batched), "
                f"got {observation.ndim} instead"
            )
        expand_obs = observation.ndim == 2
        if expand_obs:
            observation = jnp.expand_dims(observation, 0)
        features = self._torso(observation.astype(jnp.float32))
        if expand_obs:
            features = jnp.squeeze(features, 0)
        return features

    def unroll(self, inputs, initial_state, start_of_episode=None):
        embeddings = self.embed(inputs)
        if start_of_episode is not None:
            core_input = (embeddings, start_of_episode)
            core = hk.ResetCore(self._core)
        else:
            core_input = embeddings
            core = self._core
        initial_state = hk.LSTMState(initial_state.hidden, initial_state.cell)
        core_outputs, final_state = hk.static_unroll(core, core_input, initial_state)
        return self._head(core_outputs), final_state


def main(_):
    np.random.seed(FLAGS.seed)

    raw_environment = bsuite.load_and_record_to_csv(
        bsuite_id=FLAGS.bsuite_id,
        results_dir=FLAGS.results_dir,
        overwrite=FLAGS.overwrite,
    )
    env = wrappers.SinglePrecisionWrapper(raw_environment)
    spec = specs.make_environment_spec(env)

    def forward_fn(x, s):
        model = MyNetwork(spec.actions.num_values)
        return model(x, s)

    def initial_state_fn(batch_size: Optional[int] = None):
        model = MyNetwork(spec.actions.num_values)
        return model.initial_state(batch_size)

    def unroll_fn(inputs, state, start_of_episode=None):
        model = MyNetwork(spec.actions.num_values)
        return model.unroll(inputs, state, start_of_episode)

    # Construct the agent.
    agent = impala.IMPALAFromConfig(
        environment_spec=spec,
        forward_fn=forward_fn,
        initial_state_fn=initial_state_fn,
        unroll_fn=unroll_fn,
        config=impala.IMPALAConfig(
            sequence_length=3,
            sequence_period=3,
            batch_size=6,
        ),
    )

    loop = acme.EnvironmentLoop(env, agent)
    loop.run(num_episodes=env.bsuite_num_episodes)  # pytype: disable=attribute-error


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")
    app.run(main)
