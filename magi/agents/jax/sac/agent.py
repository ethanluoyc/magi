import acme
from acme.agents import agent
from acme.agents.jax import actors
from acme import core, types
from acme.agents import replay
from magi.agents.jax.sac import learning
from magi.agents.jax.sac import acting
from acme.jax import variable_utils
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import dm_env

import tensorflow_probability
import dataclasses

tfp = tensorflow_probability.experimental.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors


def compute_target(
    policy,
    critic,
    rng,
    policy_params,
    critic1_target_params,
    critic2_target_params,
    reward,
    observation_tp1,
    done,
    alpha,
    discount,
):
  mean_tp1, logstd_tp1 = policy.apply(policy_params, observation_tp1)
  action_tp1_base_dist = tfd.MultivariateNormalDiag(loc=mean_tp1,
                                                    scale_diag=jnp.exp(logstd_tp1))
  action_tp1_dist = tfd.TransformedDistribution(action_tp1_base_dist, tfb.Tanh())
  # TODO handle constraints
  # Squash the action to be bounded between [-1, 1]
  squashed_action_tp1 = action_tp1_dist.sample(seed=rng)
  q1 = critic.apply(critic1_target_params, observation_tp1, squashed_action_tp1)
  q2 = critic.apply(critic2_target_params, observation_tp1, squashed_action_tp1)
  return (reward + discount * (1 - done) *
          (jnp.minimum(q1, q2) - alpha * action_tp1_dist.log_prob(squashed_action_tp1)))


def q_value_loss(policy, critic, rng, policy_params, critic1_params, critic2_params,
                 critic1_target_params, critic2_target_params, observation_t, reward,
                 observation_tp1, done, alpha, discount):
  target = compute_target(policy, critic, rng, policy_params, critic1_target_params,
                          critic2_target_params, reward, observation_tp1, done, alpha,
                          discount)
  q1 = critic.apply(critic1_params, observation_t)
  q2 = critic.apply(critic2_params, observation_t)
  q1_loss = jnp.square(q1 - jax.lax.stop_gradient(target))
  q2_loss = jnp.square(q2 - jax.lax.stop_gradient(target))
  return jnp.mean(q1_loss + q2_loss)


@dataclasses.dataclass
class SACConfig:
  """Configuration options for DQN agent."""
  discount: float = 0.99
  n_step: int = 1
  # Replay options
  batch_size: int = 256  # Number of transitions per batch.
  min_replay_size: int = 1  # Minimum replay size.
  max_replay_size: int = 1_000_000  # Maximum replay size.


class SACAgent(agent.Agent):

  def __init__(self,
               environment_spec,
               policy_network,
               critic_network,
               key,
               min_observations=1,
               observations_per_step=1):
    learner_key, actor_key = jax.random.split(key)
    rng = hk.PRNGSequence(actor_key)
    config = SACConfig()
    reverb_replay = replay.make_reverb_prioritized_nstep_replay(
        environment_spec=environment_spec,
        n_step=config.n_step,
        batch_size=config.batch_size,
        max_replay_size=config.max_replay_size,
        min_replay_size=config.min_replay_size,
        discount=config.discount,
    )
    self._server = reverb_replay.server

    # reverb_replay = Replay()
    learner = learning.SACLearner(environment_spec, policy_network, critic_network,
                                  reverb_replay.data_iterator, learner_key)
    variable_client = variable_utils.VariableClient(learner, '')

    @jax.jit
    def forward_fn(params, observations):
      mean, logstd = policy_network.apply(
          params, jax.tree_map(lambda x: jnp.expand_dims(x, 0), observations))
      return jnp.squeeze(mean, 0), jnp.squeeze(logstd, 0)

    actor = acting.SACActor(
        forward_fn,
        rng,
        variable_client,
        reverb_replay.adder,
    )
    super().__init__(
        actor,
        learner,
        min_observations=max(config.batch_size, config.min_replay_size),
        observations_per_step=observations_per_step,
    )
