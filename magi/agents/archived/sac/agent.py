import dataclasses

from acme.adders import reverb as adders
from acme.agents import agent
from acme import datasets
from acme.jax import variable_utils
from acme import types
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import reverb
from reverb import rate_limiters
import tensorflow_probability

from magi.agents.archived.sac import acting
from magi.agents.archived.sac import learning

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
  num_seed_steps: int = 5000


class SACAgent(agent.Agent):

  def __init__(self,
               environment_spec,
               policy_network,
               critic_network,
               key,
               logger=None):
    learner_key, actor_key, explore_key = jax.random.split(key, 3)
    rng = hk.PRNGSequence(actor_key)
    config = SACConfig()
    self._num_seed_steps = config.num_seed_steps
    replay_table = reverb.Table(name=adders.DEFAULT_PRIORITY_TABLE,
                                sampler=reverb.selectors.Uniform(),
                                remover=reverb.selectors.Fifo(),
                                max_size=config.max_replay_size,
                                rate_limiter=rate_limiters.MinSize(1),
                                signature=adders.NStepTransitionAdder.signature(
                                    environment_spec=environment_spec))
    self._server = reverb.Server([replay_table], port=None)

    # The adder is used to insert observations into replay.
    address = f'localhost:{self._server.port}'
    adder = adders.NStepTransitionAdder(client=reverb.Client(address),
                                        n_step=config.n_step,
                                        discount=config.discount)

    # The dataset provides an interface to sample from replay.
    dataset = datasets.make_reverb_dataset(server_address=address,
                                           environment_spec=environment_spec,
                                           batch_size=config.batch_size,
                                           prefetch_size=1,
                                           transition_adder=True)
    self._batch_size = config.batch_size

    learner = learning.SACLearner(environment_spec,
                                  policy_network,
                                  critic_network,
                                  dataset.as_numpy_iterator(),
                                  learner_key,
                                  logger=logger)
    self._learner = learner
    variable_client = variable_utils.VariableClient(learner, '')

    @jax.jit
    def forward_fn(params, observations):
      mean, logstd = policy_network.apply(
          params, jax.tree_map(lambda x: jnp.expand_dims(x, 0), observations))
      return jnp.squeeze(mean, 0), jnp.squeeze(logstd, 0)

    self._actor = acting.SACActor(
        forward_fn,
        rng,
        variable_client,
        adder,
    )
    self._random_actor = acting.RandomActor(environment_spec.actions,
                                            rng=hk.PRNGSequence(explore_key),
                                            adder=adder)
    self._num_observations = 0

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    if self._num_observations < self._num_seed_steps:
      return self._random_actor.select_action(observation)
    else:
      return self._actor.select_action(observation)

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._num_observations < self._num_seed_steps:
      self._random_actor.observe_first(timestep)
    else:
      self._actor.observe_first(timestep)

  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    self._num_observations += 1
    if self._num_observations < self._num_seed_steps:
      self._random_actor.observe(action, next_timestep)
    else:
      self._actor.observe(action, next_timestep)

  def update(self, wait=True):
    if self._num_observations < self._batch_size:
      num_steps = 0
    else:
      num_steps = 1
    for _ in range(num_steps):
      # Run learner steps (usually means gradient steps).
      self._learner.step()
    if num_steps > 0:
      # Update the actor weights when learner updates.
      self._actor.update(wait=wait)

  def get_variables(self, names):
    return self._learner.get_variables(names)
