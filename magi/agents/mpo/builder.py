"""MPO builder"""
from typing import Callable, Iterator, List, Optional

from acme import adders
import numpy as np
from acme import core
from acme import datasets
from acme import specs
from acme.adders import reverb as adders_reverb
from acme.agents.jax import actor_core
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.jax import losses
from acme.jax import networks as networks_lib
from acme.jax import variable_utils
from acme.jax.types import Networks
from acme.jax.types import PolicyNetwork
from acme.utils import counting
from acme.utils import loggers
import optax
import reverb
from reverb import rate_limiters

from magi.agents.mpo import config as mpo_config
from magi.agents.mpo import learning as learning_lib
from magi.agents.mpo import retrace_learning
import jax.numpy as jnp


def make_inference_fn(networks, action_spec, eval_mode: bool):

  def policy_network(params, key, obs):
    action_dist = networks['policy'].apply(params, obs)
    action = action_dist.mode() if eval_mode else action_dist.sample(seed=key)
    log_prob = action_dist.log_prob(action)
    action = jnp.clip(action, action_spec.minimum, action_spec.maximum)
    extra = {'log_prob': log_prob}
    return action, extra

  return policy_network


class MPOBuilder(builders.ActorLearnerBuilder):
  """MPO agent builder"""

  def __init__(
      self,
      config: mpo_config.MPOConfig,
      policy_loss_fn: Optional[losses.MPO] = None,
      logger_fn: Callable[[], loggers.Logger] = lambda: None,
  ):
    """Create a builder for assembling MPO agents.

    Args:
      config: configuration for MPO agent.
      policy_loss_fn: policy loss for MPO, if None, then a default
          will be chosen similar to the Acme TF MPO agent.
          See `MPOLearner` for defaults.
      logger_fn: a callable that returns the logger used by the MPO
          learner component.
    """
    self._config = config
    self._logger_fn = logger_fn
    self._policy_loss_fn = policy_loss_fn

  def make_replay_tables(
      self,
      environment_spec: specs.EnvironmentSpec,
  ) -> List[reverb.Table]:
    if not self._config.retrace:
      replay_table = reverb.Table(
          name=self._config.replay_table_name,
          sampler=reverb.selectors.Uniform(),
          remover=reverb.selectors.Fifo(),
          max_size=self._config.max_replay_size,
          rate_limiter=rate_limiters.MinSize(self._config.min_replay_size),
          signature=adders_reverb.NStepTransitionAdder.signature(
              environment_spec=environment_spec),
      )
    else:
      replay_table = reverb.Table(
          name=self._config.replay_table_name,
          sampler=reverb.selectors.Uniform(),
          remover=reverb.selectors.Fifo(),
          max_size=self._config.max_replay_size,
          rate_limiter=rate_limiters.MinSize(self._config.min_replay_size),
          signature=adders_reverb.SequenceAdder.signature(
              extras_spec={'log_prob': np.zeros((), dtype=np.float32)},
              sequence_length=self._config.retrace_sequence_length + 1,
              environment_spec=environment_spec),
      )

    # Cache the environment spec here, this is needed as the
    # MPO learner needs to know the action_spec to initialize the dual variables
    self._environment_spec = environment_spec
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
    )
    return dataset.as_numpy_iterator()

  def make_adder(self, replay_client: reverb.Client) -> Optional[adders.Adder]:
    if not self._config.retrace:
      return adders_reverb.NStepTransitionAdder(
          client=replay_client,
          n_step=self._config.n_step,
          discount=self._config.discount,
      )
    else:
      return adders_reverb.SequenceAdder(
          client=replay_client,
          sequence_length=self._config.retrace_sequence_length + 1,
          period=self._config.retrace_sequence_length,
          end_of_episode_behavior=adders_reverb.EndBehavior.ZERO_PAD,
      )

  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy_network: PolicyNetwork,
      adder: Optional[adders.Adder] = None,
      variable_source: Optional[core.VariableSource] = None,
  ) -> core.Actor:
    assert variable_source is not None
    variable_client = variable_utils.VariableClient(
        variable_source, 'policy', device='cpu')
    variable_client.update_and_wait()

    return actors.GenericActor(
        actor_core.batched_feed_forward_with_extras_to_actor_core(
            policy_network),
        random_key,
        variable_client=variable_client,
        adder=adder,
    )

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: Networks,
      dataset: Iterator[reverb.ReplaySample],
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    del replay_client
    policy_optimizer = optax.adam(self._config.policy_learning_rate)
    critic_optimizer = optax.adam(self._config.critic_learning_rate)

    if self._config.clipping:
      policy_optimizer = optax.chain(
          optax.clip_by_global_norm(40.0), policy_optimizer)
      critic_optimizer = optax.chain(
          optax.clip_by_global_norm(40.0), critic_optimizer)

    dual_optimizer = optax.adam(self._config.dual_learning_rate)
    if self._config.retrace:
      return retrace_learning.MPORetraceLearner(
          policy_network=networks['policy'],
          critic_network=networks['critic'],
          random_key=random_key,
          dataset=dataset,
          policy_optimizer=policy_optimizer,
          critic_optimizer=critic_optimizer,
          dual_optimizer=dual_optimizer,
          discount=self._config.discount,
          num_samples=self._config.num_samples,
          action_dim=self._environment_spec.actions.shape[0],
          target_policy_update_period=self._config.target_policy_update_period,
          target_critic_update_period=self._config.target_critic_update_period,
          policy_loss_fn=self._policy_loss_fn,
          logger=self._logger_fn(),
          counter=counter,
      )
    else:
      return learning_lib.MPOLearner(
          policy_network=networks['policy'],
          critic_network=networks['critic'],
          random_key=random_key,
          dataset=dataset,
          policy_optimizer=policy_optimizer,
          critic_optimizer=critic_optimizer,
          dual_optimizer=dual_optimizer,
          discount=self._config.discount,
          num_samples=self._config.num_samples,
          action_dim=self._environment_spec.actions.shape[0],
          target_policy_update_period=self._config.target_policy_update_period,
          target_critic_update_period=self._config.target_critic_update_period,
          policy_loss_fn=self._policy_loss_fn,
          logger=self._logger_fn(),
          counter=counter,
      )
