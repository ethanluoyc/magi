"""Soft Actor-Critic implementation"""
from functools import partial
from typing import Iterator, Optional, Sequence

from acme import core
from acme import datasets
from acme import specs
from acme.adders import reverb as adders
from acme.jax import networks as network_lib
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
from magi.agents.sac import acting as acting_lib
from magi.agents.sac import losses


def heuristic_target_entropy(action_spec):
  """Compute the heuristic target entropy"""
  return -float(np.prod(action_spec.shape))


def make_apply_sample_fn(forward_fn, is_eval=True):

  def fn(params, key, observation):
    dist = forward_fn(params, observation)
    if is_eval:
      return dist.mode()
    else:
      return dist.sample(key)

  return fn


class SACLearner(core.Learner, core.VariableSource):

  def __init__(self,
               environment_spec: specs.EnvironmentSpec,
               policy: hk.Transformed,
               critic: hk.Transformed,
               key: jax.random.PRNGKey,
               dataset: Iterator[reverb.ReplaySample],
               gamma: float = 0.99,
               tau: float = 5e-3,
               lr_actor: float = 3e-4,
               lr_critic: float = 3e-4,
               lr_alpha: float = 3e-4,
               init_alpha: float = 1.0,
               adam_b1_alpha: float = 0.9,
               max_gradient_norm: float = 0.5,
               logger: Optional[loggers.Logger] = None,
               counter: Optional[counting.Counter] = None):
    self._rng = hk.PRNGSequence(key)
    self._iterator = dataset
    self._gamma = gamma

    self._logger = logger if logger is not None else loggers.make_default_logger(
        label='learner', save_data=False)
    self._counter = counter if counter is not None else counting.Counter()

    # Define fake input for critic.
    dummy_state = utils.add_batch_dim(utils.zeros_like(environment_spec.observations))
    dummy_action = utils.add_batch_dim(utils.zeros_like(environment_spec.actions))

    # Critic.
    self.critic = critic
    self.params_critic = self.params_critic_target = self.critic.init(
        next(self._rng), dummy_state, dummy_action)
    opt_init, self.opt_critic = optax.chain(
        optax.clip_by_global_norm(max_gradient_norm), optax.adam(lr_critic))
    self.opt_state_critic = opt_init(self.params_critic)
    # Actor.
    self.actor = policy
    self.params_actor = self.actor.init(next(self._rng), dummy_state)
    opt_init, self.opt_actor = optax.chain(optax.clip_by_global_norm(max_gradient_norm),
                                           optax.adam(lr_actor))
    self.opt_state_actor = opt_init(self.params_actor)
    # Entropy coefficient.
    self.target_entropy = heuristic_target_entropy(environment_spec.actions)
    self.log_alpha = jnp.array(np.log(init_alpha), dtype=jnp.float32)
    opt_init, self.opt_alpha = optax.adam(lr_alpha, b1=adam_b1_alpha)
    self.opt_state_alpha = opt_init(self.log_alpha)

    @jax.jit
    def _update_actor(params_actor, opt_state, key, params_critic, log_alpha,
                      observation):

      def loss_fn(actor_params):
        return losses.actor_loss_fn(self.actor, self.critic, actor_params, key,
                                    params_critic, log_alpha, observation)

      (loss, aux), grad = jax.value_and_grad(loss_fn, has_aux=True)(params_actor)
      update, opt_state = self.opt_actor(grad, opt_state)
      params_actor = optax.apply_updates(params_actor, update)
      return params_actor, opt_state, loss, aux

    @jax.jit
    def _update_critic(params_critic, opt_state, key, critic_target_params,
                       actor_params, log_alpha, batch):

      def loss_fn(critic_params):
        return losses.critic_loss_fn(self.actor,
                                     self.critic,
                                     critic_params,
                                     key,
                                     critic_target_params,
                                     actor_params,
                                     log_alpha,
                                     batch,
                                     gamma=self._gamma)

      (loss, aux), grad = jax.value_and_grad(loss_fn, has_aux=True)(params_critic)
      update, opt_state = self.opt_critic(grad, opt_state)
      params_critic = optax.apply_updates(params_critic, update)
      return params_critic, opt_state, loss, aux

    @jax.jit
    def _update_alpha(log_alpha, opt_state, entropy):

      def loss_fn(log_alpha):
        return losses.alpha_loss_fn(log_alpha,
                                    entropy,
                                    target_entropy=self.target_entropy)

      (loss, aux), grad = jax.value_and_grad(loss_fn, has_aux=True)(log_alpha)
      update, opt_state = self.opt_alpha(grad, opt_state)
      log_alpha = optax.apply_updates(log_alpha, update)
      return log_alpha, opt_state, loss, aux

    self._update_actor = _update_actor
    self._update_critic = _update_critic
    self._update_alpha = _update_alpha
    self._update_target = jax.jit(partial(optax.incremental_update, step_size=tau))

  def step(self):
    batch = next(self._iterator)

    # Update critic.
    self.params_critic, self.opt_state_critic, loss_critic, _ = self._update_critic(
        self.params_critic,
        self.opt_state_critic,
        key=next(self._rng),
        critic_target_params=self.params_critic_target,
        actor_params=self.params_actor,
        log_alpha=self.log_alpha,
        batch=batch)
    # Update actor
    self.params_actor, self.opt_state_actor, loss_actor, actor_stats = (
        self._update_actor(self.params_actor,
                           self.opt_state_actor,
                           key=next(self._rng),
                           params_critic=self.params_critic,
                           log_alpha=self.log_alpha,
                           observation=batch.data.observation))
    self.log_alpha, self.opt_state_alpha, loss_alpha, _ = self._update_alpha(
        self.log_alpha,
        self.opt_state_alpha,
        entropy=actor_stats['entropy'],
    )
    self._counter.increment(steps=1)
    results = {
        'loss_alpha': loss_alpha,
        'loss_actor': loss_actor,
        'loss_critic': loss_critic,
        'alpha': jnp.exp(self.log_alpha)
    }
    self._logger.write(results)
    # Update target network.
    self.params_critic_target = self._update_target(self.params_critic,
                                                    self.params_critic_target)

  def get_variables(self, names):
    del names
    return [self.params_actor]


class SACAgent(core.Actor):

  def __init__(self,
               environment_spec,
               policy,
               critic,
               seed,
               gamma=0.99,
               buffer_size=10**6,
               batch_size=256,
               start_steps=10000,
               tau=5e-3,
               lr_actor=3e-4,
               lr_critic=3e-4,
               lr_alpha=3e-4,
               init_alpha=1.0,
               adam_b1_alpha=0.9,
               logger=None,
               counter=None):
    # self.rng = hk.PRNGSequence(seed)
    learner_key, actor_key, actor_key2, random_key = jax.random.split(
        jax.random.PRNGKey(seed), 4)
    self._num_observations = 0
    self._start_steps = start_steps

    replay_table = reverb.Table(name=adders.DEFAULT_PRIORITY_TABLE,
                                sampler=reverb.selectors.Uniform(),
                                remover=reverb.selectors.Fifo(),
                                max_size=buffer_size,
                                rate_limiter=rate_limiters.MinSize(1),
                                signature=adders.NStepTransitionAdder.signature(
                                    environment_spec=environment_spec))
    self._server = reverb.Server([replay_table], port=None)

    # The adder is used to insert observations into replay.
    # discount is 1.0 as we are multiplying gamma during learner step
    address = f'localhost:{self._server.port}'
    # The dataset provides an interface to sample from replay.
    dataset = datasets.make_reverb_dataset(server_address=address,
                                           environment_spec=environment_spec,
                                           batch_size=batch_size,
                                           prefetch_size=1,
                                           transition_adder=True)
    self._learner = SACLearner(environment_spec,
                               policy,
                               critic,
                               key=learner_key,
                               dataset=dataset.as_numpy_iterator(),
                               gamma=gamma,
                               tau=tau,
                               lr_actor=lr_actor,
                               lr_critic=lr_critic,
                               lr_alpha=lr_alpha,
                               init_alpha=init_alpha,
                               adam_b1_alpha=adam_b1_alpha,
                               logger=logger,
                               counter=counter)

    adder = adders.NStepTransitionAdder(client=reverb.Client(address),
                                        n_step=1,
                                        discount=1.0)

    client = variable_utils.VariableClient(self._learner, '')
    self._actor = acting_lib.SACActor(policy.apply,
                                      actor_key,
                                      is_eval=False,
                                      variable_client=client,
                                      adder=adder)
    self._eval_actor = acting_lib.SACActor(policy.apply,
                                           actor_key2,
                                           is_eval=False,
                                           variable_client=client)
    self._random_actor = actors.RandomActor(environment_spec.actions,
                                            random_key,
                                            adder=adder)

  def select_action(self, observation: network_lib.Observation) -> network_lib.Action:
    return self._get_active_actor().select_action(observation)

  def observe_first(self, timestep: dm_env.TimeStep):
    return self._get_active_actor().observe_first(timestep)

  def observe(self, action: network_lib.Action, next_timestep: dm_env.TimeStep):
    self._num_observations += 1
    self._get_active_actor().observe(action, next_timestep)

  def _get_active_actor(self):
    if self._num_observations < self._start_steps:
      return self._random_actor
    else:
      return self._actor

  def update(self):
    if self._num_observations < self._start_steps:
      return
    self._learner.step()
    self._actor.update(wait=True)

  def get_variables(self, names: Sequence[str]):
    return [self._learner.get_variables(names)]
