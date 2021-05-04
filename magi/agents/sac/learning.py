from functools import partial
from typing import Iterator, Optional

from acme import core
from acme import specs
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import reverb

from magi.agents.sac import losses


def heuristic_target_entropy(action_spec):
  """Compute the heuristic target entropy"""
  return -float(np.prod(action_spec.shape))


class SACLearner(core.Learner, core.VariableSource):

  def __init__(self,
               environment_spec: specs.EnvironmentSpec,
               policy: hk.Transformed,
               critic: hk.Transformed,
               key: jax.random.PRNGKey,
               dataset: Iterator[reverb.ReplaySample],
               actor_optimizer: optax.GradientTransformation,
               critic_optimizer: optax.GradientTransformation,
               alpha_optimizer: optax.GradientTransformation,
               gamma: float = 0.99,
               tau: float = 5e-3,
               init_alpha: float = 1.0,
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
    opt_init, self.opt_critic = critic_optimizer
    self.opt_state_critic = opt_init(self.params_critic)
    # Actor.
    self.actor = policy
    self.params_actor = self.actor.init(next(self._rng), dummy_state)
    opt_init, self.opt_actor = actor_optimizer
    self.opt_state_actor = opt_init(self.params_actor)
    # Entropy coefficient.
    self.target_entropy = heuristic_target_entropy(environment_spec.actions)
    self.log_alpha = jnp.array(np.log(init_alpha), dtype=jnp.float32)
    opt_init, self.opt_alpha = alpha_optimizer
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
