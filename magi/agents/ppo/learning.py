"""PPO learner."""
from functools import partial
import time
from typing import List, NamedTuple

import acme
from acme.utils import counting
from acme.utils import loggers
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
import tree


class TrainingState(NamedTuple):
    """Training state consists of network parameters and optimiser state."""

    params: hk.Params
    opt_state: optax.OptState


def _select_batch(exps, indices):
    return tree.map_structure(lambda x: x[indices], exps)


class PPOLearner(acme.Learner, acme.Saveable):
    def __init__(
        self,
        obs_spec,
        network_fn,
        optimizer,
        iterator,
        random_key,
        discount: float = 0.99,
        lambd: float = 0.95,
        clip_eps: float = 0.2,
        entropy_cost: float = 0.01,
        value_cost: float = 0.5,
        ppo_epochs=4,
        normalize_rew=False,
        minibatch_size=256,
        counter=None,
        logger=None,
    ):
        # Critic.
        self.discount = discount
        self.lambd = lambd
        self.clip_eps = clip_eps
        self.entropy_cost = entropy_cost
        self.value_cost = value_cost
        self.rng = hk.PRNGSequence(random_key)
        self.ppo_epochs = ppo_epochs
        self.normalize_rew = normalize_rew
        self._minibatch_size = minibatch_size
        dummy_inputs = jnp.zeros((1, *obs_spec.shape), dtype=jnp.float32)

        # Actor.
        network = hk.without_apply_rng(hk.transform(network_fn))
        params = network.init(next(self.rng), dummy_inputs)
        opt_state = optimizer.init(params)
        self.opt = optimizer

        self._state = TrainingState(params=params, opt_state=opt_state)

        self._logger = logger or loggers.make_default_logger("learner")
        self._counter = counter or counting.Counter()
        self._forward = jax.jit(network.apply)
        self._iterator = iterator

    @partial(jax.jit, static_argnums=0)
    def _prepare_inputs(self, batch):
        # start = time.time()
        # Prepare inputs for mini-batch training
        observations = batch.observation
        action = batch.action[:, :-1]
        logits = batch.extras["logits"][:, :-1]
        reward = batch.reward[:, :-1]
        discount = self.discount * batch.discount[:, :-1]
        values = batch.extras["value"]

        reward = (reward - reward.mean()) / (reward.std() + 1e-8)
        log_pi_old = jax.vmap(rlax.softmax().logprob, in_axes=(0, 0), out_axes=0)(
            action, logits
        )
        advantage = jax.vmap(
            rlax.truncated_generalized_advantage_estimation,
            in_axes=(0, 0, None, 0),
            out_axes=0,
        )(reward, discount, self.lambd, values)
        batch = {
            "obs": observations[:, :-1],
            "action": action,
            "reward": reward,
            "advantage": advantage,
            "value": values[:, :-1],
            "log_prob": log_pi_old,
            # 'return': advantage + values[:, :-1],
        }
        batch["return"] = batch["value"] + batch["advantage"]
        return tree.map_structure(lambda x: x.reshape((-1,) + x.shape[2:]), batch)

    def step(self):
        sample = next(self._iterator)
        exps = self._prepare_inputs(sample.data)
        num_frames = tree.flatten(exps)[0].shape[0]

        for _ in range(self.ppo_epochs):
            indices = np.arange(0, num_frames)
            indices = np.random.permutation(indices)
            # Minibatch training
            for start in range(0, num_frames, self._minibatch_size):
                end = start + self._minibatch_size
                batch_ind = indices[start:end]
                batch = _select_batch(exps, batch_ind)
                self._state, results = self._sgd_step(self._state, batch)
                counts = self._counter.increment(
                    steps=1, time_elapsed=time.time() - start
                )
                self._logger.write({**results, **counts})

    @partial(jax.jit, static_argnums=0)
    def _sgd_step(self, training_state: TrainingState, batch):
        params = training_state.params
        opt_state = training_state.opt_state

        (loss, stats), grads = jax.value_and_grad(self._loss, has_aux=True)(
            params, batch
        )
        stats["loss"] = loss
        stats["grad_norm"] = optax.global_norm(grads)
        updates, new_opt_state = self.opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        new_state = TrainingState(params=new_params, opt_state=new_opt_state)
        return new_state, stats

    def _loss(self, params: hk.Params, batch) -> jnp.ndarray:
        obs = batch["obs"]
        action = batch["action"]
        log_prob = batch["log_prob"]

        logits, value = self._forward(params, obs)
        entropy = rlax.softmax().entropy(logits).mean()

        ratio = jnp.exp(rlax.softmax().logprob(action, logits) - log_prob)
        surr1 = ratio * batch["advantage"]
        surr2 = (
            jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
            * batch["advantage"]
        )
        policy_loss = -jnp.minimum(surr1, surr2).mean()

        value_clipped = batch["value"] + jnp.clip(
            value - batch["value"], -self.clip_eps, self.clip_eps
        )
        surr1 = jnp.square(value - batch["return"])
        surr2 = jnp.square((value_clipped - batch["return"]))
        value_loss = jnp.maximum(surr1, surr2).mean()

        loss = policy_loss - self.entropy_cost * entropy + self.value_cost * value_loss
        return loss, {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "value": value.mean(),
        }

    def get_variables(self, names: List[str]) -> List[hk.Params]:
        return [self._state.params]

    def save(self) -> TrainingState:
        return self._state

    def restore(self, state: TrainingState):
        self._state = state
