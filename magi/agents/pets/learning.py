from typing import Dict, List, NamedTuple, Optional

from absl import logging
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

from magi.agents.pets import models
from magi.agents.pets import replay as replay_lib


class TraningState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState
    state: models.ModelState


class ModelBasedLearner(core.Learner):
    def __init__(
        self,
        spec: specs.EnvironmentSpec,
        model: models.EnsembleModel,
        replay: replay_lib.ReplayBuffer,
        optimizer,
        batch_size: int = 32,
        num_epochs: int = 100,
        seed: jnp.ndarray = None,
        min_delta=0.01,
        patience=5,
        val_ratio=0,
        logger: loggers.Logger = None,
        counter: counting.Counter = None,
    ):
        self.opt = optimizer
        self._replay = replay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self._val_ratio = val_ratio
        self._model = model
        self._logger = logger or loggers.TerminalLogger("learner", time_delta=0.01)
        self._counter = counter or counting.Counter("learner")
        self._patience = patience
        self._min_delta = min_delta

        init_key, iter_key = jax.random.split(seed)
        self._rng = np.random.default_rng(np.asarray(iter_key))

        def init():
            obs = utils.add_batch_dim(utils.zeros_like(spec.observations))
            act = utils.add_batch_dim(utils.zeros_like(spec.actions))
            params, state = model.init(init_key, obs, act)
            return TraningState(params, self.opt.init(params), state)

        @jax.jit
        def update(state: TraningState, x, a, xnext):
            """Learning rule (stochastic gradient descent)."""
            value, grads = jax.value_and_grad(model.loss)(
                state.params, state.state, x, a, xnext
            )
            updates, new_opt_state = self.opt.update(
                grads, state.opt_state, state.params
            )
            new_params = optax.apply_updates(state.params, updates)
            return TraningState(new_params, new_opt_state, state.state), value

        self._update = update

        self._state = init()

    def _evaluate(self, params: hk.Params, state: models.ModelState, iterator):
        if isinstance(iterator, replay_lib.BootstrapIterator):
            iterator.toggle_bootstrap()

        losses = []
        for batch in iterator:
            batch_loss = self._model.evaluate(
                params, state, batch.obs, batch.act, batch.next_obs
            )
            losses.append(batch_loss)
        if isinstance(iterator, replay_lib.BootstrapIterator):
            iterator.toggle_bootstrap()

        # Mean along batch dimension
        return jnp.stack(losses).mean(axis=0)

    def _train(self, replay: replay_lib.ReplayBuffer):
        transitions = replay.get_all()
        new_state = self._model.update_normalizer(
            self._state.state, transitions.obs, transitions.act, transitions.next_obs
        )
        self._state = self._state._replace(state=new_state)
        logging.info("Normalizer mean %s", self._state.state.normalizer.mean)
        logging.info("Normalizer std %s", self._state.state.normalizer.std)
        train_iterator, val_iterator = replay.get_iterators(
            self.batch_size,
            val_ratio=self._val_ratio,
            train_ensemble=True,
            ensemble_size=self._model.num_ensembles,
            shuffle_each_epoch=True,
            bootstrap_permutes=False,
        )
        if val_iterator is None:
            val_iterator = train_iterator

        training_losses, val_scores = [], []
        epochs_since_update = 0
        best_params: Optional[hk.Params] = None
        best_val_score = self._evaluate(
            self._state.params, self._state.state, val_iterator
        )
        for epoch in range(self.num_epochs):
            batch_losses: List[float] = []
            for batch in train_iterator:
                self._state, loss = self._update(
                    self._state, batch.obs, batch.act, batch.next_obs
                )
                batch_losses.append(loss)
            total_avg_loss = np.mean(np.asarray(batch_losses)).mean()
            training_losses.append(total_avg_loss)

            eval_score = self._evaluate(
                self._state.params, self._state.state, val_iterator
            )
            val_scores.append(eval_score.mean().item())

            maybe_best_params = self._maybe_get_best_params(best_val_score, eval_score)
            if maybe_best_params:
                best_val_score = np.minimum(best_val_score, eval_score)
                best_params = maybe_best_params
                epochs_since_update = 0
            else:
                epochs_since_update += 1

            logging.info(
                "epoch %d, model_loss %.3f, model_val_loss %.3f, model_best_val_score %.3f",
                epoch,
                total_avg_loss.item(),
                eval_score.mean().item(),
                best_val_score.mean().item(),
            )
            if self._patience and epochs_since_update >= self._patience:
                break
        del best_params
        # Saving the best models:
        # NOTE(yl): disabled for now, mbrl-lib has a bug on not setting the best weights
        # https://github.com/facebookresearch/mbrl-lib/issues/98
        # look at the performance difference before enabling.
        # self._maybe_set_best_params(best_params, best_val_score)

    def _maybe_set_best_params(self, best_params, best_val_score):
        # TODO: implement elites
        del best_val_score
        if best_params is not None:
            self._state = self._state._replace(params=best_params)
        # if len(best_val_score) > 1 and hasattr(self.model, "num_elites"):
        #   sorted_indices = np.argsort(best_val_score.tolist())
        #   elite_models = sorted_indices[: self.model.num_elites]
        #   self.model.set_elite(elite_models)

    def _maybe_get_best_params(
        self,
        best_val_score: jnp.ndarray,
        val_score: jnp.ndarray,
    ) -> Optional[Dict]:
        improvement = (best_val_score - val_score) / best_val_score
        improved = (improvement > 0.01).any().item()
        return self._state.params if improved else None

    def step(self):
        """Perform an update step of the learner's parameters."""
        self._train(self._replay)

    def get_variables(self, names: List[str]) -> List[hk.Params]:
        del names
        return [{"params": self._state.params, "state": self._state.state}]
