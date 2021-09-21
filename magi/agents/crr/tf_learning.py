# python3
"""Feed-forward CRR learner implementation."""

import time
from typing import Dict, List, Optional

from acme import core
from acme import types
from acme.tf import losses
from acme.tf import networks
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import reverb
import sonnet as snt  # type: ignore
import tensorflow as tf


class CRRLearner(core.Learner):
    """Recurrent CRR learner.

    This is the learning component of a RCRR agent. It takes a dataset as
    input and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        policy_network: snt.Module,
        critic_network: networks.Module,
        target_policy_network: snt.Module,
        target_critic_network: networks.Module,
        dataset: tf.data.Dataset,
        behavior_network: Optional[snt.Module] = None,
        cwp_network: Optional[snt.Module] = None,
        policy_optimizer: Optional[snt.Optimizer] = None,
        critic_optimizer: Optional[snt.Optimizer] = None,
        discount: float = 0.99,
        target_update_period: int = 100,
        num_action_samples_td_learning: int = 1,
        num_action_samples_policy_weight: int = 4,
        baseline_reduce_function: str = "mean",
        clipping: bool = True,
        policy_improvement_modes: str = "exp",
        ratio_upper_bound: float = 20.0,
        beta: float = 1.0,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        checkpoint: bool = False,
    ):
        """Initializes the learner.

        Args:
          policy_network: the online (optimized) policy.
          critic_network: the online critic.
          target_policy_network: the target policy (which lags behind the online
            policy).
          target_critic_network: the target critic.
          dataset: dataset to learn from, whether fixed or from a replay buffer
            (see `acme.datasets.reverb.make_dataset` documentation).
          accelerator_strategy: the strategy used to distribute computation,
            whether on a single, or multiple, GPU or TPU; as supported by
            tf.distribute.
          behavior_network: The network to snapshot under `policy` name. If None,
            snapshots `policy_network` instead.
          cwp_network: CWP network to snapshot: samples actions
            from the policy and weighs them with the critic, then returns the action
            by sampling from the softmax distribution using critic values as logits.
            Used only for snapshotting, not training.
          policy_optimizer: the optimizer to be applied to the policy loss.
          critic_optimizer: the optimizer to be applied to the distributional
            Bellman loss.
          discount: discount to use for TD updates.
          target_update_period: number of learner steps to perform before updating
            the target networks.
          num_action_samples_td_learning: number of action samples to use to
            estimate expected value of the critic loss w.r.t. stochastic policy.
          num_action_samples_policy_weight: number of action samples to use to
            estimate the advantage function for the CRR weighting of the policy
            loss.
          baseline_reduce_function: one of 'mean', 'max', 'min'. Way of aggregating
            values from `num_action_samples` estimates of the value function.
          clipping: whether to clip gradients by global norm.
          policy_improvement_modes: one of 'exp', 'binary', 'all'. CRR mode which
            determines how the advantage function is processed before being
            multiplied by the policy loss.
          ratio_upper_bound: if policy_improvement_modes is 'exp', determines
            the upper bound of the weight (i.e. the weight is
              min(exp(advantage / beta), upper_bound)
            ).
          beta: if policy_improvement_modes is 'exp', determines the beta (see
            above).
          counter: counter object used to keep track of steps.
          logger: logger object to be used by learner.
          checkpoint: boolean indicating whether to checkpoint the learner.
        """

        self._policy_improvement_modes = policy_improvement_modes
        self._ratio_upper_bound = ratio_upper_bound
        self._num_action_samples_td_learning = num_action_samples_td_learning
        self._num_action_samples_policy_weight = num_action_samples_policy_weight
        self._baseline_reduce_function = baseline_reduce_function
        self._beta = beta

        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.TerminalLogger("learner", time_delta=1.0)
        self._discount = discount
        self._clipping = clipping

        self._target_update_period = target_update_period

        # Necessary to track when to update target networks.
        self._num_steps = tf.Variable(0, dtype=tf.int32)
        self._iterator = iter(dataset)

        # Create the optimizers.
        self._critic_optimizer = critic_optimizer or snt.optimizers.Adam(1e-4)
        self._policy_optimizer = policy_optimizer or snt.optimizers.Adam(1e-4)

        # Store online and target networks.
        self._policy_network = policy_network
        self._critic_network = critic_network
        self._target_policy_network = target_policy_network
        self._target_critic_network = target_critic_network

        # Expose the variables.
        self._variables = {
            "critic": self._target_critic_network.variables,
            "policy": self._target_policy_network.variables,
        }

        # Create a checkpointer object.
        self._checkpointer = None
        self._snapshotter = None
        if checkpoint:
            self._checkpointer = tf2_savers.Checkpointer(
                objects_to_save={
                    "counter": self._counter,
                    "policy": self._policy_network,
                    "critic": self._critic_network,
                    "target_policy": self._target_policy_network,
                    "target_critic": self._target_critic_network,
                    "policy_optimizer": self._policy_optimizer,
                    "critic_optimizer": self._critic_optimizer,
                    "num_steps": self._num_steps,
                },
                time_delta_minutes=30.0,
            )

            raw_policy = snt.DeepRNN(
                [policy_network, networks.StochasticSamplingHead()]
            )
            critic_mean = networks.CriticDeepRNN(
                [critic_network, networks.StochasticMeanHead()]
            )
            objects_to_save = {
                "raw_policy": raw_policy,
                "critic": critic_mean,
            }
            if behavior_network is not None:
                objects_to_save["policy"] = behavior_network
            if cwp_network is not None:
                objects_to_save["cwp_policy"] = cwp_network
            self._snapshotter = tf2_savers.Snapshotter(
                objects_to_save=objects_to_save, time_delta_minutes=30
            )
        # Timestamp to keep track of the wall time.
        self._walltime_timestamp = time.time()

    def _step(self, sample: reverb.ReplaySample) -> Dict[str, tf.Tensor]:
        # Transpose batch and sequence axes, i.e. [B, T, ...] to [T, B, ...].
        sample: types.Transition = sample.data
        observations = sample.observation
        actions = sample.action
        rewards = sample.reward
        discounts = sample.discount
        next_observation = sample.next_observation

        dtype = rewards.dtype

        # Cast the additional discount to match the environment discount dtype.
        discount = tf.cast(self._discount, discounts.dtype)

        # Loss cumulants across time. These cannot be python mutable objects.
        critic_loss = 0.0
        policy_loss = 0.0

        # Each transition induces a policy loss, which we then weight using
        # the `policy_loss_coef_t`; shape [B], see https://arxiv.org/abs/2006.15134.
        # `policy_loss_coef` is a scalar average of these coefficients across
        # the batch and sequence length dimensions.
        policy_loss_coef = 0.0
        # per_device_batch_size = actions.shape[1]

        with tf.GradientTape(persistent=True) as tape:
            o_tm1 = observations
            a_tm1 = actions
            r_t = rewards
            d_t = discounts
            o_t = next_observation

            # ========================= Critic learning ============================
            q_tm1 = self._critic_network(o_tm1, a_tm1)
            target_action_distribution = self._target_policy_network(o_t)

            sampled_actions_t = target_action_distribution.sample(
                self._num_action_samples_td_learning
            )
            # [N, B, ...]
            tiled_o_t = tf2_utils.tile_nested(o_t, self._num_action_samples_td_learning)

            # Compute the target critic's Q-value of the sampled actions.
            sampled_q_t = snt.BatchApply(self._target_critic_network)(
                tiled_o_t, sampled_actions_t
            )

            # Compute average logits by first reshaping them to [N, B, A] and then
            # normalizing them across atoms.
            new_shape = [self._num_action_samples_td_learning, r_t.shape[0], -1]
            sampled_logits = tf.reshape(sampled_q_t.logits, new_shape)
            sampled_logprobs = tf.math.log_softmax(sampled_logits, axis=-1)
            averaged_logits = tf.reduce_logsumexp(sampled_logprobs, axis=0)

            # Construct the expected distributional value for bootstrapping.
            q_t = networks.DiscreteValuedDistribution(
                values=sampled_q_t.values, logits=averaged_logits
            )
            critic_loss_t = losses.categorical(q_tm1, r_t, discount * d_t, q_t)
            critic_loss_t = tf.reduce_mean(critic_loss_t)

            # ========================= Actor learning =============================
            action_distribution_tm1 = self._policy_network(o_tm1)
            q_tm1_mean = q_tm1.mean()

            # Compute the estimate of the value function based on
            # self._num_action_samples_policy_weight samples from the policy.
            tiled_o_tm1 = tf2_utils.tile_nested(
                o_tm1, self._num_action_samples_policy_weight
            )
            action_tm1 = action_distribution_tm1.sample(
                self._num_action_samples_policy_weight
            )
            tiled_z_tm1 = snt.BatchApply(self._critic_network)(tiled_o_tm1, action_tm1)
            tiled_v_tm1 = tf.reshape(
                tiled_z_tm1.mean(), [self._num_action_samples_policy_weight, -1]
            )

            # Use mean, min, or max to aggregate Q(s, a_i), a_i ~ pi(s) into the
            # final estimate of the value function.
            if self._baseline_reduce_function == "mean":
                v_tm1_estimate = tf.reduce_mean(tiled_v_tm1, axis=0)
            elif self._baseline_reduce_function == "max":
                v_tm1_estimate = tf.reduce_max(tiled_v_tm1, axis=0)
            elif self._baseline_reduce_function == "min":
                v_tm1_estimate = tf.reduce_min(tiled_v_tm1, axis=0)

            # Assert that action_distribution_tm1 is a batch of multivariate
            # distributions (in contrast to e.g. a [batch, action_size] collection
            # of 1d distributions).
            assert len(action_distribution_tm1.batch_shape) == 1
            policy_loss_batch = -action_distribution_tm1.log_prob(a_tm1)

            advantage = q_tm1_mean - v_tm1_estimate
            if self._policy_improvement_modes == "exp":
                policy_loss_coef_t = tf.math.minimum(
                    tf.math.exp(advantage / self._beta), self._ratio_upper_bound
                )
            elif self._policy_improvement_modes == "binary":
                policy_loss_coef_t = tf.cast(advantage > 0, dtype)
            elif self._policy_improvement_modes == "all":
                # Regress against all actions (effectively pure BC).
                policy_loss_coef_t = 1.0
            policy_loss_coef_t = tf.stop_gradient(policy_loss_coef_t)

            policy_loss_batch *= policy_loss_coef_t
            policy_loss_t = tf.reduce_mean(policy_loss_batch)

            critic_loss += critic_loss_t
            policy_loss += policy_loss_t
            policy_loss_coef += tf.reduce_mean(policy_loss_coef_t)  # For logging.

        # Compute gradients.
        critic_gradients = tape.gradient(
            critic_loss, self._critic_network.trainable_variables
        )
        policy_gradients = tape.gradient(
            policy_loss, self._policy_network.trainable_variables
        )

        # Delete the tape manually because of the persistent=True flag.
        del tape

        # Sync gradients across GPUs or TPUs.
        # ctx = tf.distribute.get_replica_context()
        # critic_gradients = ctx.all_reduce("mean", critic_gradients)
        # policy_gradients = ctx.all_reduce("mean", policy_gradients)

        # Maybe clip gradients.
        if self._clipping:
            policy_gradients, policy_g_norm = tf.clip_by_global_norm(
                policy_gradients, 40.0
            )
            critic_gradients, critic_g_norm = tf.clip_by_global_norm(
                critic_gradients, 40.0
            )

        # Apply gradients.
        self._critic_optimizer.apply(
            critic_gradients, self._critic_network.trainable_variables
        )
        self._policy_optimizer.apply(
            policy_gradients, self._policy_network.trainable_variables
        )

        source_variables = (
            self._critic_network.variables + self._policy_network.variables
        )
        target_variables = (
            self._target_critic_network.variables
            + self._target_policy_network.variables
        )

        # Make online -> target network update ops.
        if tf.math.mod(self._num_steps, self._target_update_period) == 0:
            for src, dest in zip(source_variables, target_variables):
                dest.assign(src)
        self._num_steps.assign_add(1)

        return {
            "critic_loss": critic_loss,
            "policy_loss": policy_loss,
            "policy_loss_coef": policy_loss_coef,
            "advantages": tf.reduce_mean(advantage),
            "policy_g_norm": policy_g_norm,
            "critic_g_norm": critic_g_norm,
            "q": tf.reduce_mean(q_tm1_mean),
        }

    @tf.function
    def _replicated_step(self) -> Dict[str, tf.Tensor]:
        sample = next(self._iterator)
        # fetches = self._accelerator_strategy.run(self._step, args=(sample,))
        fetches = self._step(sample)
        # mean = tf.distribute.ReduceOp.MEAN
        return fetches

    def step(self):
        # Run the learning step.
        fetches = self._replicated_step()
        # Update our counts and record it.
        new_timestamp = time.time()
        time_passed = new_timestamp - self._walltime_timestamp
        self._walltime_timestamp = new_timestamp
        counts = self._counter.increment(steps=1, wall_time=time_passed)
        fetches.update(counts)

        # Checkpoint and attempt to write the logs.
        if self._checkpointer is not None:
            self._checkpointer.save()
            self._snapshotter.save()
        self._logger.write(fetches)

    def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
        return [tf2_utils.to_numpy(self._variables[name]) for name in names]
