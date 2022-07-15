"""Implements the MPO loss.

The MPO loss uses MPOParams, which can be initialized using init_params,
to track the temperature and the dual variables.

Tensor shapes are annotated, where helpful, as follow:
  B: batch size,
  N: number of sampled actions, see MPO paper for more details,
  D: dimensionality of the action space.
"""

from typing import NamedTuple, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import tensorflow_probability

tfp = tensorflow_probability.substrates.jax
tfd = tensorflow_probability.substrates.jax.distributions

_MPO_FLOAT_EPSILON = 1e-8
_MIN_LOG_TEMPERATURE = -18.0
_MIN_LOG_ALPHA = -18.0

Shape = Tuple[int]
DType = type(jnp.float32)  # _ScalarMeta, a private type.


class MPOParams(NamedTuple):
  """NamedTuple to store trainable loss parameters."""
  log_temperature: jnp.ndarray
  log_alpha_mean: jnp.ndarray
  log_alpha_stddev: jnp.ndarray
  log_alpha_categorical: jnp.ndarray
  log_penalty_temperature: Optional[jnp.ndarray] = None


class MPOStats(NamedTuple):
  """NamedTuple to store loss statistics."""
  dual_alpha_mean: float
  dual_alpha_stddev: float
  dual_temperature: float

  loss_policy: float
  loss_alpha: float
  loss_temperature: float
  kl_q_rel: float

  kl_mean_rel: float
  kl_stddev_rel: float

  q_min: float
  q_max: float

  pi_stddev_min: float
  pi_stddev_max: float
  pi_stddev_cond: float

  penalty_kl_q_rel: Optional[float] = None


class RHPO:
  """RHPO loss with decoupled KL constraints as in (Abdolmaleki et al., 2018)."""

  def __init__(self,
               epsilon: float,
               epsilon_mean: float,
               epsilon_stddev: float,
               epsilon_categorical: float,
               init_log_temperature: float,
               init_log_alpha_mean: float,
               init_log_alpha_stddev: float,
               init_log_alpha_categorical: float,
               per_dim_constraining: bool = True,
               action_penalization: bool = True,
               epsilon_penalty: float = 0.001):
    """Initialize and configure the MPO loss.

    Args:
      epsilon: KL constraint on the non-parametric auxiliary policy, the one
        associated with the dual variable called temperature.
      epsilon_mean: KL constraint on the mean of the Gaussian policy, the one
        associated with the dual variable called alpha_mean.
      epsilon_stddev: KL constraint on the stddev of the Gaussian policy, the
        one associated with the dual variable called alpha_mean.
      init_log_temperature: initial value for the temperature in log-space, note
        a softplus (rather than an exp) will be used to transform this.
      init_log_alpha_mean: initial value for the alpha_mean in log-space, note a
        softplus (rather than an exp) will be used to transform this.
      init_log_alpha_stddev: initial value for the alpha_stddev in log-space,
        note a softplus (rather than an exp) will be used to transform this.
      per_dim_constraining: whether to enforce the KL constraint on each
        dimension independently; this is the default. Otherwise the overall KL
        is constrained, which allows some dimensions to change more at the
        expense of others staying put.
      action_penalization: whether to use a KL constraint to penalize actions
        via the MO-MPO algorithm.
      epsilon_penalty: KL constraint on the probability of violating the action
        constraint.
    """

    # MPO constrain thresholds.
    self._epsilon = epsilon
    self._epsilon_mean = epsilon_mean
    self._epsilon_stddev = epsilon_stddev
    self._epsilon_categorical = epsilon_categorical

    # Initial values for the constraints' dual variables.
    self._init_log_temperature = init_log_temperature
    self._init_log_alpha_mean = init_log_alpha_mean
    self._init_log_alpha_stddev = init_log_alpha_stddev
    self._init_log_alpha_categorical = init_log_alpha_categorical

    # Whether to penalize out-of-bound actions via MO-MPO and its corresponding
    # constraint threshold.
    self._action_penalization = action_penalization
    self._epsilon_penalty = epsilon_penalty

    # Whether to ensure per-dimension KL constraint satisfication.
    self._per_dim_constraining = per_dim_constraining

  @property
  def per_dim_constraining(self):
    return self._per_dim_constraining

  def init_params(self,
                  num_components: int,
                  action_dim: int,
                  dtype: DType = jnp.float32):
    """Creates an initial set of parameters."""

    if self._per_dim_constraining:
      dual_variable_shape = [num_components, action_dim]
    else:
      dual_variable_shape = [num_components, 1]

    log_temperature = jnp.full([1], self._init_log_temperature, dtype=dtype)

    log_alpha_mean = jnp.full(
        dual_variable_shape, self._init_log_alpha_mean, dtype=dtype)

    log_alpha_stddev = jnp.full(
        dual_variable_shape, self._init_log_alpha_stddev, dtype=dtype)

    if self._action_penalization:
      log_penalty_temperature = jnp.full([1],
                                         self._init_log_temperature,
                                         dtype=dtype)
    else:
      log_penalty_temperature = None

    log_alpha_categorical = jnp.full([1],
                                     self._init_log_alpha_categorical,
                                     dtype=dtype)

    return MPOParams(
        log_temperature=log_temperature,
        log_alpha_mean=log_alpha_mean,
        log_alpha_stddev=log_alpha_stddev,
        log_penalty_temperature=log_penalty_temperature,
        log_alpha_categorical=log_alpha_categorical,
    )

  def __call__(
      self,
      params: MPOParams,
      online_action_distribution: Union[tfd.MultivariateNormalDiag,
                                        tfd.Independent],
      target_action_distribution: Union[tfd.MultivariateNormalDiag,
                                        tfd.Independent],
      actions: jnp.ndarray,  # Shape [N, B, D].
      q_values: jnp.ndarray,  # Shape [N, B].
  ) -> Tuple[jnp.ndarray, MPOStats]:
    """Computes the decoupled MPO loss.

    Args:
      params: parameters tracking the temperature and the dual variables.
      online_action_distribution: online distribution returned by the online
        policy network; expects batch_dims of [B] and event_dims of [D].
      target_action_distribution: target distribution returned by the target
        policy network; expects same shapes as online distribution.
      actions: actions sampled from the target policy; expects shape [N, B, D].
      q_values: Q-values associated with each action; expects shape [N, B].

    Returns:
      Loss, combining the policy loss, KL penalty, and dual losses required to
        adapt the dual variables.
      Stats, for diagnostics and tracking performance.
    """
    target_component_distribution = target_action_distribution.components_distribution
    target_categorical_distribution = target_action_distribution.mixture_distribution
    online_component_distribution = online_action_distribution.components_distribution
    online_categorical_distribution = online_action_distribution.mixture_distribution
    del target_action_distribution, online_action_distribution
    num_components = online_component_distribution.batch_shape[-1]
    batch_size = actions.shape[1]
    action_size = actions.shape[-1]

    # Cast `MultivariateNormalDiag`s to Independent Normals.
    # The latter allows us to satisfy KL constraints per-dimension.
    if isinstance(target_component_distribution, tfd.MultivariateNormalDiag):
      target_component_distribution = tfd.Independent(
          tfd.Normal(target_component_distribution.mean(),
                     target_component_distribution.stddev()),
          reinterpreted_batch_ndims=1)
      online_component_distribution = tfd.Independent(
          tfd.Normal(online_component_distribution.mean(),
                     online_component_distribution.stddev()),
          reinterpreted_batch_ndims=1)

    # Transform dual variables from log-space.
    # Note: using softplus instead of exponential for numerical stability.
    temperature = jax.nn.softplus(params.log_temperature) + _MPO_FLOAT_EPSILON
    alpha_mean = jax.nn.softplus(params.log_alpha_mean) + _MPO_FLOAT_EPSILON
    alpha_stddev = jax.nn.softplus(params.log_alpha_stddev) + _MPO_FLOAT_EPSILON
    alpha_categorical = jax.nn.softplus(
        params.log_alpha_categorical) + _MPO_FLOAT_EPSILON

    # Get online and target means and stddevs in preparation for decomposition.
    online_mean = online_component_distribution.distribution.mean()
    online_scale = online_component_distribution.distribution.stddev()
    online_logits = online_categorical_distribution.logits
    target_mean = target_component_distribution.distribution.mean()
    target_scale = target_component_distribution.distribution.stddev()
    target_logits = target_categorical_distribution.logits

    # Compute normalized importance weights, used to compute expectations with
    # respect to the non-parametric policy; and the temperature loss, used to
    # adapt the tempering of Q-values.
    normalized_weights, loss_temperature = compute_weights_and_temperature_loss(
        q_values, self._epsilon, temperature)

    # Only needed for diagnostics: Compute estimated actualized KL between the
    # non-parametric and current target policies.
    kl_nonparametric = compute_nonparametric_kl_from_normalized_weights(
        normalized_weights)

    if self._action_penalization:
      # Transform action penalization temperature.
      penalty_temperature = jax.nn.softplus(
          params.log_penalty_temperature) + _MPO_FLOAT_EPSILON

      # Compute action penalization cost.
      # Note: the cost is zero in [-1, 1] and quadratic beyond.
      diff_out_of_bound = actions - jnp.clip(actions, -1.0, 1.0)
      cost_out_of_bound = -jnp.linalg.norm(diff_out_of_bound, axis=-1)

      penalty_normalized_weights, loss_penalty_temperature = (
          compute_weights_and_temperature_loss(cost_out_of_bound,
                                               self._epsilon_penalty,
                                               penalty_temperature))

      # Only needed for diagnostics: Compute estimated actualized KL between the
      # non-parametric and current target policies.
      penalty_kl_nonparametric = compute_nonparametric_kl_from_normalized_weights(
          penalty_normalized_weights)

      # Combine normalized weights.
      normalized_weights += penalty_normalized_weights
      loss_temperature += loss_penalty_temperature

    # Decompose the online policy into fixed-mean & fixed-stddev distributions.
    # This has been documented as having better performance in bandit settings,
    # see e.g. https://arxiv.org/pdf/1812.02256.pdf.
    policy_mean_distribution = tfd.MixtureSameFamily(
        tfd.Categorical(logits=target_logits),
        tfd.Independent(
            tfd.Normal(loc=online_mean, scale=target_scale),
            reinterpreted_batch_ndims=1),
    )
    policy_stddev_distribution = tfd.MixtureSameFamily(
        tfd.Categorical(logits=target_logits),
        tfd.Independent(
            tfd.Normal(loc=target_mean, scale=online_scale),
            reinterpreted_batch_ndims=1))
    policy_categorical_distribution = tfd.MixtureSameFamily(
        tfd.Categorical(logits=online_logits),
        tfd.Independent(
            tfd.Normal(loc=target_mean, scale=target_scale),
            reinterpreted_batch_ndims=1))
    assert policy_mean_distribution.batch_shape == (batch_size,)
    assert policy_stddev_distribution.batch_shape == (batch_size,)
    assert policy_categorical_distribution.batch_shape == (batch_size,)

    # Compute the decomposed policy losses.
    loss_policy_mean = compute_cross_entropy_loss(actions, normalized_weights,
                                                  policy_mean_distribution)
    loss_policy_stddev = compute_cross_entropy_loss(actions, normalized_weights,
                                                    policy_stddev_distribution)
    loss_policy_categorical = compute_cross_entropy_loss(
        actions, normalized_weights, policy_categorical_distribution)

    # Compute the decomposed KL between the target and online policies.
    kl_categorical = target_categorical_distribution.kl_divergence(
        policy_categorical_distribution.mixture_distribution)  # [B,]
    if self._per_dim_constraining:
      kl_mean = target_component_distribution.distribution.kl_divergence(
          policy_mean_distribution.components_distribution.distribution
      )  # Shape [B, L, D].
      kl_stddev = target_component_distribution.distribution.kl_divergence(
          policy_stddev_distribution.components_distribution.distribution
      )  # Shape [B, L, D].
      chex.assert_shape([kl_mean, kl_stddev],
                        (batch_size, num_components, action_size))
    else:
      kl_mean = target_component_distribution.kl_divergence(
          policy_mean_distribution.components_distribution)  # Shape [B, L].
      kl_stddev = target_component_distribution.kl_divergence(
          policy_stddev_distribution.components_distribution)  # Shape [B, L].
      chex.assert_shape([kl_mean, kl_stddev], (batch_size, num_components))

    # Compute the alpha-weighted KL-penalty and dual losses to adapt the alphas.
    loss_kl_mean, loss_alpha_mean = compute_parametric_kl_penalty_and_dual_loss(
        kl_mean, alpha_mean, self._epsilon_mean)
    loss_kl_stddev, loss_alpha_stddev = compute_parametric_kl_penalty_and_dual_loss(
        kl_stddev, alpha_stddev, self._epsilon_stddev)
    loss_kl_categorical, loss_alpha_categorical = (
        compute_parametric_kl_penalty_and_dual_loss(kl_categorical,
                                                    alpha_categorical,
                                                    self._epsilon_categorical))

    # Combine losses.
    loss_policy = loss_policy_mean + loss_policy_stddev + loss_policy_categorical
    loss_kl_penalty = loss_kl_mean + loss_kl_stddev + loss_kl_categorical
    loss_dual = (
        loss_alpha_mean + loss_alpha_stddev + loss_temperature +
        loss_alpha_categorical)
    loss = loss_policy + loss_kl_penalty + loss_dual

    # Create statistics.
    pi_stddev = online_component_distribution.distribution.stddev()
    stats = MPOStats(
        # Dual Variables.
        dual_alpha_mean=jnp.mean(alpha_mean),
        dual_alpha_stddev=jnp.mean(alpha_stddev),
        dual_temperature=jnp.mean(temperature),
        # Losses.
        loss_policy=jnp.mean(loss),
        loss_alpha=jnp.mean(loss_alpha_mean + loss_alpha_stddev),
        loss_temperature=jnp.mean(loss_temperature),
        # KL measurements.
        kl_q_rel=jnp.mean(kl_nonparametric) / self._epsilon,
        penalty_kl_q_rel=((jnp.mean(penalty_kl_nonparametric) /
                           self._epsilon_penalty)
                          if self._action_penalization else None),
        kl_mean_rel=jnp.mean(kl_mean, axis=0) / self._epsilon_mean,
        kl_stddev_rel=jnp.mean(kl_stddev, axis=0) / self._epsilon_stddev,
        # Q measurements.
        q_min=jnp.mean(jnp.min(q_values, axis=0)),
        q_max=jnp.mean(jnp.max(q_values, axis=0)),
        # If the policy has stddev, log summary stats for this as well.
        pi_stddev_min=jnp.mean(jnp.min(pi_stddev, axis=-1)),
        pi_stddev_max=jnp.mean(jnp.max(pi_stddev, axis=-1)),
        # Condition number of the diagonal covariance (actually, stddev) matrix.
        pi_stddev_cond=jnp.mean(
            jnp.max(pi_stddev, axis=-1) / jnp.min(pi_stddev, axis=-1)),
    )

    return loss, stats


def compute_weights_and_temperature_loss(
    q_values: jnp.ndarray,
    epsilon: float,
    temperature: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Computes normalized importance weights for the policy optimization.

  Args:
    q_values: Q-values associated with the actions sampled from the target
      policy; expected shape [N, B].
    epsilon: Desired constraint on the KL between the target and non-parametric
      policies.
    temperature: Scalar used to temper the Q-values before computing normalized
      importance weights from them. This is really the Lagrange dual variable in
      the constrained optimization problem, the solution of which is the
      non-parametric policy targeted by the policy loss.

  Returns:
    Normalized importance weights, used for policy optimization.
    Temperature loss, used to adapt the temperature.
  """

  # Temper the given Q-values using the current temperature.
  tempered_q_values = jax.lax.stop_gradient(q_values) / temperature

  # Compute the normalized importance weights used to compute expectations with
  # respect to the non-parametric policy.
  normalized_weights = jax.nn.softmax(tempered_q_values, axis=0)
  normalized_weights = jax.lax.stop_gradient(normalized_weights)

  # Compute the temperature loss (dual of the E-step optimization problem).
  q_logsumexp = jax.scipy.special.logsumexp(tempered_q_values, axis=0)
  log_num_actions = jnp.log(q_values.shape[0] / 1.)
  loss_temperature = epsilon + jnp.mean(q_logsumexp) - log_num_actions
  loss_temperature = temperature * loss_temperature

  return normalized_weights, loss_temperature


def compute_nonparametric_kl_from_normalized_weights(
    normalized_weights: jnp.ndarray) -> jnp.ndarray:
  """Estimate the actualized KL between the non-parametric and target policies."""

  # Compute integrand.
  num_action_samples = normalized_weights.shape[0] / 1.
  integrand = jnp.log(num_action_samples * normalized_weights + 1e-8)

  # Return the expectation with respect to the non-parametric policy.
  return jnp.sum(normalized_weights * integrand, axis=0)


def compute_cross_entropy_loss(
    sampled_actions: jnp.ndarray,
    normalized_weights: jnp.ndarray,
    online_action_distribution: tfd.Distribution,
) -> jnp.ndarray:
  """Compute cross-entropy online and the reweighted target policy.

  Args:
    sampled_actions: samples used in the Monte Carlo integration in the policy
      loss. Expected shape is [N, B, ...], where N is the number of sampled
      actions and B is the number of sampled states.
    normalized_weights: target policy multiplied by the exponentiated Q values
      and normalized; expected shape is [N, B].
    online_action_distribution: policy to be optimized.

  Returns:
    loss_policy_gradient: the cross-entropy loss that, when differentiated,
      produces the policy gradient.
  """

  # Compute the M-step loss.
  log_prob = online_action_distribution.log_prob(sampled_actions)

  # Compute the weighted average log-prob using the normalized weights.
  loss_policy_gradient = -jnp.sum(log_prob * normalized_weights, axis=0)

  # Return the mean loss over the batch of states.
  return jnp.mean(loss_policy_gradient, axis=0)


def compute_parametric_kl_penalty_and_dual_loss(
    kl: jnp.ndarray,
    alpha: jnp.ndarray,
    epsilon: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Computes the KL cost to be added to the Lagragian and its dual loss.

  The KL cost is simply the alpha-weighted KL divergence and it is added as a
  regularizer to the policy loss. The dual variable alpha itself has a loss that
  can be minimized to adapt the strength of the regularizer to keep the KL
  between consecutive updates at the desired target value of epsilon.

  Args:
    kl: KL divergence between the target and online policies.
    alpha: Lagrange multipliers (dual variables) for the KL constraints.
    epsilon: Desired value for the KL.

  Returns:
    loss_kl: alpha-weighted KL regularization to be added to the policy loss.
    loss_alpha: The Lagrange dual loss minimized to adapt alpha.
  """

  # Compute the mean KL over the batch.
  mean_kl = jnp.mean(kl, axis=0)

  # Compute the regularization.
  loss_kl = jnp.sum(jax.lax.stop_gradient(alpha) * mean_kl)

  # Compute the dual loss.
  loss_alpha = jnp.sum(alpha * (epsilon - jax.lax.stop_gradient(mean_kl)))

  return loss_kl, loss_alpha


def clip_mpo_params(params: MPOParams, per_dim_constraining: bool) -> MPOParams:
  clipped_params = MPOParams(
      log_temperature=jnp.maximum(_MIN_LOG_TEMPERATURE, params.log_temperature),
      log_alpha_mean=jnp.maximum(_MIN_LOG_ALPHA, params.log_alpha_mean),
      log_alpha_stddev=jnp.maximum(_MIN_LOG_ALPHA, params.log_alpha_stddev),
      log_alpha_categorical=jnp.maximum(_MIN_LOG_ALPHA,
                                        params.log_alpha_categorical))
  if not per_dim_constraining:
    return clipped_params
  else:
    return clipped_params._replace(
        log_penalty_temperature=jnp.maximum(_MIN_LOG_TEMPERATURE,
                                            params.log_penalty_temperature))
