"""Distributions used by CRR agent"""

from typing import Optional

from tensorflow_probability.python.internal.backend.jax.compat import v2 as tf
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax.internal import tensor_util

tfd = tfp.distributions


class DiscreteValuedDistribution(tfd.Categorical):
    """This is a generalization of a categorical distribution.

    The support for the DiscreteValued distribution can be any real valued range,
    whereas the categorical distribution has support [0, n_categories - 1] or
    [1, n_categories]. This generalization allows us to take the mean of the
    distribution over its support.
    """

    def __init__(
        self,
        values: tf.Tensor,
        logits: Optional[tf.Tensor] = None,
        probs: Optional[tf.Tensor] = None,
        name: str = "DiscreteValuedDistribution",
    ):
        """Initialization.

        Args:
          values: Values making up support of the distribution. Should have a shape
            compatible with logits.
          logits: An N-D Tensor, N >= 1, representing the log probabilities of a set
            of Categorical distributions. The first N - 1 dimensions index into a
            batch of independent distributions and the last dimension indexes into
            the classes.
          probs: An N-D Tensor, N >= 1, representing the probabilities of a set of
            Categorical distributions. The first N - 1 dimensions index into a batch
            of independent distributions and the last dimension represents a vector
            of probabilities for each class. Only one of logits or probs should be
            passed in.
          name: Name of the distribution object.
        """
        self._values = tensor_util.convert_nonref_to_tensor(values)
        # shape_strings = [f"D{i}" for i, _ in enumerate(values.shape)]

        if logits is not None:
            logits = tensor_util.convert_nonref_to_tensor(logits)
            # tf.debugging.assert_shapes(
            #     [(values, shape_strings), (logits, [..., *shape_strings])]
            # )
        if probs is not None:
            probs = tensor_util.convert_nonref_to_tensor(probs)
            # tf.debugging.assert_shapes(
            #     [(values, shape_strings), (probs, [..., *shape_strings])]
            # )

        super().__init__(logits=logits, probs=probs, name=name)

        self._parameters = dict(values=values, logits=logits, probs=probs, name=name)

    @property
    def values(self) -> tf.Tensor:
        return self._values

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return dict(
            values=tfp.util.ParameterProperties(event_ndims=None),
            logits=tfp.util.ParameterProperties(
                event_ndims=lambda self: len(self.values.shape)
            ),
            probs=tfp.util.ParameterProperties(
                event_ndims=lambda self: len(self.values.shape), is_preferred=False
            ),
        )

    def _sample_n(self, n, seed=None) -> tf.Tensor:
        indices = super()._sample_n(n, seed=seed)
        return tf.gather(self.values, indices, axis=-1)

    def _mean(self) -> tf.Tensor:
        """Overrides the Categorical mean by incorporating category values."""
        return tf.reduce_sum(self.probs_parameter() * self.values, axis=-1)

    def _variance(self) -> tf.Tensor:
        """Overrides the Categorical variance by incorporating category values."""
        dist_squared = tf.square(tf.expand_dims(self.mean(), -1) - self.values)
        return tf.reduce_sum(self.probs_parameter() * dist_squared, axis=-1)

    def _event_shape(self):
        # Omit the atoms axis, to return just the shape of a single (i.e. unbatched)
        # sample value.
        return self._values.shape[:-1]

    def _event_shape_tensor(self):
        return tf.shape(self._values)[:-1]
