from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import numpy as np
import rlax

from magi.agents.crr import distributions
from magi.agents.crr import losses


class CategoricalTDLearningTest(parameterized.TestCase):
    """Tests taken from rlax to test categorical td learning"""

    def setUp(self):
        super().setUp()
        self.atoms = np.array([0.5, 1.0, 1.5], dtype=np.float32)

        self.logits_tm1 = np.array(
            [[0, 9, 0], [9, 0, 9], [0, 9, 0], [9, 9, 0], [9, 0, 9]], dtype=np.float32
        )
        self.r_t = np.array([0.5, 0.0, 0.5, 0.8, -0.1], dtype=np.float32)
        self.discount_t = np.array([0.8, 1.0, 0.8, 0.0, 1.0], dtype=np.float32)
        self.logits_t = np.array(
            [[0, 0, 9], [1, 1, 1], [0, 0, 9], [1, 1, 1], [0, 9, 9]], dtype=np.float32
        )

        self.expected = np.array(
            [8.998915, 3.6932087, 8.998915, 0.69320893, 5.1929307], dtype=np.float32
        )

    @chex.all_variants()  # pylint: disable=all
    def test_categorical_td_learning_batch(self):
        """Tests for a full batch."""
        # Not using vmap for atoms.
        def fn(v_logits_tm1, r_t, discount_t, v_logits_t):
            return rlax.categorical_td_learning(
                v_atoms_tm1=self.atoms,
                v_logits_tm1=v_logits_tm1,
                r_t=r_t,
                discount_t=discount_t,
                v_atoms_t=self.atoms,
                v_logits_t=v_logits_t,
            )

        categorical_td_learning = self.variant(jax.vmap(fn))
        # Test outputs.
        actual = categorical_td_learning(
            self.logits_tm1, self.r_t, self.discount_t, self.logits_t
        )
        np.testing.assert_allclose(self.expected, actual, rtol=1e-5)

    def test_categorical_td_learning_batch_acme(self):
        """Tests for a full batch."""
        actual = losses.categorical(
            distributions.DiscreteValuedDistribution(
                logits=self.logits_tm1, values=self.atoms
            ),
            self.r_t,
            self.discount_t,
            distributions.DiscreteValuedDistribution(
                logits=self.logits_t, values=self.atoms
            ),
        )
        np.testing.assert_allclose(self.expected, actual, rtol=1e-5)


if __name__ == "__main__":
    absltest.main()
