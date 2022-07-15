"""Tests for MPO agent."""
from absl.testing import absltest
import acme
from acme import specs
from acme.testing import fakes
from acme.utils import loggers

from magi.agents import mpo


class RHPONetworksTestCase(absltest.TestCase):
  """Tests for RHPO agent."""

  def test_mpo_local_agent(self):
    pass


if __name__ == '__main__':
  absltest.main()
