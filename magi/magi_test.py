"""Magi test placeholder."""
import unittest


class MagiTest(unittest.TestCase):
  def test_magi(self):
    import magi  # pylint:disable=import-outside-toplevel
    del magi


if __name__ == '__main__':
  unittest.main()
