import unittest


class MagiTest(unittest.TestCase):
  def test_magi(self):
    import magi
    del magi


if __name__ == '__main__':
  unittest.main()
