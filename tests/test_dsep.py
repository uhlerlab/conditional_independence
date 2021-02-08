from unittest import TestCase
import unittest
from conditional_independence import kci_test
import numpy as np


class TestDsep(TestCase):
    def test_dsep(self):
        samples = np.random.normal(size=(100, 3))
        res = kci_test(samples, 0, 1)
        print(res)


if __name__ == '__main__':
    unittest.main()
