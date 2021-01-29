from unittest import TestCase
import unittest
from conditional_independence import hsic_test
import numpy as np


class TestHSIC(TestCase):
    def test_marginal_independence(self):
        samples = np.random.normal(size=(100, 3))
        res = hsic_test(samples, 0, 1)
        print(res)


if __name__ == '__main__':
    unittest.main()
