from unittest import TestCase
import unittest
from conditional_independence import hsic_invariance_test
import numpy as np
import random


class TestHSICInvariance(TestCase):
    def test_marginal_independence(self):
        np.random.seed(123)
        random.seed(123)
        samples1 = np.random.normal(size=(100, 3))
        samples2 = np.random.normal(size=(100, 3))
        suffstat = {"obs_samples": samples1, 0: samples2}
        res = hsic_invariance_test(suffstat, 0, 0)
        self.assertTrue(np.isclose(res["statistic"], 0.0015871445309950237))
        self.assertTrue(np.isclose(res["p_value"], 0.11989454560151747))
        self.assertTrue(np.isclose(res["mean_approx"], 0.0008173708103847649))
        self.assertTrue(np.isclose(res["var_approx"], 4.417792594600709e-07))
        self.assertFalse(res["reject"])



if __name__ == '__main__':
    unittest.main()
