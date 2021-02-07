from unittest import TestCase
import unittest
from conditional_independence import hsic_invariance_test
import numpy as np
import random


class TestHSICInvariance(TestCase):
    def test_marginal_invariance(self):
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

    def test_conditional_invariance1(self):
        np.random.seed(123)
        random.seed(123)
        samples1 = np.random.normal(size=(100, 3))
        samples2 = np.random.normal(size=(100, 3))
        suffstat = {"obs_samples": samples1, 0: samples2}
        res = hsic_invariance_test(suffstat, 0, 0, cond_set=1)
        self.assertTrue(np.isclose(res["statistic"], 0.0015783198618706637))
        self.assertTrue(np.isclose(res["p_value"], 0.09143792620832669))
        self.assertTrue(np.isclose(res["mean_approx"], 0.0007521989930065182))
        self.assertTrue(np.isclose(res["var_approx"], 3.4184267470157246e-07))
        self.assertFalse(res["reject"])

    def test_conditional_invariance2(self):
        np.random.seed(123)
        random.seed(123)
        samples1 = np.random.normal(size=(100, 3))
        samples2 = np.random.normal(size=(100, 3))
        suffstat = {"obs_samples": samples1, 0: samples2}
        res = hsic_invariance_test(suffstat, 0, 0, cond_set=[1, 2])
        self.assertTrue(np.isclose(res["statistic"], 0.0019570107453586944))
        self.assertTrue(np.isclose(res["p_value"], 0.03746439447513661))
        self.assertTrue(np.isclose(res["mean_approx"], 0.0007269496716795982))
        self.assertTrue(np.isclose(res["var_approx"], 3.100689837426494e-07))
        self.assertTrue(res["reject"])


if __name__ == '__main__':
    unittest.main()
