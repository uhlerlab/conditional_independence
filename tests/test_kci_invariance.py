from unittest import TestCase
import unittest
from conditional_independence import kci_invariance_test
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


class TestKCIInvariance(TestCase):
    def test_marginal_invariance(self):
        np.random.seed(123)
        random.seed(123)
        samples1 = np.random.normal(size=(100, 3))
        samples2 = np.random.normal(size=(100, 3))
        suffstat = {"obs_samples": samples1, 0: samples2}
        res = kci_invariance_test(suffstat, 0, 0)
        self.assertTrue(np.isclose(res["statistic"], 0.49628849461621344))
        self.assertTrue(np.isclose(res["critval"], 0.6469068502874128))
        self.assertTrue(np.isclose(res["p_value"], 0.1083970459442456))
        self.assertFalse(res["reject"])

    def test_conditional_invariance1(self):
        np.random.seed(123)
        random.seed(123)
        samples1 = np.random.normal(size=(100, 3))
        samples2 = np.random.normal(size=(100, 3))
        suffstat = {"obs_samples": samples1, 0: samples2}
        res = kci_invariance_test(suffstat, 0, 0, cond_set=1)
        self.assertTrue(np.isclose(res["statistic"], 0.5492539874089464))
        self.assertTrue(np.isclose(res["critval"], 0.6310289276236885))
        self.assertTrue(np.isclose(res["p_value"], 0.08571151570394275,))
        self.assertFalse(res["reject"])

    def test_conditional_invariance2(self):
        np.random.seed(123)
        random.seed(123)
        samples1 = np.random.normal(size=(100, 3))
        samples2 = np.random.normal(size=(100, 3))
        suffstat = {"obs_samples": samples1, 0: samples2}
        res = kci_invariance_test(suffstat, 0, 0, cond_set=[1, 2])
        self.assertTrue(np.isclose(res["statistic"], 0.6470752135761941))
        self.assertTrue(np.isclose(res["critval"], 0.6310001292988091))
        self.assertTrue(np.isclose(res["p_value"], 0.044441562110249966))
        self.assertTrue(res["reject"])


if __name__ == '__main__':
    unittest.main()
