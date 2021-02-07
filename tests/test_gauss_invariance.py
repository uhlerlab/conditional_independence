from unittest import TestCase
import unittest
from conditional_independence import gauss_invariance_test, gauss_invariance_suffstat, MemoizedInvarianceTester
import numpy as np
import random


class TestGaussInvariance(TestCase):
    def test_marginal_invariance(self):
        np.random.seed(121)
        random.seed(121)
        samples = np.random.normal(size=(100, 3))
        samples2 = np.random.normal(size=(200, 3))
        suffstat = gauss_invariance_suffstat(samples, [samples2])
        res = gauss_invariance_test(suffstat, 0, 0)
        self.assertTrue(np.isclose(res["ftest_stat"], 1.196161251519466))
        self.assertTrue(np.isclose(res["f_pvalue"], 0.2904060474248411))
        self.assertTrue(np.isclose(res["rc_stat"], 0.63948672845725))
        self.assertTrue(np.isclose(res["rc_pvalue"], 0.8490631212686699))

    def test_conditional_invariance_1node(self):
        np.random.seed(121)
        random.seed(121)
        samples = np.random.normal(size=(100, 3))
        samples2 = np.random.normal(size=(200, 3))
        suffstat = gauss_invariance_suffstat(samples, [samples2])
        res = gauss_invariance_test(suffstat, 0, 0, cond_set=1)
        self.assertTrue(np.isclose(res["ftest_stat"], 1.2073363913933624))
        self.assertTrue(np.isclose(res["f_pvalue"], 0.2663061302206631))
        self.assertTrue(np.isclose(res["rc_stat"], 0.4169454062701565))
        self.assertTrue(np.isclose(res["rc_pvalue"], 0.6811184647527357))

    def test_conditional_invariance_2nodes(self):
        np.random.seed(121)
        random.seed(121)
        samples = np.random.normal(size=(100, 3))
        samples2 = np.random.normal(size=(200, 3))
        suffstat = gauss_invariance_suffstat(samples, [samples2])
        res = gauss_invariance_test(suffstat, 0, 0, cond_set=[1, 2])
        self.assertTrue(np.isclose(res["ftest_stat"], 1.213427074984623))
        self.assertTrue(np.isclose(res["f_pvalue"], 0.25384731599427557))
        self.assertTrue(np.isclose(res["rc_stat"], 0.3723302924116713))
        self.assertTrue(np.isclose(res["rc_pvalue"], 0.45394498735399946))


if __name__ == '__main__':
    unittest.main()
