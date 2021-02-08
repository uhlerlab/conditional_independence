from unittest import TestCase
import unittest
from conditional_independence import partial_correlation_test, partial_correlation_suffstat
import numpy as np
import random


class TestPartialCorrelation(TestCase):
    def test_marginal_correlation(self):
        np.random.seed(1231231)
        random.seed(98)
        samples = np.random.normal(size=(100, 3))
        suffstat = partial_correlation_suffstat(samples)
        res = partial_correlation_test(suffstat, 0, 1)
        self.assertTrue(np.isclose(res["statistic"], 0.7318399778331555))
        self.assertTrue(np.isclose(res["p_value"], 0.46426624491465285))
        self.assertFalse(res["reject"])

    def test_partial_correlation_single(self):
        np.random.seed(1231231)
        random.seed(98)
        samples = np.random.normal(size=(100, 4))
        suffstat = partial_correlation_suffstat(samples)
        res = partial_correlation_test(suffstat, 0, 1, {2})
        self.assertTrue(np.isclose(res["statistic"], 0.15092242834388334,))
        self.assertTrue(np.isclose(res["p_value"], 0.8800369078764101,))
        self.assertFalse(res["reject"])

    def test_partial_correlation_2nodes(self):
        np.random.seed(1231231)
        random.seed(98)
        samples = np.random.normal(size=(100, 5))
        suffstat = partial_correlation_suffstat(samples)
        res = partial_correlation_test(suffstat, 0, 1, {2, 3})
        self.assertTrue(np.isclose(res["statistic"], 0.007180967582131507))
        self.assertTrue(np.isclose(res["p_value"], 0.9942704660764401,))
        self.assertFalse(res["reject"])

    def test_partial_correlation_large_inverse(self):
        np.random.seed(1231231)
        random.seed(98)
        samples = np.random.normal(size=(100, 5))
        suffstat = partial_correlation_suffstat(samples)
        res = partial_correlation_test(suffstat, 0, 1, {2, 3, 4})
        self.assertTrue(np.isclose(res["statistic"], 0.008489182518689698))
        self.assertTrue(np.isclose(res["p_value"], 0.9932266936890279))
        self.assertFalse(res["reject"])


if __name__ == '__main__':
    unittest.main()
