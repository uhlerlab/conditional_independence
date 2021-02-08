from unittest import TestCase
import unittest
from conditional_independence import PlainCI_Tester, MemoizedCI_Tester
from conditional_independence import partial_correlation_test, partial_correlation_suffstat
import numpy as np
import random


class TestCITester(TestCase):
    def test_plain_ci_tester(self):
        np.random.seed(1231)
        random.seed(1231)
        samples = np.random.normal(size=(100, 3))
        suffstat = partial_correlation_suffstat(samples)
        ci_tester = PlainCI_Tester(partial_correlation_test, suffstat)
        self.assertTrue(ci_tester.is_ci(0, 1))

    def test_memoized_ci_tester(self):
        np.random.seed(1231)
        random.seed(1231)
        samples = np.random.normal(size=(100, 3))
        suffstat = partial_correlation_suffstat(samples)
        ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat)
        self.assertTrue(ci_tester.is_ci(0, 1))
        ci_tester.clear()

    def test_memoized_ci_tester_detailed(self):
        np.random.seed(1231)
        random.seed(1231)
        samples = np.random.normal(size=(100, 3))
        suffstat = partial_correlation_suffstat(samples)
        ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, detailed=True)
        self.assertTrue(ci_tester.is_ci(0, 1))

    def test_memoized_ci_tester_times(self):
        np.random.seed(1231)
        random.seed(1231)
        samples = np.random.normal(size=(100, 3))
        suffstat = partial_correlation_suffstat(samples)
        ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, track_times=True)
        self.assertTrue(ci_tester.is_ci(0, 1))


if __name__ == '__main__':
    unittest.main()
