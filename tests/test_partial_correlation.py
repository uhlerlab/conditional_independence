from unittest import TestCase
import unittest
from conditional_independence import compute_partial_correlation, partial_correlation_suffstat
import numpy as np


class TestPartialCorrelation(TestCase):
    def test_marginal_correlation(self):
        samples = np.random.normal(size=(100, 3))
        suffstat = partial_correlation_suffstat(samples)
        rho = compute_partial_correlation(suffstat, 0, 1)

    def test_partial_correlation_single(self):
        samples = np.random.normal(size=(100, 4))
        suffstat = partial_correlation_suffstat(samples)
        rho = compute_partial_correlation(suffstat, 0, 1, {2})

    def test_partial_correlation(self):
        samples = np.random.normal(size=(100, 5))
        suffstat = partial_correlation_suffstat(samples)
        rho = compute_partial_correlation(suffstat, 0, 1, {2, 3})

if __name__ == '__main__':
    unittest.main()
