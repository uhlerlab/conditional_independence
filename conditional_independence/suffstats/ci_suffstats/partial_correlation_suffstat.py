from typing import Dict
from numpy import sqrt, diag, corrcoef, cov
import numpy as np
from numpy.linalg import pinv


def partial_correlation_suffstats_interventional(intervention_info: Dict):
    return {intervened_nodes: partial_correlation_suffstat(samples) for intervened_nodes, samples in intervention_info.items()}


def partial_correlation_suffstat(samples, invert=True) -> Dict:
    """
    Return the sufficient statistics for partial correlation testing.

    Parameters
    ----------
    samples:
        (n x p) matrix, where n is the number of samples and p is the number of variables.
    invert:
        if True, compute the inverse correlation matrix, and normalize it into the partial correlation matrix. This
        will generally speed up the gauss_ci_test if large conditioning sets are used.

    Returns
    -------
    dict
        dictionary of sufficient statistics
    """
    n, p = samples.shape
    S = cov(samples, rowvar=False)  # sample covariance matrix
    mu = np.mean(samples, axis=0)
    # TODO: NaN when variable is deterministic. Replace w/ 1 and 0?
    C = corrcoef(samples, rowvar=False)  # sample correlation matrix
    V = samples.T @ samples
    if invert:
        K = pinv(C)
        P = pinv(S)  # sample precision (inverse covariance) matrix
        rho = K/sqrt(diag(K))/sqrt(diag(K))[:, None]  # sample partial correlation matrix
        return dict(P=P, S=S, C=C, n=n, K=K, rho=rho, mu=mu, V=V)
    return dict(S=S, C=C, n=n, mu=mu, V=V)
