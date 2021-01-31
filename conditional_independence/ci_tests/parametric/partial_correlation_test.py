from typing import Dict
from math import erf
from numpy import sqrt, log1p, abs, ix_, diag, corrcoef, errstate, cov, mean
from numpy.linalg import inv, pinv
# from . import MemoizedCI_Tester


__all__ = [
    "partial_correlation_suffstat",
    "partial_correlation_test",
    "compute_partial_correlation"
]


def numba_inv(A):
    return inv(A)


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
    mu = mean(samples, axis=0)
    # TODO: NaN when variable is deterministic. Replace w/ 1 and 0?
    C = corrcoef(samples, rowvar=False)  # sample correlation matrix
    V = samples.T @ samples
    if invert:
        K = pinv(C)
        P = pinv(S)  # sample precision (inverse covariance) matrix
        rho = K/sqrt(diag(K))/sqrt(diag(K))[:, None]  # sample partial correlation matrix
        return dict(P=P, S=S, C=C, n=n, K=K, rho=rho, mu=mu, V=V)
    return dict(S=S, C=C, n=n, mu=mu, V=V)


def compute_partial_correlation(suffstat, i, j, cond_set=None):
    """
    Compute the partial correlation between i and j given ``cond_set``.

    Parameters
    ----------
    suffstat:
        dictionary containing:
        'n' -- number of samples
        'C' -- correlation matrix
        'K' (optional) -- inverse correlation matrix
        'rho' (optional) -- partial correlation matrix (K, normalized so diagonals are 1).
    i:
        position of first variable in correlation matrix.
    j:
        position of second variable in correlation matrix.
    cond_set:
        positions of conditioning set in correlation matrix.

    Returns
    -------
    float
        partial correlation
    """
    C = suffstat.get('C')
    p = C.shape[0]
    rho = suffstat.get('rho')
    K = suffstat.get('K')

    # === COMPUTE PARTIAL CORRELATION
    # partial correlation is correlation if there is no conditioning
    if cond_set is None or len(cond_set) == 0:
        r = C[i, j]
    # used closed-form
    elif len(cond_set) == 1:
        k = list(cond_set)[0]
        r = (C[i, j] - C[i, k]*C[j, k]) / sqrt((1 - C[j, k]**2) * (1 - C[i, k]**2))
    # when conditioning on everything, partial correlation comes from normalized precision matrix
    elif len(cond_set) == p - 2 and rho is not None:
        r = -rho[i, j]
    # faster to use Schur complement if conditioning set is large and precision matrix is pre-computed
    elif len(cond_set) >= p/2 and K is not None:
        rest = list(set(range(C.shape[0])) - {i, j, *cond_set})

        if len(rest) == 1:
            theta_ij = K[ix_([i, j], [i, j])] - K[ix_([i, j], rest)] @ K[ix_(rest, [i, j])] / K[rest[0], rest[0]]
        else:
            theta_ij = K[ix_([i, j], [i, j])] - K[ix_([i, j], rest)] @ pinv(K[ix_(rest, rest)]) @ K[ix_(rest, [i, j])]  # TODO: what to do if not invertible?
        r = -theta_ij[0, 1] / sqrt(theta_ij[0, 0] * theta_ij[1, 1])
    else:
        theta = pinv(C[ix_([i, j, *cond_set], [i, j, *cond_set])])  # TODO: what to do if not invertible?
        r = -theta[0, 1]/sqrt(theta[0, 0] * theta[1, 1])

    return r


def partial_monte_carlo_correlation_suffstat(samples, invert=True) -> Dict:
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
    mu = mean(samples, axis=0)
    # TODO: NaN when variable is deterministic. Replace w/ 1 and 0?
    C = corrcoef(samples, rowvar=False)  # sample correlation matrix
    if invert:
        K = pinv(C)
        P = pinv(S)  # sample precision (inverse covariance) matrix
        rho = K/sqrt(diag(K))/sqrt(diag(K))[:, None]  # sample partial correlation matrix
        return dict(P=P, S=S, C=C, n=n, K=K, rho=rho, mu=mu, samples=samples)
    return dict(S=S, C=C, n=n, mu=mu, samples=samples)


def partial_correlation_test(suffstat: Dict, i, j, cond_set=None, alpha=None):
    """
    Test the null hypothesis that i and j are conditionally independent given ``cond_set``.

    Uses Fisher's z-transform.

    Parameters
    ----------
    suffstat:
        dictionary containing:

        * ``n`` -- number of samples
        * ``C`` -- correlation matrix
        * ``K`` (optional) -- inverse correlation matrix
        * ``rho`` (optional) -- partial correlation matrix (K, normalized so diagonals are 1).
    i:
        position of first variable in correlation matrix.
    j:
        position of second variable in correlation matrix.
    cond_set:
        positions of conditioning set in correlation matrix.
    alpha:
        Significance level.

    Returns
    -------
    dict
        dictionary containing:

        * ``statistic``
        * ``p_value``
        * ``reject``
    """
    n = suffstat['n']
    n_cond = 0 if cond_set is None else len(cond_set)
    alpha = 1/n if alpha is None else alpha

    r = compute_partial_correlation(suffstat, i, j, cond_set=cond_set)

    # === COMPUTE STATISTIC AND P-VALUE
    # note: log1p(2r/(1-r)) = log((1+r)/(1-r)) but is more numerically stable for r near 0
    # r = 1 causes warnings but gives the correct answer
    with errstate(divide='ignore', invalid='ignore'):
        statistic = sqrt(n - n_cond - 3) * abs(.5 * log1p(2*r/(1 - r)))
    # note: erf is much faster than norm.cdf
    p_value = 2*(1 - .5*(1 + erf(statistic/sqrt(2))))

    return dict(statistic=statistic, p_value=p_value, reject=p_value < alpha)


# class MemoizedGaussCI_Tester(MemoizedCI_Tester):
#     def __init__(self, suffstat: Dict, track_times=False, detailed=False, **kwargs):
#         MemoizedCI_Tester.__init__(self, partial_correlation_test, suffstat, track_times=track_times, detailed=detailed)


if __name__ == '__main__':
    import numpy as np

    x = np.random.normal(size=(100, 3))
    s = partial_correlation_suffstat(x)
    res = partial_correlation_test(s, 0, 1)
    print(res)


