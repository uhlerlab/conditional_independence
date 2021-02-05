import numpy as np
from typing import Union, List, Optional
from ...utils import to_list
from scipy.special import stdtr, ncfdtr
from numpy.linalg import pinv


def gauss_invariance_suffstat(
        obs_samples,
        context_samples_list
):
    """
    Helper function to compute the sufficient statistics for the gauss_invariance_test from data.

    Parameters
    ----------
    obs_samples:
        (n x p) matrix, where n is the number of samples and p is the number of variables.
    context_samples_list:
        list of (n x p) matrices, one for each context besides observational

    Return
    ------
    dict
        dictionary of sufficient statistics
    """
    obs_samples = np.hstack((obs_samples, np.ones([obs_samples.shape[0], 1])))
    obs_cov = np.cov(obs_samples, rowvar=False)
    obs_suffstat = dict(samples=obs_samples, G=obs_samples.T @ obs_samples, S=obs_cov)
    context_suffstats = []
    for context_samples in context_samples_list:
        context_samples = np.hstack((context_samples, np.ones([context_samples.shape[0], 1])))
        context_cov = np.cov(context_samples, rowvar=False)
        context_suffstats.append(dict(samples=context_samples, G=context_samples.T @ context_samples, S=context_cov))

    return dict(obs=obs_suffstat, contexts=context_suffstats)
