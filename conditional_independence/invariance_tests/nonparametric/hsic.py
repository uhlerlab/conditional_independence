from typing import Union, List, Optional
import numpy as np
from conditional_independence.utils import combined_mat, to_list
from conditional_independence.ci_tests.nonparametric.hsic import hsic_test


def hsic_invariance_test(
        suffstat,
        context,
        i: int,
        cond_set: Optional[Union[List[int], int]]=None,
        alpha: float=0.05
):
    """
    TODO

    Parameters
    ----------
    TODO

    Examples
    --------
    TODO
    """
    cond_set = to_list(cond_set)
    obs_samples = suffstat['obs_samples']
    iv_samples = suffstat[context]

    mat = combined_mat(obs_samples, iv_samples, i, cond_set)
    return hsic_test(mat, 0, 1, list(range(2, 2+len(cond_set))), alpha=alpha)
