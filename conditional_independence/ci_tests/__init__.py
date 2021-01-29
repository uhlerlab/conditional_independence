from .ci_tester import *
from .oracle import *
from .parametric import *
from .nonparametric import *


def get_ci_tester(
        samples,
        test="partial_correlation",
        memoize=False,
        **kwargs
):
    if test == "partial_correlation":
        ci_test = partial_correlation_test
        suffstat = partial_correlation_suffstat(samples)
    elif test == "hsic":
        ci_test = hsic_test
        suffstat = samples
    elif test == "kci":
        ci_test = kci_test
        suffstat = samples
    elif test == "dsep":
        ci_test = dsep_test
        suffstat = samples
    elif test == "msep":
        ci_test = msep_test
        suffstat = samples
    else:
        raise ValueError()

    if memoize:
        return MemoizedCI_Tester(ci_test, suffstat, **kwargs)
