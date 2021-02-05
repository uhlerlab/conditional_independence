[![PyPI version](https://badge.fury.io/py/conditional_independence.svg)](https://badge.fury.io/py/conditional_independence)
[![Build Status](https://travis-ci.com/uhlerlab/conditional_independence.svg?branch=main)](https://travis-ci.com/uhlerlab/conditional_independence)
[![codecov](https://codecov.io/gh/uhlerlab/conditional_independence/branch/main/graph/badge.svg?token=TC78IEMINI)](https://codecov.io/gh/uhlerlab/conditional_independence)

`conditional_independence` is a Python package for conditional independence testing.

### Install
Install the latest version of `conditional_independence`:
```
$ pip3 install conditional_independence
```

### Documentation
Documentation is available at https://conditional-independence.readthedocs.io/en/latest/


### Simple Example

```
>>> from conditional_independence import partial_correlation_suffstat, partial_correlation_test
>>> import numpy as np
>>> np.random.seed(121122)
>>> samples = np.random.normal(size=(100, 3))
>>> suffstat = partial_correlation_suffstat(samples)
>>> partial_correlation_test(suffstat, 0, 1)
{'statistic': 0.5671513111036371,
 'p_value': 0.5706113842986253,
 'reject': False}
>>> partial_correlation_test(suffstat, 0, 1, {2})
{'statistic': 0.6879909848126664,
 'p_value': 0.4914584585239892,
 'reject': False}
```

### License

Released under the 3-Clause BSD license (see LICENSE.txt):
```
Copyright (C) 2021
Chandler Squires <csquires@mit.edu>
```
