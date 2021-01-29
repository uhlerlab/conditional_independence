from typing import Union, List


def dsep_test(
        dag,
        i,
        j,
        cond_set: Union[List[int], int]=None
):
    return dict(reject=not dag.dsep(i, j, cond_set))


def msep_test(
        mag,
        i,
        j,
        cond_set: Union[List[int], int]=None
):
    return dict(reject=not mag.msep(i, j, cond_set))
