
from significantdigits.stats import (dense, sparse)


def dispatch(a, method=None):
    if method is None:
        raise NotImplementedError(method)

    if sparse.issparse(a):
        return sparse.dispatcher(method)

    if dense.isdense(a):
        return dense.dispatcher(method)

    raise TypeError(type(a))
