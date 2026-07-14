
from significantdigits.stats import (dense, gpu, sparse)


def dispatch(a, *arrays, method=None):
    if method is None and arrays and isinstance(arrays[-1], str):
        method = arrays[-1]
        arrays = arrays[:-1]

    if method is None:
        raise NotImplementedError(method)

    arrays = (a,) + arrays

    if any(gpu.iscupy(a) for a in arrays):
        return gpu.dispatcher(method)

    if any(sparse.issparse(a) for a in arrays):
        return sparse.dispatcher(method)

    if dense.isdense(a):
        return dense.dispatcher(method)

    raise TypeError(type(a))
