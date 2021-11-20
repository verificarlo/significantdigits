
import numpy as np
import scipy


def issparse(a):
    if scipy.sparse.issparse(a):
        return True
    elif isinstance(a, np.ndarray) and a.ndim == 0:
        return scipy.sparse.issparse(a.item())
    elif isinstance(a, np.ndarray) and a.ndim != 0:
        return any(map(scipy.sparse.issparse, a))
    else:
        return False


def mean(a, /, *args, axis=None, **kwargs):
    return np.sum(a, axis=axis, *args, **kwargs)/len(a)


def var(a, /, *args, **kwargs):
    return mean(a**2, *args, **kwargs) - np.square(mean(a, *args, **kwargs))


def std(a, /, *args, **kwargs):
    return np.sqrt(var(a, *args, **kwargs))


def absolute_error(x, y):
    return np.abs(x - y)


def relative_error(x, y):
    _y_inv = scipy.sparse.coo_matrix(y).power(-1)
    return np.asarray([(i - y) * _y_inv for i in x])


def asarray(x):
    return x


def dispatcher(method):
    if method == 'mean':
        return mean
    if method == 'std':
        return std
    if method == 'var':
        return var
    if method == 'absolute_error':
        return absolute_error
    if method == 'relative_error':
        return relative_error
    if method == 'asarray':
        return asarray
