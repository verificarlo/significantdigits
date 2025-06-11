
from icecream import ic
import numpy as np
import scipy


def isdense(a):
    return not scipy.sparse.issparse(a)


def mean(*args, **kwargs):
    return np.mean(*args, **kwargs)


def var(*args, **kwargs):
    return np.var(*args, **kwargs)


def std(*args, **kwargs):
    return np.std(*args, **kwargs)


def absolute_error(x, y):
    return np.abs(x - y)


def relative_error(x, y):
    return np.abs(x/y - 1)


def asarray(x):
    return np.asanyarray(x)


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
