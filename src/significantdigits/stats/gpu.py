
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


def is_available():
    """Return True if CuPy is installed"""
    return cp is not None


def iscupy(a):
    return cp is not None and isinstance(a, cp.ndarray)


def get_array_module(*arrays):
    """Return the array module (cupy or numpy) matching the given arrays

    Returns cupy if at least one of the arrays is a cupy.ndarray,
    numpy otherwise.
    """
    for a in arrays:
        if iscupy(a):
            return cp
    return np


def mean(*args, **kwargs):
    return cp.mean(*args, **kwargs)


def var(*args, **kwargs):
    return cp.var(*args, **kwargs)


def std(*args, **kwargs):
    return cp.std(*args, **kwargs)


def absolute_error(x, y):
    x = cp.asanyarray(x)
    y = cp.asanyarray(y)
    return cp.abs(x - y)


def relative_error(x, y):
    x = cp.asanyarray(x)
    y = cp.asanyarray(y)
    return cp.abs(x / y - 1)


def asarray(x):
    return cp.asanyarray(x)


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
    raise NotImplementedError(f"Method '{method}' not implemented in gpu dispatcher")
