
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

_CUPY_MISSING_ERROR = (
    "CuPy is required for GPU operations but is not installed. "
    "Install it with `pip install cupy`."
)


def _require_cupy():
    if cp is None:
        raise ImportError(_CUPY_MISSING_ERROR)


def is_available():
    """Return True if CuPy is installed and a CUDA device is usable"""
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


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
    _require_cupy()
    return cp.mean(*args, **kwargs)


def var(*args, **kwargs):
    _require_cupy()
    return cp.var(*args, **kwargs)


def std(*args, **kwargs):
    _require_cupy()
    return cp.std(*args, **kwargs)


def absolute_error(x, y):
    _require_cupy()
    x = cp.asanyarray(x)
    y = cp.asanyarray(y)
    return cp.abs(x - y)


def relative_error(x, y):
    _require_cupy()
    x = cp.asanyarray(x)
    y = cp.asanyarray(y)
    return cp.abs(x / y - 1)


def asarray(x):
    _require_cupy()
    return cp.asanyarray(x)


def dispatcher(method):
    _require_cupy()
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
