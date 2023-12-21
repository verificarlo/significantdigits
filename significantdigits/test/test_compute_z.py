import pytest

import significantdigits as sd
from significantdigits._significantdigits import _compute_z
import numpy as np

# Test  for compute z function
# Tests:
# 1. Y is None
# 2. X.ndim == Y.ndim
# 3. X.ndim - 1 == Y.ndim
# 4. Y.ndim == 0
# 5. Invalid shape provided


@pytest.mark.parametrize("error", [sd.Error.Absolute, sd.Error.Relative])
def test_compute_z_none_none(error):
    """
    Test compute_z when X and Y are None
    """
    with pytest.raises(TypeError):
        _compute_z(np.array(), None, axis=0, error=error)


def test_compute_z_x_rv():
    """
    Test compute_z when Y is None
    X is split into two arrays, X1 and X2
    """
    x = np.array([1, 2, 3, 4])
    y = None

    z_rel = x[:2] / x[2:] - 1
    z = _compute_z(x, y, axis=0, error=sd.Error.Relative)
    assert np.all(np.equal(z, z_rel))

    z_abs = x[:2] - x[2:]
    z = _compute_z(x, y, axis=0, error=sd.Error.Absolute)
    assert np.all(np.equal(z, z_abs))

    x = np.array([1, 2, 3])
    # Invalid shape, must be a multiple of two
    with pytest.raises(sd.SignificantDigitsException):
        _compute_z(x, y, axis=0, error=sd.Error.Absolute)


def test_compute_z_x_rv_y_rv():
    """
    Test compute_z when X and Y are both arrays of same dimension
    """
    x = np.array([1, 2, 3, 4])
    y = np.array([1, 2, 3, 4]) + 1e-5

    z_rel = x / y - 1
    z = _compute_z(x, y, axis=0, error=sd.Error.Relative)
    assert np.all(np.equal(z, z_rel))

    z_abs = x - y
    z = _compute_z(x, y, axis=0, error=sd.Error.Absolute)
    assert np.all(np.equal(z, z_abs))


def test_compute_z_x_rv_y_scalar():
    """
    Test compute_z when X is an array and Y is a scalar
    """
    x = np.array([1, 2, 3, 4])
    y = np.array(1)

    z_rel = x / y - 1
    z = _compute_z(x, y, axis=0, error=sd.Error.Relative)
    assert np.all(np.equal(z, z_rel))

    z_abs = x - y
    z = _compute_z(x, y, axis=0, error=sd.Error.Absolute)
    assert np.all(np.equal(z, z_abs))


def test_compute_z_x_rv_y_ref():
    """
    Test compute_z when X is an array and Y is an array with ndim = X.ndim - 1
    """
    x = np.arange(3**3).reshape(3, 3, 3)
    y = x.sum(axis=0)

    z_rel = x / y - 1
    z_abs = x - y

    z_rel_axis = x / y[np.newaxis, :, :] - 1
    z_abs_axis = x - y[np.newaxis, :, :]

    assert np.all(np.equal(z_rel, z_rel_axis))
    assert np.all(np.equal(z_abs, z_abs_axis))

    indexes = [
        np.index_exp[np.newaxis, :, :],
        np.index_exp[:, np.newaxis, :],
        np.index_exp[:, :, np.newaxis],
    ]
    for axis, index in enumerate(indexes):
        y = x.sum(axis=axis)

        z_rel = x / y[index] - 1
        z = _compute_z(x, y, axis=axis, error=sd.Error.Relative)
        assert np.all(np.equal(z, z_rel))

        z_abs = x - y[index]
        z = _compute_z(x, y, axis=axis, error=sd.Error.Absolute)
        assert np.all(np.equal(z, z_abs))
