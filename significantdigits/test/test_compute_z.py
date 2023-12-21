import pytest

import significantdigits as sd
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
        sd.compute_z(np.array(), None, axis=0, error=error)


def test_compute_z_x_rv():
    """
    Test compute_z when Y is None
    X is split into two arrays, X1 and X2
    """
    x = np.array([1, 2, 3, 4])
    y = None

    z_rel = x[:2] / x[2:] - 1
    z = sd.compute_z(x, y, axis=0, error=sd.Error.Relative)
    assert np.all(np.equal(z, z_rel))

    z_abs = x[:2] - x[2:]
    z = sd.compute_z(x, y, axis=0, error=sd.Error.Absolute)
    assert np.all(np.equal(z, z_abs))

    x = np.array([1, 2, 3])
    # Invalid shape, must be a multiple of two
    with pytest.raises(sd.SignificantDigitsException):
        sd.compute_z(x, y, axis=0, error=sd.Error.Absolute)


def test_compute_z_x_rv_y_rv():
    """
    Test compute_z when X and Y are both arrays of same dimension
    """
    x = np.array([1, 2, 3, 4])
    y = np.array([1, 2, 3, 4]) + 1e-5

    z_rel = x / y - 1
    z = sd.compute_z(x, y, axis=0, error=sd.Error.Relative)
    assert np.all(np.equal(z, z_rel))

    z_abs = x - y
    z = sd.compute_z(x, y, axis=0, error=sd.Error.Absolute)
    assert np.all(np.equal(z, z_abs))


def test_compute_z_x_rv_y_scalar():
    """
    Test compute_z when X is an array and Y is a scalar
    """
    x = np.array([1, 2, 3, 4])
    y = np.array(1)

    z_rel = x / y - 1
    z = sd.compute_z(x, y, axis=0, error=sd.Error.Relative)
    assert np.all(np.equal(z, z_rel))

    z_abs = x - y
    z = sd.compute_z(x, y, axis=0, error=sd.Error.Absolute)
    assert np.all(np.equal(z, z_abs))


def tesT_compute_z_x_rv_y_srv():
    """
    Test compute_z when X is an array and Y is an array with ndim = X.ndim - 1
    """
    x = np.arange(3 * 3).reshape(3, 3, 3)

    indexes = [
        np.index_exp[np.newaxis, :, :],
        np.index_exp[:, np.newaxis, :],
        np.index_exp[:, :, np.newaxis],
    ]
    for axis, index in enumerate(indexes):
        y = x.sum(axis=axis)

        z_rel = x[index] / y - 1
        z = sd.compute_z(x, y, axis=axis, error=sd.Error.Relative)
        assert np.all(np.equal(z, z_rel))

        z_abs = x[index] - y
        z = sd.compute_z(x, y, axis=axis, error=sd.Error.Absolute)
        assert np.all(np.equal(z, z_abs))
