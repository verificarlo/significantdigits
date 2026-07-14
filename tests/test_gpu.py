"""
Tests for the CuPy GPU backend

Dispatch and detection tests run everywhere.
Tests marked with ``gpu`` require CuPy and a usable CUDA device
and are skipped otherwise.
"""

import numpy as np
import pytest

import significantdigits as sd
from significantdigits import stats
from significantdigits.stats import dense, gpu

try:
    import cupy as cp

    try:
        cp.cuda.runtime.getDeviceCount()
        GPU_AVAILABLE = True
    except Exception:
        GPU_AVAILABLE = False
except ImportError:
    cp = None
    GPU_AVAILABLE = False

requires_gpu = pytest.mark.skipif(
    not GPU_AVAILABLE, reason="requires CuPy and a usable CUDA device"
)


def assert_matches_numpy(gpu_result, cpu_result, rtol=1e-12):
    """Check gpu_result is a cupy array numerically equal to cpu_result"""
    assert gpu.iscupy(gpu_result)
    np.testing.assert_allclose(cp.asnumpy(gpu_result), cpu_result, rtol=rtol)


class TestGpuDetection:
    """Detection and dispatch behavior that does not require a GPU"""

    def test_is_available_returns_bool(self):
        assert isinstance(gpu.is_available(), bool)

    def test_iscupy_rejects_numpy_array(self):
        assert not gpu.iscupy(np.array([1.0, 2.0]))

    def test_iscupy_rejects_non_array(self):
        assert not gpu.iscupy([1.0, 2.0])
        assert not gpu.iscupy(None)

    def test_get_array_module_defaults_to_numpy(self):
        assert gpu.get_array_module(np.array([1.0]), None, [1.0]) is np

    def test_dispatch_numpy_routes_to_dense(self):
        from significantdigits.stats.dispatch import dispatch

        assert dispatch(np.array([1.0, 2.0]), "mean") is dense.mean

    def test_gpu_dispatcher_unknown_method(self):
        with pytest.raises(NotImplementedError):
            gpu.dispatcher("unknown_method")


@pytest.mark.gpu
@requires_gpu
class TestGpuStats:
    """Stats backend functions on cupy arrays"""

    def test_iscupy_accepts_cupy_array(self):
        assert gpu.iscupy(cp.asarray([1.0, 2.0]))

    def test_get_array_module_returns_cupy(self):
        assert gpu.get_array_module(np.array([1.0]), cp.asarray([1.0])) is cp

    def test_dispatch_cupy_routes_to_gpu(self):
        from significantdigits.stats.dispatch import dispatch

        assert dispatch(cp.asarray([1.0, 2.0]), "mean") is gpu.mean

    def test_mean(self):
        x = np.random.default_rng(0).normal(size=(10, 5))
        assert_matches_numpy(stats.mean(cp.asarray(x), axis=0), np.mean(x, axis=0))

    def test_var(self):
        x = np.random.default_rng(1).normal(size=(10, 5))
        assert_matches_numpy(stats.var(cp.asarray(x), axis=0), np.var(x, axis=0))

    def test_std(self):
        x = np.random.default_rng(2).normal(size=(10, 5))
        assert_matches_numpy(stats.std(cp.asarray(x), axis=0), np.std(x, axis=0))

    def test_absolute_error(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.5, 1.5, 1.5])
        assert_matches_numpy(
            stats.absolute_error(cp.asarray(x), cp.asarray(y)), np.abs(x - y)
        )

    def test_relative_error(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 2.0, 2.0])
        assert_matches_numpy(
            stats.relative_error(cp.asarray(x), cp.asarray(y)), np.abs(x / y - 1)
        )

    def test_asarray(self):
        x = cp.asarray([1.0, 2.0])
        assert gpu.iscupy(stats.asarray(x))


@pytest.mark.gpu
@requires_gpu
class TestGpuDigits:
    """End-to-end metric computations on cupy arrays match numpy"""

    eps = 2**-52

    def samples(self, nsamples, shape=()):
        rng = np.random.default_rng(42)
        return 1 + rng.uniform(-1, 1, size=(nsamples,) + shape) * self.eps

    @pytest.mark.parametrize("method", [sd.Method.CNH, sd.Method.General])
    @pytest.mark.parametrize("error", [sd.Error.Absolute, sd.Error.Relative])
    def test_significant_digits_scalar_reference(self, method, error, nsamples):
        x = self.samples(nsamples)
        cpu = sd.significant_digits(x, reference=1.0, method=method, error=error)
        res = sd.significant_digits(
            cp.asarray(x), reference=1.0, method=method, error=error
        )
        assert_matches_numpy(res, cpu)

    @pytest.mark.parametrize("method", [sd.Method.CNH, sd.Method.General])
    @pytest.mark.parametrize("error", [sd.Error.Absolute, sd.Error.Relative])
    def test_contributing_digits_scalar_reference(self, method, error, nsamples):
        x = self.samples(nsamples)
        cpu = sd.contributing_digits(x, reference=1.0, method=method, error=error)
        res = sd.contributing_digits(
            cp.asarray(x), reference=1.0, method=method, error=error
        )
        assert_matches_numpy(res, cpu)

    @pytest.mark.parametrize("method", [sd.Method.CNH, sd.Method.General])
    def test_significant_digits_array_reference(self, method, nsamples):
        x = self.samples(nsamples, shape=(3, 4))
        reference = x.mean(axis=0)
        cpu = sd.significant_digits(x, reference=reference, method=method)
        res = sd.significant_digits(
            cp.asarray(x), reference=cp.asarray(reference), method=method
        )
        assert_matches_numpy(res, cpu)

    @pytest.mark.parametrize("method", [sd.Method.CNH, sd.Method.General])
    def test_significant_digits_no_reference(self, method, nsamples):
        # reference=None splits the sample axis in two, so it must stay even
        x = self.samples(2 * nsamples, shape=(3,))
        cpu = sd.significant_digits(x, reference=None, method=method)
        res = sd.significant_digits(cp.asarray(x), reference=None, method=method)
        assert_matches_numpy(res, cpu)

    def test_significant_digits_numpy_reference_is_promoted(self, nsamples):
        x = self.samples(nsamples)
        cpu = sd.significant_digits(x, reference=np.float64(1.0))
        res = sd.significant_digits(cp.asarray(x), reference=np.float64(1.0))
        assert_matches_numpy(res, cpu)

    def test_change_basis(self, nsamples):
        x = self.samples(nsamples)
        cpu = sd.significant_digits(x, reference=1.0, basis=10)
        res = sd.significant_digits(cp.asarray(x), reference=1.0, basis=10)
        assert_matches_numpy(res, cpu)

    def test_format_uncertainty_returns_host_array(self, nsamples):
        x = self.samples(nsamples)
        cpu = sd.format_uncertainty(x, reference=1.0)
        res = sd.format_uncertainty(cp.asarray(x), reference=1.0)
        assert isinstance(res, np.ndarray)
        assert np.array_equal(res, cpu)

    def test_invalid_0d_cupy_array(self):
        with pytest.raises(TypeError):
            sd.significant_digits(cp.asarray(1.0), reference=1.0)
