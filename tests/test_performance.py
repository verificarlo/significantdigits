"""
Performance regression tests for significantdigits.

These tests ensure that performance optimizations don't introduce
regressions and that the library maintains reasonable performance
characteristics under various conditions.
"""
import time
import numpy as np
import pytest

import significantdigits as sd


class TestPerformanceBaselines:
    """Test performance baselines to detect regressions."""

    @pytest.mark.performance
    def test_significant_digits_performance_baseline(self):
        """Test significant digits computation performance baseline."""
        # Generate test data
        np.random.seed(42)
        x = np.random.normal(1.0, 0.1, 1000)
        ref = np.mean(x)
        
        # Measure time for CNH method
        start_time = time.time()
        result_cnh = sd.significant_digits(x, reference=ref, method=sd.Method.CNH)
        cnh_time = time.time() - start_time
        
        # Measure time for General method
        start_time = time.time()
        result_gen = sd.significant_digits(x, reference=ref, method=sd.Method.General)
        gen_time = time.time() - start_time
        
        # Performance assertions (adjust thresholds as needed)
        assert cnh_time < 1.0, f"CNH method too slow: {cnh_time:.3f}s"
        assert gen_time < 5.0, f"General method too slow: {gen_time:.3f}s"
        
        # Results should be finite
        assert np.all(np.isfinite(result_cnh))
        assert np.all(np.isfinite(result_gen))

    @pytest.mark.performance
    def test_contributing_digits_performance_baseline(self):
        """Test contributing digits computation performance baseline."""
        np.random.seed(42)
        x = np.random.normal(1.0, 0.1, 1000)
        ref = np.mean(x)
        
        # Measure time for CNH method
        start_time = time.time()
        result_cnh = sd.contributing_digits(x, reference=ref, method=sd.Method.CNH)
        cnh_time = time.time() - start_time
        
        # Measure time for General method
        start_time = time.time()
        result_gen = sd.contributing_digits(x, reference=ref, method=sd.Method.General)
        gen_time = time.time() - start_time
        
        # Performance assertions
        assert cnh_time < 1.0, f"CNH method too slow: {cnh_time:.3f}s"
        assert gen_time < 5.0, f"General method too slow: {gen_time:.3f}s"
        
        # Results should be finite
        assert np.all(np.isfinite(result_cnh))
        assert np.all(np.isfinite(result_gen))

    @pytest.mark.performance
    def test_format_uncertainty_performance_baseline(self):
        """Test format uncertainty performance baseline."""
        np.random.seed(42)
        x = np.random.normal(1.0, 0.1, 100)  # Smaller for formatting
        ref = np.mean(x)
        
        # Measure time for formatting
        start_time = time.time()
        result = sd.format_uncertainty(x, reference=ref)
        format_time = time.time() - start_time
        
        # Measure time for tuple output
        start_time = time.time()
        value, error = sd.format_uncertainty(x, reference=ref, as_tuple=True)
        tuple_time = time.time() - start_time
        
        # Performance assertions
        assert format_time < 2.0, f"Formatting too slow: {format_time:.3f}s"
        assert tuple_time < 2.0, f"Tuple formatting too slow: {tuple_time:.3f}s"
        
        # Results should be valid
        assert result.shape == x.shape
        assert value.shape == x.shape
        assert error.shape == x.shape


class TestScalabilityPerformance:
    """Test performance scalability with different data sizes."""

    @pytest.mark.performance
    def test_array_size_scalability(self):
        """Test performance scaling with array size."""
        sizes = [100, 500, 1000, 2000]
        times = []
        
        for size in sizes:
            np.random.seed(42)
            x = np.random.normal(1.0, 0.1, size)
            ref = np.mean(x)
            
            start_time = time.time()
            result = sd.significant_digits(x, reference=ref, method=sd.Method.CNH)
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            # Should complete in reasonable time
            assert elapsed < 5.0, f"Size {size} too slow: {elapsed:.3f}s"
            assert np.all(np.isfinite(result))
        
        # Performance should scale reasonably (not exponentially)
        # Allow for some variation but check general trend
        for i in range(1, len(times)):
            scale_factor = sizes[i] / sizes[i-1]
            time_factor = times[i] / times[i-1]
            # Time factor should not be much worse than scale factor squared
            assert time_factor < scale_factor ** 2.5, \
                f"Performance scaling too poor: {time_factor:.2f}x for {scale_factor:.2f}x data"

    @pytest.mark.performance
    def test_multidimensional_scalability(self):
        """Test performance with multidimensional arrays."""
        shapes = [(100,), (10, 10), (5, 5, 4), (2, 5, 5, 2)]
        
        for shape in shapes:
            np.random.seed(42)
            x = np.random.normal(1.0, 0.1, shape)
            ref = np.mean(x, axis=0)
            
            start_time = time.time()
            result = sd.significant_digits(x, reference=ref, axis=0)
            elapsed = time.time() - start_time
            
            # Should handle multidimensional arrays efficiently
            assert elapsed < 3.0, f"Shape {shape} too slow: {elapsed:.3f}s"
            assert np.all(np.isfinite(result))
            assert result.shape == shape[1:]

    @pytest.mark.performance
    def test_high_precision_performance(self):
        """Test performance with high precision requirements."""
        np.random.seed(42)
        x = np.random.normal(1.0, 1e-10, 1000)  # Very small variance
        ref = np.mean(x)
        
        # High precision requirements
        start_time = time.time()
        result = sd.significant_digits(
            x, reference=ref, 
            probability=0.999, confidence=0.999,
            method=sd.Method.CNH
        )
        elapsed = time.time() - start_time
        
        assert elapsed < 2.0, f"High precision too slow: {elapsed:.3f}s"
        assert np.all(np.isfinite(result))


class TestMemoryPerformance:
    """Test memory usage and efficiency."""

    @pytest.mark.performance
    def test_memory_efficiency_large_arrays(self):
        """Test memory efficiency with large arrays."""
        # This test checks that we don't have excessive memory usage
        np.random.seed(42)
        
        # Use a reasonably large array
        size = 10000
        x = np.random.normal(1.0, 0.1, size)
        ref = np.mean(x)
        
        # These should complete without memory issues
        result1 = sd.significant_digits(x, reference=ref)
        assert np.all(np.isfinite(result1))
        
        result2 = sd.contributing_digits(x, reference=ref)
        assert np.all(np.isfinite(result2))
        
        # Format uncertainty should work efficiently
        formatted = sd.format_uncertainty(x[:100], reference=ref)  # Smaller for formatting
        assert formatted.shape == (100,)

    @pytest.mark.performance
    def test_memory_efficiency_basis_conversion(self):
        """Test memory efficiency in basis conversion."""
        np.random.seed(42)
        x = np.random.normal(1.0, 0.1, 1000)
        ref = np.mean(x)
        
        # Test different bases - should not consume excessive memory
        for basis in [2, 8, 10, 16]:
            result = sd.significant_digits(x, reference=ref, basis=basis)
            assert np.all(np.isfinite(result))
            assert result.shape == ()  # Scalar for 1D input


class TestOptimizationRegressions:
    """Test that optimizations don't introduce regressions."""

    @pytest.mark.performance
    def test_power_operation_optimization(self):
        """Test that power operation optimizations work correctly."""
        np.random.seed(42)
        x = np.random.normal(1.0, 0.1, 100)
        ref = np.mean(x)
        
        # Test that optimized power operations (np.exp2) work
        start_time = time.time()
        result_general = sd.significant_digits(x, reference=ref, method=sd.Method.General)
        general_time = time.time() - start_time
        
        # Should be reasonably fast and produce valid results
        assert general_time < 2.0, f"General method with power optimizations too slow: {general_time:.3f}s"
        assert np.all(np.isfinite(result_general))

    @pytest.mark.performance
    def test_bitwise_operation_optimization(self):
        """Test that bitwise operation optimizations work correctly."""
        np.random.seed(42)
        x = np.random.normal(1.0, 0.1, 100)
        ref = np.mean(x)
        
        # Test contributing digits which use bitwise operations
        start_time = time.time()
        result = sd.contributing_digits(x, reference=ref, method=sd.Method.General)
        elapsed = time.time() - start_time
        
        # Should be fast with bitwise optimizations
        assert elapsed < 2.0, f"Contributing digits with bitwise optimizations too slow: {elapsed:.3f}s"
        assert np.all(np.isfinite(result))

    @pytest.mark.performance
    def test_array_operation_optimization(self):
        """Test that array operation optimizations work correctly."""
        np.random.seed(42)
        
        # Test with multidimensional arrays (uses optimized broadcasting)
        x = np.random.normal(1.0, 0.1, (100, 10))
        ref = np.mean(x, axis=0)
        
        start_time = time.time()
        result = sd.significant_digits(x, reference=ref, axis=0)
        elapsed = time.time() - start_time
        
        # Should be fast with optimized array operations
        assert elapsed < 2.0, f"Array operations too slow: {elapsed:.3f}s"
        assert np.all(np.isfinite(result))
        assert result.shape == (10,)

    @pytest.mark.performance
    def test_memory_allocation_optimization(self):
        """Test that memory allocation optimizations work correctly."""
        np.random.seed(42)
        x = np.random.normal(1.0, 0.1, 100)
        ref = np.mean(x)
        
        # Test tuple output (optimized memory allocation)
        start_time = time.time()
        value, error = sd.format_uncertainty(x, reference=ref, as_tuple=True)
        tuple_time = time.time() - start_time
        
        # Test single output (optimized memory allocation)
        start_time = time.time()
        formatted = sd.format_uncertainty(x, reference=ref, as_tuple=False)
        single_time = time.time() - start_time
        
        # Both should be reasonably fast
        assert tuple_time < 1.0, f"Tuple formatting with memory optimization too slow: {tuple_time:.3f}s"
        assert single_time < 1.0, f"Single formatting with memory optimization too slow: {single_time:.3f}s"
        
        # Results should be valid
        assert value.shape == x.shape
        assert error.shape == x.shape
        assert formatted.shape == x.shape


class TestConcurrencyPerformance:
    """Test performance under concurrent usage patterns."""

    @pytest.mark.performance
    def test_repeated_computation_performance(self):
        """Test performance of repeated computations."""
        np.random.seed(42)
        x = np.random.normal(1.0, 0.1, 500)
        ref = np.mean(x)
        
        # Time multiple repeated computations
        start_time = time.time()
        results = []
        for _ in range(10):
            result = sd.significant_digits(x, reference=ref)
            results.append(result)
        total_time = time.time() - start_time
        
        # Should handle repeated computations efficiently
        assert total_time < 5.0, f"Repeated computations too slow: {total_time:.3f}s"
        
        # All results should be identical and finite
        for result in results:
            assert np.all(np.isfinite(result))
            assert np.allclose(result, results[0])

    @pytest.mark.performance
    def test_different_parameter_performance(self):
        """Test performance with different parameter combinations."""
        np.random.seed(42)
        x = np.random.normal(1.0, 0.1, 200)
        ref = np.mean(x)
        
        parameter_combinations = [
            {"method": sd.Method.CNH, "error": sd.Error.Absolute},
            {"method": sd.Method.CNH, "error": sd.Error.Relative},
            {"method": sd.Method.General, "error": sd.Error.Absolute},
            {"method": sd.Method.General, "error": sd.Error.Relative},
        ]
        
        for params in parameter_combinations:
            start_time = time.time()
            result = sd.significant_digits(x, reference=ref, **params)
            elapsed = time.time() - start_time
            
            # Each combination should be reasonably fast
            assert elapsed < 3.0, f"Parameters {params} too slow: {elapsed:.3f}s"
            assert np.all(np.isfinite(result))


@pytest.mark.performance
class TestBenchmarkSuite:
    """Comprehensive benchmark suite for performance tracking."""

    def test_comprehensive_benchmark(self):
        """Comprehensive benchmark across multiple scenarios."""
        scenarios = [
            {"name": "Small_CNH", "size": 100, "method": sd.Method.CNH},
            {"name": "Medium_CNH", "size": 1000, "method": sd.Method.CNH},
            {"name": "Small_General", "size": 100, "method": sd.Method.General},
            {"name": "Medium_General", "size": 500, "method": sd.Method.General},  # Smaller for General
        ]
        
        benchmark_results = {}
        
        for scenario in scenarios:
            np.random.seed(42)
            x = np.random.normal(1.0, 0.1, scenario["size"])
            ref = np.mean(x)
            
            # Benchmark significant digits
            start_time = time.time()
            sig_result = sd.significant_digits(x, reference=ref, method=scenario["method"])
            sig_time = time.time() - start_time
            
            # Benchmark contributing digits
            start_time = time.time()
            con_result = sd.contributing_digits(x, reference=ref, method=scenario["method"])
            con_time = time.time() - start_time
            
            benchmark_results[scenario["name"]] = {
                "significant_time": sig_time,
                "contributing_time": con_time,
                "size": scenario["size"]
            }
            
            # Verify results are valid
            assert np.all(np.isfinite(sig_result))
            assert np.all(np.isfinite(con_result))
        
        # Print benchmark results for tracking
        print("\nPerformance Benchmark Results:")
        print("-" * 50)
        for name, results in benchmark_results.items():
            print(f"{name}:")
            print(f"  Size: {results['size']}")
            print(f"  Significant digits: {results['significant_time']:.4f}s")
            print(f"  Contributing digits: {results['contributing_time']:.4f}s")
            print(f"  Rate (samples/sec): {results['size']/max(results['significant_time'], 0.001):.0f}")
        
        # Assert reasonable performance across all scenarios
        for name, results in benchmark_results.items():
            assert results["significant_time"] < 10.0, f"{name} significant digits too slow"
            assert results["contributing_time"] < 10.0, f"{name} contributing digits too slow"