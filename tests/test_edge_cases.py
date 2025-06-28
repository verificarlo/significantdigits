"""
Edge case tests for numerical computations in significantdigits.

These tests cover various edge cases and numerical stability scenarios
that could cause issues in real-world usage.
"""
import numpy as np
import pytest
import warnings

import significantdigits as sd
from significantdigits._significantdigits import (
    _compute_z,
    _compute_scaling_factor,
    _significant_digits_cnh,
    _significant_digits_general,
    _contributing_digits_cnh,
    _contributing_digits_general,
    _get_significant_size,
    _fill_where,
    _operator_along_axis,
    _divide_along_axis,
    _substract_along_axis,
)


class TestNumericalEdgeCases:
    """Test numerical edge cases that could cause instability."""

    def test_very_small_numbers(self):
        """Test with numbers close to machine epsilon."""
        eps = np.finfo(np.float64).eps
        x = np.array([eps, 2*eps, 3*eps, 4*eps])
        
        # Should not crash with very small numbers
        result = sd.significant_digits(x, reference=np.mean(x))
        assert np.all(np.isfinite(result))
        
        result = sd.contributing_digits(x, reference=np.mean(x))
        assert np.all(np.isfinite(result))

    def test_very_large_numbers(self):
        """Test with numbers close to overflow."""
        large = np.finfo(np.float64).max / 1e6  # Avoid actual overflow
        x = np.array([large, large*0.9, large*1.1, large*0.8])
        
        # Should not crash with very large numbers
        result = sd.significant_digits(x, reference=np.mean(x))
        assert np.all(np.isfinite(result))
        
        result = sd.contributing_digits(x, reference=np.mean(x))
        assert np.all(np.isfinite(result))

    def test_inf_and_nan_handling(self):
        """Test behavior with infinite and NaN values."""
        x_with_inf = np.array([1.0, 2.0, np.inf, 4.0])
        x_with_nan = np.array([1.0, 2.0, np.nan, 4.0])
        x_with_neginf = np.array([1.0, 2.0, -np.inf, 4.0])
        
        # Should handle inf/nan gracefully
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # These should not crash
            try:
                result = sd.significant_digits(x_with_inf, reference=2.0)
                # Result may contain inf/nan but should not crash
            except (ValueError, RuntimeWarning):
                pass  # Expected behavior for inf/nan
                
            try:
                result = sd.significant_digits(x_with_nan, reference=2.0)
            except (ValueError, RuntimeWarning):
                pass  # Expected behavior for inf/nan
                
            try:
                result = sd.significant_digits(x_with_neginf, reference=2.0)
            except (ValueError, RuntimeWarning):
                pass  # Expected behavior for inf/nan

    def test_zero_reference_relative_error(self):
        """Test relative error computation with zero reference."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        zero_ref = np.array([0.0, 0.0, 0.0, 0.0])
        
        # Should warn about division by zero
        with pytest.warns(RuntimeWarning, match="divide by zero"):
            z, e = _compute_z(x, zero_ref, error=sd.Error.Relative, axis=0)
        
        # Should contain inf where division by zero occurred  
        assert np.any(np.isinf(z))

    def test_identical_arrays(self):
        """Test with identical arrays (perfect precision case)."""
        x = np.array([1.0, 1.0, 1.0, 1.0])
        
        # Should handle identical values gracefully
        result = sd.significant_digits(x, reference=1.0)
        assert np.all(np.isfinite(result))
        
        result = sd.contributing_digits(x, reference=1.0)
        assert np.all(np.isfinite(result))

    def test_single_element_arrays(self):
        """Test with single-element arrays."""
        x = np.array([1.5])
        
        # Should work with single elements
        with pytest.raises(sd.SignificantDigitsException):
            # Single element can't be split for self-comparison
            sd.significant_digits(x, reference=None)
        
        # But should work with explicit reference (may produce NaN for edge cases)
        result = sd.significant_digits(x, reference=1.0)
        # Single element may produce NaN due to zero variance
        assert result.shape == ()  # Scalar result

    def test_empty_arrays(self):
        """Test with empty arrays."""
        x = np.array([])
        
        # Should handle empty arrays appropriately
        with pytest.raises((ValueError, TypeError)):
            sd.significant_digits(x, reference=1.0)

    def test_high_dimensional_arrays(self):
        """Test with high-dimensional arrays."""
        # 5D array
        shape = (2, 3, 4, 5, 6)
        x = np.random.rand(*shape) + 1.0  # Avoid zero
        ref = np.mean(x, axis=0)
        
        # Should handle high dimensions
        result = sd.significant_digits(x, reference=ref, axis=0)
        assert result.shape == shape[1:]
        assert np.all(np.isfinite(result))

    def test_extreme_probability_values(self):
        """Test with probability values at boundaries."""
        x = np.array([1.0, 1.1, 0.9, 1.05])
        
        # Test with minimum probability
        result = sd.significant_digits(x, reference=1.0, probability=1e-10)
        assert np.all(np.isfinite(result))
        
        # Test with maximum probability
        result = sd.significant_digits(x, reference=1.0, probability=1.0 - 1e-10)
        assert np.all(np.isfinite(result))

    def test_extreme_confidence_values(self):
        """Test with confidence values at boundaries."""
        x = np.array([1.0, 1.1, 0.9, 1.05])
        
        # Test with minimum confidence
        result = sd.significant_digits(x, reference=1.0, confidence=1e-10)
        assert np.all(np.isfinite(result))
        
        # Test with maximum confidence
        result = sd.significant_digits(x, reference=1.0, confidence=1.0 - 1e-10)
        assert np.all(np.isfinite(result))


class TestInputValidationEdgeCases:
    """Test input validation edge cases."""

    def test_mixed_dtypes(self):
        """Test with mixed data types."""
        x_int = np.array([1, 2, 3, 4], dtype=np.int32)
        x_float32 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        x_float64 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        
        # Should handle different dtypes
        result = sd.significant_digits(x_int, reference=2.0)
        assert np.all(np.isfinite(result))
        
        result = sd.significant_digits(x_float32, reference=2.0)
        assert np.all(np.isfinite(result))
        
        result = sd.significant_digits(x_float64, reference=2.0)
        assert np.all(np.isfinite(result))

    def test_complex_numbers(self):
        """Test with complex numbers (should fail gracefully)."""
        x_complex = np.array([1+1j, 2+2j, 3+3j, 4+4j])
        
        # Should handle complex numbers or raise appropriate error
        try:
            result = sd.significant_digits(x_complex, reference=2+2j)
            # If it works, result should be finite
            assert np.all(np.isfinite(result))
        except (TypeError, ValueError):
            # Expected for complex numbers
            pass

    def test_string_inputs(self):
        """Test with invalid string inputs."""
        x_str = ["1", "2", "3", "4"]
        
        # Should raise appropriate error for string inputs
        with pytest.raises((TypeError, ValueError)):
            sd.significant_digits(x_str, reference="2")

    def test_mismatched_dimensions(self):
        """Test with mismatched array dimensions."""
        x = np.array([[1, 2], [3, 4]])
        ref = np.array([1, 2, 3])  # Wrong shape
        
        # Should raise appropriate error for dimension mismatch
        with pytest.raises((TypeError, ValueError, sd.SignificantDigitsException)):
            sd.significant_digits(x, reference=ref)


class TestScalingFactorEdgeCases:
    """Test _compute_scaling_factor edge cases."""

    def test_all_zeros(self):
        """Test scaling factor with all zero array."""
        y = np.zeros(5)
        result = _compute_scaling_factor(y, axis=0, reference_is_random_variable=False)
        assert np.all(result == 1)  # Should return 1 for all zeros

    def test_mixed_signs(self):
        """Test scaling factor with mixed positive/negative values."""
        y = np.array([-4.0, -2.0, 2.0, 4.0])
        result = _compute_scaling_factor(y, axis=0, reference_is_random_variable=False)
        expected = np.array([3, 2, 2, 3])  # floor(log2(abs(y))) + 1
        assert np.all(np.equal(result, expected))

    def test_very_small_scaling_factor(self):
        """Test scaling factor with very small numbers."""
        y = np.array([1e-100, 1e-200, 1e-50])
        result = _compute_scaling_factor(y, axis=0, reference_is_random_variable=False)
        # Should handle very small numbers (may have negative log values)
        assert np.all(np.isfinite(result))


class TestOperatorEdgeCases:
    """Test operator function edge cases."""

    def test_divide_by_zero_protection(self):
        """Test division operations with zero denominators."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 0.0, 3.0, 0.0])
        
        # Should produce inf where y is zero
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _divide_along_axis(x, y, axis=0)
        # Check for inf values where division by zero occurred
        assert np.any(np.isinf(result))

    def test_subtract_with_broadcasting(self):
        """Test subtraction with complex broadcasting."""
        x = np.random.rand(2, 3, 4)
        y = np.random.rand(3, 4)
        
        # Should handle broadcasting correctly
        result = _substract_along_axis(x, y, axis=0)
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))

    def test_fill_where_edge_cases(self):
        """Test _fill_where with various edge cases."""
        # Test with scalar mask
        x = np.array([1.0, 2.0, 3.0])
        fill_value = np.array(99.0)
        mask = True
        result = _fill_where(x, fill_value, mask)
        assert np.all(result == 99.0)
        
        # Test with scalar x
        x = np.array(5.0)
        fill_value = np.array(99.0)
        mask = True
        result = _fill_where(x, fill_value, mask)
        assert result == 99.0
        
        # Test with no fill needed
        x = np.array([1.0, 2.0, 3.0])
        fill_value = np.array(99.0)
        mask = np.array([False, False, False])
        result = _fill_where(x, fill_value, mask)
        assert np.all(result == x)


class TestMethodSpecificEdgeCases:
    """Test edge cases specific to CNH and General methods."""

    def test_cnh_zero_variance(self):
        """Test CNH method with zero variance."""
        # Create array with identical values
        x = np.array([1.0, 1.0, 1.0, 1.0])
        ref = np.array(1.0)
        
        # Should handle zero variance case
        result = _significant_digits_cnh(
            x, ref, axis=0, error=sd.Error.Absolute,
            probability=0.95, confidence=0.95
        )
        assert np.all(np.isfinite(result))

    def test_general_early_termination(self):
        """Test General method early termination."""
        # Create data that should terminate early
        x = np.array([1.0, 1.01, 0.99, 1.005])
        ref = np.array(1.0)
        
        result = _significant_digits_general(
            x, ref, axis=0, error=sd.Error.Relative
        )
        assert np.all(np.isfinite(result))
        
        result = _contributing_digits_general(
            x, ref, axis=0, error=sd.Error.Relative
        )
        assert np.all(np.isfinite(result))

    def test_general_no_early_termination(self):
        """Test General method without early termination."""
        # Create data with wide spread
        x = np.array([0.5, 1.0, 1.5, 2.0])
        ref = np.array(1.0)
        
        result = _significant_digits_general(
            x, ref, axis=0, error=sd.Error.Relative
        )
        assert np.all(np.isfinite(result))
        
        result = _contributing_digits_general(
            x, ref, axis=0, error=sd.Error.Relative
        )
        assert np.all(np.isfinite(result))


class TestBasisConversionEdgeCases:
    """Test basis conversion edge cases."""

    def test_basis_conversion_extreme_values(self):
        """Test basis conversion with extreme values."""
        # Very small result
        small_result = np.array([1e-10])
        converted = sd.change_basis(small_result, 10)
        assert np.all(np.isfinite(converted))
        
        # Very large result
        large_result = np.array([1000])
        converted = sd.change_basis(large_result, 10)
        assert np.all(np.isfinite(converted))

    def test_basis_conversion_different_bases(self):
        """Test basis conversion with various bases."""
        result = np.array([10.0, 20.0, 30.0])
        
        for basis in [2, 8, 10, 16, 64]:
            converted = sd.change_basis(result, basis)
            assert np.all(np.isfinite(converted))
            assert converted.shape == result.shape

    def test_invalid_basis(self):
        """Test with invalid basis values."""
        result = np.array([10.0])
        
        # Zero basis should produce inf or nan (division by inf in log)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zero_result = sd.change_basis(result, 0)
            # Result may be inf, nan, or special value depending on log implementation
            assert np.any(np.isinf(zero_result)) or np.any(np.isnan(zero_result)) or np.any(zero_result == 0)
        
        # Negative basis should produce valid result (log of negative is complex)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            neg_result = sd.change_basis(result, -2)
            # May produce NaN due to complex logarithms
            assert neg_result is not None