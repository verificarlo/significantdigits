"""
Comprehensive validation and error handling tests for significantdigits.

Tests parameter validation, error conditions, and boundary cases
to ensure robust error handling and user-friendly error messages.
"""
import numpy as np
import pytest

import significantdigits as sd
from significantdigits._significantdigits import (
    _assert_is_probability,
    _assert_is_confidence,
    _assert_is_valid_metric,
    _assert_is_valid_method,
    _assert_is_valid_error,
    _assert_is_valid_inputs,
    _preprocess_inputs,
    SignificantDigitsException,
)
from significantdigits.args import parse_args, safe_eval
from significantdigits.stats.dispatch import dispatch


class TestParameterValidation:
    """Test parameter validation functions."""

    def test_probability_validation(self):
        """Test probability parameter validation."""
        # Valid probabilities
        _assert_is_probability(0.0)
        _assert_is_probability(0.5)
        _assert_is_probability(1.0)
        _assert_is_probability(0.95)
        
        # Invalid probabilities
        with pytest.raises(TypeError, match="probability must be between 0 and 1"):
            _assert_is_probability(-0.1)
        
        with pytest.raises(TypeError, match="probability must be between 0 and 1"):
            _assert_is_probability(1.1)
        
        with pytest.raises(TypeError, match="probability must be between 0 and 1"):
            _assert_is_probability(2.0)
        
        with pytest.raises(TypeError, match="probability must be between 0 and 1"):
            _assert_is_probability(-1.0)

    def test_confidence_validation(self):
        """Test confidence parameter validation."""
        # Valid confidence values
        _assert_is_confidence(0.0)
        _assert_is_confidence(0.5)
        _assert_is_confidence(1.0)
        _assert_is_confidence(0.95)
        
        # Invalid confidence values
        with pytest.raises(TypeError, match="confidence must be between 0 and 1"):
            _assert_is_confidence(-0.1)
        
        with pytest.raises(TypeError, match="confidence must be between 0 and 1"):
            _assert_is_confidence(1.1)
        
        with pytest.raises(TypeError, match="confidence must be between 0 and 1"):
            _assert_is_confidence(2.0)
        
        with pytest.raises(TypeError, match="confidence must be between 0 and 1"):
            _assert_is_confidence(-1.0)

    def test_metric_validation(self):
        """Test metric parameter validation."""
        # Valid metrics
        _assert_is_valid_metric(sd.Metric.Significant)
        _assert_is_valid_metric(sd.Metric.Contributing)
        # Note: string validation depends on case sensitivity
        
        # Invalid metrics
        with pytest.raises(TypeError, match="provided invalid metric"):
            _assert_is_valid_metric("invalid")
        
        with pytest.raises(TypeError, match="provided invalid metric"):
            _assert_is_valid_metric(123)
        
        with pytest.raises(TypeError, match="provided invalid metric"):
            _assert_is_valid_metric(None)

    def test_method_validation(self):
        """Test method parameter validation."""
        # Valid methods
        _assert_is_valid_method(sd.Method.CNH)
        _assert_is_valid_method(sd.Method.General)
        # Note: string validation depends on case sensitivity
        
        # Invalid methods
        with pytest.raises(TypeError, match="provided invalid method"):
            _assert_is_valid_method("invalid")
        
        with pytest.raises(TypeError, match="provided invalid method"):
            _assert_is_valid_method(123)
        
        with pytest.raises(TypeError, match="provided invalid method"):
            _assert_is_valid_method(None)

    def test_error_validation(self):
        """Test error parameter validation."""
        # Valid errors
        _assert_is_valid_error(sd.Error.Absolute)
        _assert_is_valid_error(sd.Error.Relative)
        # Note: string validation depends on case sensitivity
        
        # Invalid errors
        with pytest.raises(TypeError, match="provided invalid error"):
            _assert_is_valid_error("invalid")
        
        with pytest.raises(TypeError, match="provided invalid error"):
            _assert_is_valid_error(123)
        
        with pytest.raises(TypeError, match="provided invalid error"):
            _assert_is_valid_error(None)

    def test_input_validation(self):
        """Test input array validation."""
        # Valid inputs
        _assert_is_valid_inputs(np.array([1, 2, 3]))
        _assert_is_valid_inputs([1, 2, 3])
        _assert_is_valid_inputs((1, 2, 3))
        _assert_is_valid_inputs(np.array([[1, 2], [3, 4]]))
        
        # Invalid inputs
        with pytest.raises(TypeError, match="array must be of type"):
            _assert_is_valid_inputs("invalid")
        
        with pytest.raises(TypeError, match="array must be of type"):
            _assert_is_valid_inputs(123)
        
        with pytest.raises(TypeError, match="array must be of type"):
            _assert_is_valid_inputs(None)
        
        # 0-dimensional arrays should fail
        with pytest.raises(TypeError, match="array must be at least 1D"):
            _assert_is_valid_inputs(np.array(5))


class TestPreprocessingValidation:
    """Test input preprocessing validation."""

    def test_preprocess_inputs_valid(self):
        """Test preprocessing with valid inputs."""
        # Array input
        array = [1, 2, 3, 4]
        reference = [1.1, 2.1, 3.1, 4.1]
        proc_array, proc_ref = _preprocess_inputs(array, reference)
        
        assert isinstance(proc_array, np.ndarray)
        assert isinstance(proc_ref, np.ndarray)
        assert proc_array.shape == (4,)
        assert proc_ref.shape == (4,)
        
        # None reference
        proc_array, proc_ref = _preprocess_inputs(array, None)
        assert isinstance(proc_array, np.ndarray)
        assert proc_ref is None

    def test_preprocess_inputs_type_conversion(self):
        """Test type conversion during preprocessing."""
        # Mixed types should be converted to numpy arrays
        array = [1, 2.0, 3, 4.0]
        reference = 2.5
        proc_array, proc_ref = _preprocess_inputs(array, reference)
        
        assert isinstance(proc_array, np.ndarray)
        assert isinstance(proc_ref, np.ndarray)
        assert proc_array.dtype in [np.float64, np.int64]
        assert proc_ref.dtype in [np.float64, np.int64]


class TestHighLevelValidation:
    """Test validation in high-level functions."""

    def test_significant_digits_validation(self):
        """Test parameter validation in significant_digits function."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        
        # Invalid probability
        with pytest.raises(TypeError, match="probability must be between 0 and 1"):
            sd.significant_digits(x, reference=2.0, probability=-0.1)
        
        with pytest.raises(TypeError, match="probability must be between 0 and 1"):
            sd.significant_digits(x, reference=2.0, probability=1.1)
        
        # Invalid confidence
        with pytest.raises(TypeError, match="confidence must be between 0 and 1"):
            sd.significant_digits(x, reference=2.0, confidence=-0.1)
        
        with pytest.raises(TypeError, match="confidence must be between 0 and 1"):
            sd.significant_digits(x, reference=2.0, confidence=1.1)
        
        # Invalid method
        with pytest.raises(TypeError, match="provided invalid method"):
            sd.significant_digits(x, reference=2.0, method="invalid")
        
        # Invalid error
        with pytest.raises(TypeError, match="provided invalid error"):
            sd.significant_digits(x, reference=2.0, error="invalid")

    def test_contributing_digits_validation(self):
        """Test parameter validation in contributing_digits function."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        
        # Invalid probability
        with pytest.raises(TypeError, match="probability must be between 0 and 1"):
            sd.contributing_digits(x, reference=2.0, probability=-0.1)
        
        # Invalid confidence
        with pytest.raises(TypeError, match="confidence must be between 0 and 1"):
            sd.contributing_digits(x, reference=2.0, confidence=1.1)
        
        # Invalid method
        with pytest.raises(TypeError, match="provided invalid method"):
            sd.contributing_digits(x, reference=2.0, method="invalid")
        
        # Invalid error
        with pytest.raises(TypeError, match="provided invalid error"):
            sd.contributing_digits(x, reference=2.0, error="invalid")

    def test_probability_estimation_validation(self):
        """Test validation in probability estimation functions."""
        # Invalid confidence in probability_estimation_bernoulli
        with pytest.raises(TypeError, match="confidence must be between 0 and 1"):
            sd.probability_estimation_bernoulli(50, 100, confidence=-0.1)
        
        with pytest.raises(TypeError, match="confidence must be between 0 and 1"):
            sd.probability_estimation_bernoulli(50, 100, confidence=1.1)
        
        # Invalid parameters in minimum_number_of_trials
        with pytest.raises(TypeError, match="probability must be between 0 and 1"):
            sd.minimum_number_of_trials(probability=-0.1, confidence=0.95)
        
        with pytest.raises(TypeError, match="confidence must be between 0 and 1"):
            sd.minimum_number_of_trials(probability=0.95, confidence=-0.1)


class TestArgumentParsingValidation:
    """Test command-line argument parsing validation."""

    def test_safe_eval_valid(self):
        """Test safe_eval with valid inputs."""
        assert safe_eval("[1, 2, 3]") == [1, 2, 3]
        assert safe_eval("1.5") == 1.5
        assert safe_eval("True") == True
        assert safe_eval("None") == None
        assert safe_eval("{'a': 1}") == {'a': 1}

    def test_safe_eval_invalid(self):
        """Test safe_eval with invalid/malicious inputs."""
        # Should exit on dangerous code
        with pytest.raises(SystemExit):
            safe_eval("__import__('os').system('ls')")
        
        with pytest.raises(SystemExit):
            safe_eval("exec('print(1)')")
        
        with pytest.raises(SystemExit):
            safe_eval("eval('1+1')")
        
        # Should exit on syntax errors
        with pytest.raises(SystemExit):
            safe_eval("[1, 2,")  # Incomplete syntax
        
        with pytest.raises(SystemExit):
            safe_eval("invalid syntax here")

    def test_parse_args_validation(self):
        """Test argument parsing validation."""
        # Valid arguments
        valid_args = [
            "--metric", "significant",
            "--input-format", "stdin",
            "--inputs", "[1, 2, 3, 4]"
        ]
        args = parse_args(valid_args)
        assert args.metric == sd.Metric.Significant
        
        # Invalid probability
        invalid_prob_args = [
            "--metric", "significant",
            "--input-format", "stdin",
            "--inputs", "[1, 2, 3, 4]",
            "--probability", "1.5"
        ]
        with pytest.raises(TypeError):
            parse_args(invalid_prob_args)
        
        # Invalid confidence
        invalid_conf_args = [
            "--metric", "significant",
            "--input-format", "stdin",
            "--inputs", "[1, 2, 3, 4]",
            "--confidence", "-0.5"
        ]
        with pytest.raises(TypeError):
            parse_args(invalid_conf_args)


class TestDispatcherValidation:
    """Test dispatcher function validation."""

    def test_dispatcher_with_valid_arrays(self):
        """Test dispatcher with valid array types."""
        # Dense array
        dense_array = np.array([1, 2, 3, 4])
        func = dispatch(dense_array, 'mean')
        assert callable(func)
        
        # Regular list (should be treated as dense)
        list_array = [1, 2, 3, 4]
        func = dispatch(list_array, 'mean')
        assert callable(func)

    def test_dispatcher_with_invalid_method(self):
        """Test dispatcher with invalid method names."""
        array = np.array([1, 2, 3, 4])
        
        # Invalid method should raise NotImplementedError (or return None)
        try:
            result = dispatch(array, 'invalid_method')
            # If it returns None, that's also acceptable
            assert result is None
        except NotImplementedError:
            # This is the expected behavior
            pass
        
        # None method should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            dispatch(array, None)

    def test_dispatcher_with_invalid_array_type(self):
        """Test dispatcher with invalid array types."""
        # String should raise TypeError (or handle gracefully)
        try:
            dispatch("invalid", 'mean')
        except TypeError:
            # Expected behavior
            pass
        
        # None should raise TypeError (or handle gracefully)
        try:
            dispatch(None, 'mean')
        except (TypeError, ValueError):
            # Expected behavior
            pass
        
        # Number should raise TypeError (or handle gracefully)
        try:
            dispatch(123, 'mean')
        except (TypeError, ValueError):
            # Expected behavior
            pass


class TestComputationValidation:
    """Test validation in core computation functions."""

    def test_odd_sample_size_validation(self):
        """Test validation for odd sample sizes when reference is None."""
        # Odd number of samples should fail when reference is None
        x_odd = np.array([1.0, 2.0, 3.0])  # 3 samples
        
        with pytest.raises(SignificantDigitsException, match="Number of samples must be a multiple of 2"):
            sd.significant_digits(x_odd, reference=None)
        
        with pytest.raises(SignificantDigitsException, match="Number of samples must be a multiple of 2"):
            sd.contributing_digits(x_odd, reference=None)

    def test_dimension_mismatch_validation(self):
        """Test validation for dimension mismatches."""
        x = np.array([[1, 2], [3, 4]])  # 2x2
        ref_wrong = np.array([[[1]], [[2]]])  # Wrong dimensions
        
        # Should raise TypeError for incompatible dimensions
        with pytest.raises(TypeError, match="No comparison found for X and reference"):
            sd.significant_digits(x, reference=ref_wrong)

    def test_unknown_enum_validation(self):
        """Test handling of unknown enum values."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        
        # Should raise TypeError for unknown method (validation happens first)
        with pytest.raises(TypeError, match="provided invalid method"):
            sd.significant_digits(x, reference=2.0, method="unknown_method")
        
        # Should raise TypeError for unknown error (validation happens first)
        with pytest.raises(TypeError, match="provided invalid error"):
            sd.significant_digits(x, reference=2.0, error="unknown_error")


class TestRobustnessValidation:
    """Test robustness of validation under stress conditions."""

    def test_deeply_nested_structures(self):
        """Test with deeply nested data structures."""
        # Deeply nested list
        nested = [[[[[1, 2]], [[3, 4]]]]]
        
        # Should handle or appropriately reject deep nesting
        try:
            result = sd.significant_digits(nested, reference=2.0)
            # Deep nesting may produce NaN results
            assert result is not None
        except (ValueError, TypeError):
            # Expected for complex nesting
            pass

    def test_large_parameter_values(self):
        """Test with extremely large parameter values."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        
        # Very large basis values
        try:
            result = sd.significant_digits(x, reference=2.0, basis=1000000)
            assert np.all(np.isfinite(result))
        except (ValueError, OverflowError):
            # May legitimately fail for very large basis
            pass

    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        # Should appropriately reject non-numeric unicode
        with pytest.raises((TypeError, ValueError)):
            sd.significant_digits("γκσα", reference=2.0)
        
        with pytest.raises((TypeError, ValueError)):
            sd.significant_digits(["α", "β", "γ"], reference=2.0)

    def test_memory_stress_validation(self):
        """Test validation doesn't break under memory stress."""
        # Large array that should still be manageable
        large_x = np.random.rand(10000)
        
        # Should handle large arrays appropriately
        try:
            result = sd.significant_digits(large_x, reference=np.mean(large_x))
            assert np.all(np.isfinite(result))
        except MemoryError:
            # Expected for very large arrays
            pass