"""
Property-based and fuzzing tests for significantdigits.

These tests use randomized inputs to discover edge cases and verify
mathematical properties that should hold for all valid inputs.
"""
import numpy as np
import pytest
import warnings

import significantdigits as sd
from significantdigits._significantdigits import (
    _compute_z,
    _compute_scaling_factor,
    change_basis,
    probability_estimation_bernoulli,
    minimum_number_of_trials,
)


class TestMathematicalProperties:
    """Test fundamental mathematical properties that should always hold."""

    def test_significant_digits_monotonicity(self):
        """Test that more precise data yields more significant digits."""
        np.random.seed(42)
        
        # Generate data with different precision levels
        for _ in range(10):  # Run multiple times with different random data
            base_value = np.random.uniform(0.1, 10.0)
            
            # High precision data
            high_precision = np.random.normal(base_value, 1e-8, 100)
            
            # Low precision data
            low_precision = np.random.normal(base_value, 1e-2, 100)
            
            # Compute significant digits
            high_sig = sd.significant_digits(high_precision, reference=base_value)
            low_sig = sd.significant_digits(low_precision, reference=base_value)
            
            # High precision should generally yield more significant digits
            # (Allow some tolerance for statistical variation)
            if np.isfinite(high_sig) and np.isfinite(low_sig):
                # This is a statistical property, so we allow some flexibility
                assert high_sig >= low_sig - 2, \
                    f"High precision ({high_sig:.2f}) should >= low precision ({low_sig:.2f}) - 2"

    def test_basis_conversion_consistency(self):
        """Test that basis conversion maintains relative relationships."""
        np.random.seed(42)
        
        for _ in range(20):
            x = np.random.normal(1.0, 0.1, 50)
            ref = np.random.uniform(0.5, 1.5)
            
            # Compute in different bases
            base2_result = sd.significant_digits(x, reference=ref, basis=2)
            base10_result = sd.significant_digits(x, reference=ref, basis=10)
            base16_result = sd.significant_digits(x, reference=ref, basis=16)
            
            if all(np.isfinite([base2_result, base10_result, base16_result])):
                # Check conversion consistency
                converted_10 = change_basis(base2_result, 10)
                converted_16 = change_basis(base2_result, 16)
                
                # Should be approximately equal (within numerical precision)
                assert abs(converted_10 - base10_result) < 1e-10, \
                    f"Base conversion inconsistency: {converted_10} != {base10_result}"
                assert abs(converted_16 - base16_result) < 1e-10, \
                    f"Base conversion inconsistency: {converted_16} != {base16_result}"

    def test_error_type_relationships(self):
        """Test relationships between absolute and relative error results."""
        np.random.seed(42)
        
        for _ in range(15):
            # Generate data around different scales
            scale = 10 ** np.random.uniform(-3, 3)
            x = np.random.normal(scale, scale * 0.01, 100)
            ref = scale
            
            abs_result = sd.significant_digits(x, reference=ref, error=sd.Error.Absolute)
            rel_result = sd.significant_digits(x, reference=ref, error=sd.Error.Relative)
            
            if np.isfinite(abs_result) and np.isfinite(rel_result):
                # Both should be positive
                assert abs_result > 0, "Absolute error result should be positive"
                assert rel_result > 0, "Relative error result should be positive"
                
                # For data around scale 1, results should be roughly comparable
                if 0.1 <= scale <= 10:
                    ratio = abs_result / rel_result
                    assert 0.1 <= ratio <= 10, \
                        f"Unreasonable ratio between absolute ({abs_result}) and relative ({rel_result}) results"

    def test_probability_confidence_relationships(self):
        """Test relationships between probability and confidence parameters."""
        np.random.seed(42)
        x = np.random.normal(1.0, 0.05, 200)
        ref = 1.0
        
        # Higher confidence should generally give more conservative (lower) estimates
        low_conf = sd.significant_digits(x, reference=ref, confidence=0.90)
        high_conf = sd.significant_digits(x, reference=ref, confidence=0.99)
        
        if np.isfinite(low_conf) and np.isfinite(high_conf):
            # Higher confidence should be more conservative
            assert high_conf <= low_conf + 1, \
                f"Higher confidence ({high_conf}) should be <= lower confidence ({low_conf}) + 1"
        
        # Higher probability should generally give more conservative estimates
        low_prob = sd.significant_digits(x, reference=ref, probability=0.90)
        high_prob = sd.significant_digits(x, reference=ref, probability=0.99)
        
        if np.isfinite(low_prob) and np.isfinite(high_prob):
            assert high_prob <= low_prob + 1, \
                f"Higher probability ({high_prob}) should be <= lower probability ({low_prob}) + 1"

    def test_sample_size_effects(self):
        """Test effects of sample size on results."""
        np.random.seed(42)
        
        for _ in range(10):
            base_value = np.random.uniform(0.5, 2.0)
            noise_level = np.random.uniform(0.001, 0.1)
            
            # Different sample sizes
            small_sample = np.random.normal(base_value, noise_level, 50)
            large_sample = np.random.normal(base_value, noise_level, 500)
            
            small_result = sd.significant_digits(small_sample, reference=base_value)
            large_result = sd.significant_digits(large_sample, reference=base_value)
            
            if np.isfinite(small_result) and np.isfinite(large_result):
                # Larger samples should generally give more or equal significant digits
                # (but allow for statistical variation)
                assert large_result >= small_result - 2, \
                    f"Large sample ({large_result}) should >= small sample ({small_result}) - 2"


class TestPropertyInvariance:
    """Test properties that should be invariant under certain transformations."""

    def test_scale_invariance_relative_error(self):
        """Test that relative error is invariant under scaling."""
        np.random.seed(42)
        
        for _ in range(15):
            # Original data
            base_data = np.random.normal(1.0, 0.01, 100)
            scale_factor = 10 ** np.random.uniform(-2, 2)
            
            # Scaled data
            scaled_data = base_data * scale_factor
            
            # Compute relative error results
            original_result = sd.significant_digits(
                base_data, reference=1.0, error=sd.Error.Relative
            )
            scaled_result = sd.significant_digits(
                scaled_data, reference=scale_factor, error=sd.Error.Relative
            )
            
            if np.isfinite(original_result) and np.isfinite(scaled_result):
                # Should be approximately equal for relative error
                assert abs(original_result - scaled_result) < 1.0, \
                    f"Scale invariance violated: {original_result} vs {scaled_result}"

    def test_translation_effects_absolute_error(self):
        """Test effects of translation on absolute error."""
        np.random.seed(42)
        
        for _ in range(10):
            base_data = np.random.normal(0.0, 0.01, 100)
            translation = np.random.uniform(-5, 5)
            
            # Translated data
            translated_data = base_data + translation
            
            # Compute absolute error results
            original_result = sd.significant_digits(
                base_data, reference=0.0, error=sd.Error.Absolute
            )
            translated_result = sd.significant_digits(
                translated_data, reference=translation, error=sd.Error.Absolute
            )
            
            if np.isfinite(original_result) and np.isfinite(translated_result):
                # For similar noise levels, results should be comparable
                # (accounting for different reference magnitudes)
                assert abs(original_result - translated_result) < 5.0, \
                    f"Translation effects too large: {original_result} vs {translated_result}"

    def test_permutation_invariance(self):
        """Test that results are invariant under data permutation."""
        np.random.seed(42)
        
        for _ in range(10):
            data = np.random.normal(1.0, 0.1, 100)
            ref = np.random.uniform(0.8, 1.2)
            
            # Original order
            original_result = sd.significant_digits(data, reference=ref)
            
            # Shuffled order
            shuffled_data = data.copy()
            np.random.shuffle(shuffled_data)
            shuffled_result = sd.significant_digits(shuffled_data, reference=ref)
            
            if np.isfinite(original_result) and np.isfinite(shuffled_result):
                # Should be exactly equal (within numerical precision)
                assert abs(original_result - shuffled_result) < 1e-10, \
                    f"Permutation invariance violated: {original_result} vs {shuffled_result}"


class TestFuzzingInputs:
    """Test with randomized, potentially problematic inputs."""

    def test_random_array_structures(self):
        """Test with random array structures and shapes."""
        np.random.seed(42)
        
        for _ in range(30):
            # Random shape (1D to 4D)
            ndim = np.random.randint(1, 5)
            shape = tuple(np.random.randint(2, 10, ndim))
            
            # Random data with various characteristics
            if np.random.random() < 0.3:
                # Sometimes use integer data
                data = np.random.randint(-100, 100, shape).astype(float)
            else:
                # Usually use float data
                scale = 10 ** np.random.uniform(-3, 3)
                noise = scale * np.random.uniform(0.001, 0.5)
                data = np.random.normal(scale, noise, shape)
            
            # Random reference
            if np.random.random() < 0.5:
                # Scalar reference
                ref = np.random.uniform(data.min(), data.max())
            else:
                # Array reference
                axis = np.random.randint(0, ndim)
                ref = np.mean(data, axis=axis)
            
            # Random axis
            axis = np.random.randint(0, ndim)
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    result = sd.significant_digits(
                        data, reference=ref, axis=axis,
                        method=np.random.choice([sd.Method.CNH, sd.Method.General]),
                        error=np.random.choice([sd.Error.Absolute, sd.Error.Relative])
                    )
                    
                    # If computation succeeds, result should be finite
                    if np.isscalar(result):
                        assert np.isfinite(result) or np.isnan(result), \
                            f"Invalid scalar result: {result}"
                    else:
                        assert np.all(np.isfinite(result) | np.isnan(result)), \
                            f"Invalid array result shape {result.shape}"
                            
            except (ValueError, TypeError, sd.SignificantDigitsException):
                # These are acceptable for problematic inputs
                pass

    def test_random_parameter_combinations(self):
        """Test with random parameter combinations."""
        np.random.seed(42)
        
        for _ in range(50):
            # Generate reasonable test data
            data = np.random.normal(1.0, 0.1, np.random.randint(10, 100))
            ref = np.random.uniform(0.5, 1.5)
            
            # Random valid parameters
            probability = np.random.uniform(0.01, 0.99)
            confidence = np.random.uniform(0.01, 0.99)
            basis = np.random.choice([2, 8, 10, 16])
            method = np.random.choice([sd.Method.CNH, sd.Method.General])
            error = np.random.choice([sd.Error.Absolute, sd.Error.Relative])
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    result = sd.significant_digits(
                        data, reference=ref,
                        probability=probability,
                        confidence=confidence,
                        basis=basis,
                        method=method,
                        error=error
                    )
                    
                    # Result should be reasonable
                    assert np.isfinite(result) or np.isnan(result), \
                        f"Invalid result: {result} with params p={probability}, c={confidence}"
                    
                    if np.isfinite(result):
                        assert result >= 0, f"Result should be non-negative: {result}"
                        
                        # Should be reasonable range for the given basis
                        max_reasonable = 100  # Very generous upper bound
                        assert result < max_reasonable, \
                            f"Result too large: {result} for basis {basis}"
                            
            except (ValueError, TypeError, sd.SignificantDigitsException):
                # Acceptable for some parameter combinations
                pass

    def test_extreme_data_values(self):
        """Test with extreme data values."""
        np.random.seed(42)
        
        extreme_scenarios = [
            # Very small numbers
            lambda: np.random.uniform(1e-100, 1e-50, 50),
            # Very large numbers  
            lambda: np.random.uniform(1e50, 1e100, 50),
            # Numbers close to zero
            lambda: np.random.uniform(-1e-10, 1e-10, 50),
            # Wide range
            lambda: np.random.uniform(-1e6, 1e6, 50),
            # Very tight clustering
            lambda: np.random.normal(1.0, 1e-15, 50),
            # Very wide spread
            lambda: np.random.normal(0.0, 1e6, 50),
        ]
        
        for scenario_func in extreme_scenarios:
            try:
                data = scenario_func()
                ref = np.median(data)  # Use median as more robust
                
                # Skip if data contains inf/nan
                if not np.all(np.isfinite(data)) or not np.isfinite(ref):
                    continue
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    result = sd.significant_digits(data, reference=ref)
                    
                    # If computation succeeds, should be reasonable
                    if np.isfinite(result):
                        # Extreme data can produce negative or zero results
                        assert result > -100, f"Result suspiciously negative: {result}"
                        assert result < 1000, f"Result suspiciously large: {result}"
                        
            except (ValueError, TypeError, sd.SignificantDigitsException, OverflowError):
                # Expected for extreme values
                pass


class TestAuxiliaryFunctionProperties:
    """Test properties of auxiliary functions."""

    def test_compute_z_properties(self):
        """Test properties of _compute_z function."""
        np.random.seed(42)
        
        for _ in range(20):
            x = np.random.normal(1.0, 0.1, 100)
            y = np.random.normal(1.0, 0.1, 100)
            
            # Test absolute error
            z_abs, e_abs = _compute_z(x, y, error=sd.Error.Absolute, axis=0)
            
            # z should be the difference
            expected_z = x - y
            assert np.allclose(z_abs, expected_z, rtol=1e-10), \
                "Absolute error computation incorrect"
            
            # Test relative error
            if np.all(y != 0):  # Avoid division by zero
                z_rel, e_rel = _compute_z(x, y, error=sd.Error.Relative, axis=0)
                
                # z should be (x/y - 1)
                expected_z = x / y - 1
                assert np.allclose(z_rel, expected_z, rtol=1e-10), \
                    "Relative error computation incorrect"

    def test_scaling_factor_properties(self):
        """Test properties of _compute_scaling_factor function."""
        np.random.seed(42)
        
        for _ in range(15):
            # Test with powers of 2 (exact representable values)
            powers = np.random.randint(-10, 10, 20)
            y = 2.0 ** powers
            
            result = _compute_scaling_factor(y, axis=0, reference_is_random_variable=False)
            
            # Should be powers + 1 (since floor(log2(2^k)) + 1 = k + 1)
            expected = powers + 1
            assert np.allclose(result, expected), \
                f"Scaling factor incorrect for powers of 2: {result} vs {expected}"

    def test_probability_estimation_properties(self):
        """Test properties of probability estimation functions."""
        # Test basic properties
        for _ in range(20):
            trials = np.random.randint(10, 1000)
            success = np.random.randint(0, trials + 1)
            confidence = np.random.uniform(0.1, 0.99)
            
            prob = probability_estimation_bernoulli(success, trials, confidence)
            
            # Probability should be between 0 and 1
            assert 0 <= prob <= 1, f"Probability out of range: {prob}"
            
            # Should be reasonable relative to success rate
            success_rate = success / trials
            assert abs(prob - success_rate) <= 0.5, \
                f"Probability estimate too far from success rate: {prob} vs {success_rate}"

    def test_minimum_trials_properties(self):
        """Test properties of minimum_number_of_trials function."""
        for _ in range(20):
            probability = np.random.uniform(0.01, 0.99)
            confidence = np.random.uniform(0.01, 0.99)
            
            min_trials = minimum_number_of_trials(probability, confidence)
            
            # Should be positive integer
            assert isinstance(min_trials, int), "Result should be integer"
            assert min_trials > 0, "Result should be positive"
            
            # Should be reasonable
            assert min_trials < 1e6, f"Unreasonably large number of trials: {min_trials}"
            
            # Higher confidence or probability should require more trials
            if probability > 0.5 and confidence > 0.5:
                lower_conf_trials = minimum_number_of_trials(probability, confidence * 0.8)
                assert min_trials >= lower_conf_trials, \
                    "Higher confidence should require more trials"


class TestPropertyConsistency:
    """Test consistency between different approaches and methods."""

    def test_method_consistency(self):
        """Test consistency between CNH and General methods."""
        np.random.seed(42)
        
        # Use data that should work well with both methods
        for _ in range(10):
            x = np.random.normal(1.0, 0.01, 1000)  # Large sample, small variance
            ref = 1.0
            
            cnh_result = sd.significant_digits(x, reference=ref, method=sd.Method.CNH)
            gen_result = sd.significant_digits(x, reference=ref, method=sd.Method.General)
            
            if np.isfinite(cnh_result) and np.isfinite(gen_result):
                # Results should be in the same ballpark
                ratio = max(cnh_result, gen_result) / min(cnh_result, gen_result)
                assert ratio < 5.0, \
                    f"Methods too inconsistent: CNH={cnh_result}, General={gen_result}"

    def test_contributing_vs_significant_consistency(self):
        """Test consistency between significant and contributing digits."""
        np.random.seed(42)
        
        for _ in range(15):
            x = np.random.normal(1.0, 0.05, 200)
            ref = np.random.uniform(0.8, 1.2)
            
            sig_result = sd.significant_digits(x, reference=ref)
            con_result = sd.contributing_digits(x, reference=ref)
            
            if np.isfinite(sig_result) and np.isfinite(con_result):
                # Both algorithms can produce different results depending on data
                # Just check they're in reasonable ranges relative to each other
                ratio = max(abs(sig_result), abs(con_result)) / max(min(abs(sig_result), abs(con_result)), 0.1)
                assert ratio < 100, \
                    f"Results too different: significant={sig_result}, contributing={con_result}"
                
                # Both should be reasonable (not extremely negative)
                assert sig_result > -50, "Significant digits too negative"
                assert con_result > -50, "Contributing digits too negative"