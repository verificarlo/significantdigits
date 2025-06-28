"""
Integration tests for significantdigits CLI and end-to-end functionality.

These tests verify that the command-line interface works correctly
and that all components integrate properly.
"""
import subprocess
import sys
import tempfile
import os
import json
import numpy as np
import pytest

import significantdigits as sd
from significantdigits.__main__ import main
from significantdigits.args import parse_args


class TestCLIIntegration:
    """Test command-line interface integration."""

    def test_cli_help_command(self):
        """Test CLI help command."""
        result = subprocess.run(
            [sys.executable, "-m", "significantdigits", "--help"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "significantdigits" in result.stdout.lower()
        assert "metric" in result.stdout.lower()

    def test_cli_basic_significant_digits(self):
        """Test basic significant digits computation via CLI."""
        cmd = [
            sys.executable, "-m", "significantdigits",
            "--metric", "significant",
            "--input-format", "stdin",
            "--inputs", "[1.0, 1.1, 0.9, 1.05]",
            "--reference", "1.0"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        
        # Should output a result (exact value depends on computation)
        assert len(result.stdout.strip()) > 0

    def test_cli_basic_contributing_digits(self):
        """Test basic contributing digits computation via CLI."""
        cmd = [
            sys.executable, "-m", "significantdigits",
            "--metric", "contributing",
            "--input-format", "stdin",
            "--inputs", "[1.0, 1.1, 0.9, 1.05]",
            "--reference", "1.0"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0

    def test_cli_different_methods(self):
        """Test CLI with different methods."""
        base_cmd = [
            sys.executable, "-m", "significantdigits",
            "--metric", "significant",
            "--input-format", "stdin",
            "--inputs", "[1.0, 1.1, 0.9, 1.05]",
            "--reference", "1.0"
        ]
        
        for method in ["CNH", "General"]:
            cmd = base_cmd + ["--method", method]
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0, f"Method {method} failed"
            assert len(result.stdout.strip()) > 0

    def test_cli_different_errors(self):
        """Test CLI with different error types."""
        base_cmd = [
            sys.executable, "-m", "significantdigits",
            "--metric", "significant",
            "--input-format", "stdin",
            "--inputs", "[1.0, 1.1, 0.9, 1.05]",
            "--reference", "1.0"
        ]
        
        for error in ["absolute", "relative"]:
            cmd = base_cmd + ["--error", error]
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0, f"Error type {error} failed"
            assert len(result.stdout.strip()) > 0

    def test_cli_different_bases(self):
        """Test CLI with different bases."""
        base_cmd = [
            sys.executable, "-m", "significantdigits",
            "--metric", "significant",
            "--input-format", "stdin",
            "--inputs", "[1.0, 1.1, 0.9, 1.05]",
            "--reference", "1.0"
        ]
        
        for basis in [2, 8, 10, 16]:
            cmd = base_cmd + ["--basis", str(basis)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0, f"Basis {basis} failed"
            assert len(result.stdout.strip()) > 0

    def test_cli_probability_and_confidence(self):
        """Test CLI with custom probability and confidence."""
        cmd = [
            sys.executable, "-m", "significantdigits",
            "--metric", "significant",
            "--input-format", "stdin",
            "--inputs", "[1.0, 1.1, 0.9, 1.05]",
            "--reference", "1.0",
            "--probability", "0.99",
            "--confidence", "0.99"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0

    def test_cli_invalid_arguments(self):
        """Test CLI with invalid arguments."""
        # Missing required metric
        cmd = [
            sys.executable, "-m", "significantdigits",
            "--input-format", "stdin",
            "--inputs", "[1.0, 1.1, 0.9, 1.05]"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode != 0
        
        # Invalid metric
        cmd = [
            sys.executable, "-m", "significantdigits",
            "--metric", "invalid",
            "--input-format", "stdin",
            "--inputs", "[1.0, 1.1, 0.9, 1.05]"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode != 0
        
        # Invalid probability
        cmd = [
            sys.executable, "-m", "significantdigits",
            "--metric", "significant",
            "--input-format", "stdin",
            "--inputs", "[1.0, 1.1, 0.9, 1.05]",
            "--probability", "1.5"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode != 0


class TestFileIOIntegration:
    """Test file I/O integration."""

    def test_numpy_file_input_output(self):
        """Test with numpy file input/output."""
        # Create temporary numpy file
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as temp_input:
            test_data = np.array([1.0, 1.1, 0.9, 1.05])
            np.save(temp_input.name, test_data)
            temp_input_path = temp_input.name
        
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as temp_ref:
            ref_data = np.array(1.0)
            np.save(temp_ref.name, ref_data)
            temp_ref_path = temp_ref.name
        
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as temp_output:
            temp_output_path = temp_output.name
        
        try:
            cmd = [
                sys.executable, "-m", "significantdigits",
                "--metric", "significant",
                "--input-format", "npy",
                "--inputs", temp_input_path,
                "--reference", temp_ref_path,
                "--output-format", "npy",
                "--output", temp_output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0
            
            # Check that output file was created and contains valid data
            assert os.path.exists(temp_output_path)
            output_data = np.load(temp_output_path)
            assert np.all(np.isfinite(output_data))
        
        finally:
            # Clean up temporary files
            os.unlink(temp_input_path)
            os.unlink(temp_ref_path)
            if os.path.exists(temp_output_path):
                os.unlink(temp_output_path)

    def test_stdin_format_without_reference(self):
        """Test stdin format without explicit reference (uses mean automatically)."""
        cmd = [
            sys.executable, "-m", "significantdigits",
            "--metric", "significant",
            "--input-format", "stdin",
            "--inputs", "[1.0, 1.1, 0.9, 1.05]"
            # No reference specified - should use mean
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    def test_complete_analysis_workflow(self):
        """Test a complete analysis workflow."""
        # Simulate a complete analysis from data generation to result
        np.random.seed(42)
        
        # Generate test data
        true_value = 1.0
        noise = np.random.normal(0, 0.01, 1000)
        measurements = true_value + noise
        
        # Test significant digits analysis
        sig_digits = sd.significant_digits(
            measurements, 
            reference=true_value,
            method=sd.Method.CNH,
            error=sd.Error.Absolute,
            probability=0.95,
            confidence=0.95
        )
        
        # Test contributing digits analysis
        con_digits = sd.contributing_digits(
            measurements,
            reference=true_value,
            method=sd.Method.CNH,
            error=sd.Error.Absolute,
            probability=0.51,
            confidence=0.95
        )
        
        # Test uncertainty formatting
        formatted = sd.format_uncertainty(
            measurements[:10],  # Smaller subset for formatting
            reference=true_value,
            error=sd.Error.Absolute
        )
        
        # Test basis conversion
        sig_digits_base10 = sd.change_basis(sig_digits, 10)
        
        # Verify all results are reasonable
        assert np.all(np.isfinite(sig_digits))
        assert np.all(np.isfinite(con_digits))
        assert formatted.shape == (10,)
        assert np.all(np.isfinite(sig_digits_base10))
        
        # Results should be positive (number of digits)
        assert sig_digits > 0
        assert con_digits > 0
        assert sig_digits_base10 > 0

    def test_comparative_analysis_workflow(self):
        """Test comparative analysis between methods."""
        np.random.seed(42)
        data = np.random.normal(1.0, 0.1, 500)
        reference = np.mean(data)
        
        # Compare CNH vs General method
        results = {}
        
        for method in [sd.Method.CNH, sd.Method.General]:
            for error in [sd.Error.Absolute, sd.Error.Relative]:
                key = f"{method.name}_{error.name}"
                
                sig_result = sd.significant_digits(
                    data, reference=reference,
                    method=method, error=error
                )
                
                con_result = sd.contributing_digits(
                    data, reference=reference,
                    method=method, error=error
                )
                
                results[key] = {
                    "significant": sig_result,
                    "contributing": con_result
                }
        
        # Verify all results are finite and reasonable
        for key, result in results.items():
            assert np.all(np.isfinite(result["significant"])), f"{key} significant digits invalid"
            assert np.all(np.isfinite(result["contributing"])), f"{key} contributing digits invalid"
            assert result["significant"] > 0, f"{key} significant digits not positive"
            assert result["contributing"] > 0, f"{key} contributing digits not positive"

    def test_multidimensional_analysis_workflow(self):
        """Test workflow with multidimensional data."""
        np.random.seed(42)
        
        # 3D data: samples x measurements x repetitions
        shape = (100, 5, 3)
        data = np.random.normal(1.0, 0.05, shape)
        
        # Analyze along different axes
        for axis in range(len(shape)):
            if shape[axis] > 1:  # Need at least 2 samples for analysis
                reference = np.mean(data, axis=axis)
                
                sig_result = sd.significant_digits(
                    data, reference=reference, axis=axis
                )
                
                con_result = sd.contributing_digits(
                    data, reference=reference, axis=axis
                )
                
                expected_shape = list(shape)
                expected_shape.pop(axis)
                expected_shape = tuple(expected_shape)
                
                assert sig_result.shape == expected_shape
                assert con_result.shape == expected_shape
                assert np.all(np.isfinite(sig_result))
                assert np.all(np.isfinite(con_result))

    def test_batch_processing_workflow(self):
        """Test batch processing workflow."""
        np.random.seed(42)
        
        # Simulate multiple datasets
        datasets = []
        references = []
        
        for i in range(5):
            data = np.random.normal(1.0 + i*0.1, 0.02, 200)
            datasets.append(data)
            references.append(1.0 + i*0.1)
        
        # Process each dataset
        all_sig_results = []
        all_con_results = []
        
        for data, ref in zip(datasets, references):
            sig_result = sd.significant_digits(data, reference=ref)
            con_result = sd.contributing_digits(data, reference=ref)
            
            all_sig_results.append(sig_result)
            all_con_results.append(con_result)
        
        # Verify batch results
        assert len(all_sig_results) == 5
        assert len(all_con_results) == 5
        
        for sig_result, con_result in zip(all_sig_results, all_con_results):
            assert np.all(np.isfinite(sig_result))
            assert np.all(np.isfinite(con_result))
            assert sig_result > 0
            assert con_result > 0


class TestModuleIntegration:
    """Test integration between different modules."""

    def test_stats_module_integration(self):
        """Test integration with stats module."""
        from significantdigits import stats
        
        # Test dense arrays
        dense_data = np.array([1.0, 2.0, 3.0, 4.0])
        mean_result = stats.mean(dense_data)
        std_result = stats.std(dense_data)
        
        assert np.isfinite(mean_result)
        assert np.isfinite(std_result)
        assert std_result > 0
        
        # Use stats results in main computation
        sig_result = sd.significant_digits(dense_data, reference=mean_result)
        assert np.all(np.isfinite(sig_result))

    def test_export_module_integration(self):
        """Test integration with export module."""
        from significantdigits.export import input_formats, output_formats
        
        # Test that all input formats are available
        assert "stdin" in input_formats
        assert "npy" in input_formats
        
        # Test that all output formats are available
        assert "stdin" in output_formats
        assert "npy" in output_formats
        
        # Test stdin parser
        stdin_parser = input_formats["stdin"]()
        parsed_data = stdin_parser.parse([1.0, 2.0, 3.0], dtype=np.float64)
        assert isinstance(parsed_data, np.ndarray)
        assert parsed_data.dtype == np.float64

    def test_args_module_integration(self):
        """Test integration with args module."""
        # Test argument parsing integration
        test_args = [
            "--metric", "significant",
            "--method", "CNH",
            "--error", "relative",
            "--input-format", "stdin",
            "--inputs", "[1.0, 2.0, 3.0, 4.0]",
            "--reference", "2.5",
            "--probability", "0.95",
            "--confidence", "0.95",
            "--basis", "10"
        ]
        
        parsed_args = parse_args(test_args)
        
        # Verify all args are parsed correctly
        assert parsed_args.metric == sd.Metric.Significant
        assert parsed_args.method == sd.Method.CNH
        assert parsed_args.error == sd.Error.Relative
        assert parsed_args.probability == 0.95
        assert parsed_args.confidence == 0.95
        assert parsed_args.basis == 10


class TestErrorHandlingIntegration:
    """Test error handling across integrated components."""

    def test_cli_error_propagation(self):
        """Test that errors propagate correctly through CLI."""
        # Test invalid input format
        cmd = [
            sys.executable, "-m", "significantdigits",
            "--metric", "significant",
            "--input-format", "stdin",
            "--inputs", "invalid_syntax_here"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode != 0
        assert "error" in result.stderr.lower() or "error" in result.stdout.lower()

    def test_computation_error_handling(self):
        """Test error handling in computation pipeline."""
        # Test with data that should cause issues
        problematic_data = [np.inf, np.nan, 1.0, 2.0]
        
        # Should handle gracefully (may warn but shouldn't crash)
        try:
            result = sd.significant_digits(problematic_data, reference=1.5)
            # If it succeeds, result may contain inf/nan but shouldn't crash
        except (ValueError, RuntimeWarning):
            # Expected for problematic data
            pass

    def test_integration_error_recovery(self):
        """Test error recovery in integrated workflows."""
        # Test that one failed computation doesn't break subsequent ones
        good_data = np.array([1.0, 1.1, 0.9, 1.05])
        
        # This should work
        result1 = sd.significant_digits(good_data, reference=1.0)
        assert np.all(np.isfinite(result1))
        
        # Try something that might fail
        try:
            problematic_result = sd.significant_digits([], reference=1.0)
        except (ValueError, TypeError):
            pass  # Expected
        
        # This should still work after the failure
        result2 = sd.significant_digits(good_data, reference=1.0)
        assert np.all(np.isfinite(result2))
        assert np.allclose(result1, result2)  # Should be identical