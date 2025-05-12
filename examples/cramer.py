#!/usr/bin/env python3
"""
Analysis script for significant and contributing digits calculation and comparison.
"""

import os
import sys
import numpy as np
import significantdigits as sd
import argparse

from significantdigits import args

# Constants
PROBABILITY_SIGNIFICANT = 0.99
CONFIDENCE_SIGNIFICANT = 0.95
PROBABILITY_CONTRIBUTING = 0.51
CONFIDENCE_CONTRIBUTING = 0.95

# Reference values
SD_REFERENCE_VALUES = {
    ("CNH", "Absolute"): 26.094599393993263,
    ("CNH", "Relative"): 27.094599370144074,
    ("General", "Absolute"): 25,
    ("General", "Relative"): 26,
}

CD_REFERENCE_VALUES = {
    ("CNH", "Absolute"): 31.777744510355838,
    ("CNH", "Relative"): 32.77774448650665,
    ("General", "Absolute"): 25,
    ("General", "Relative"): 26,
}


def load_data(file_path):
    """
    Load data from a text file.

    Args:
        file_path (str): Path to the data file

    Returns:
        numpy.ndarray: Loaded data

    Raises:
        FileNotFoundError: If the file does not exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    try:
        return np.loadtxt(file_path)
    except Exception as e:
        print(f"Error loading data file: {e}", file=sys.stderr)
        sys.exit(1)


def analyze_digits(x, y, method, error_type):
    """
    Calculate significant and contributing digits and compare with reference values.

    Args:
        x (numpy.ndarray): Input data
        y (numpy.ndarray): Reference data (usually mean of x)
        method (sd.Method): Method to use (CNH or General)
        error_type (sd.Error): Error type (Absolute or Relative)

    Returns:
        tuple: (significant_digits, contributing_digits, significant_error, contributing_error)
    """
    # Calculate significant digits
    significant = sd.significant_digits(
        x,
        reference=y,
        method=method,
        error=error_type,
        probability=PROBABILITY_SIGNIFICANT,
        confidence=CONFIDENCE_SIGNIFICANT,
    )

    # Calculate contributing digits
    contributing = sd.contributing_digits(
        x,
        reference=y,
        method=method,
        error=error_type,
        probability=PROBABILITY_CONTRIBUTING,
        confidence=CONFIDENCE_CONTRIBUTING,
    )

    # Get reference values
    significant_ref = SD_REFERENCE_VALUES[(method.name, error_type.name)]
    contributing_ref = CD_REFERENCE_VALUES[(method.name, error_type.name)]

    # Calculate errors
    significant_error = significant - significant_ref
    contributing_error = contributing - contributing_ref

    return (
        significant,
        contributing,
        significant_ref,
        contributing_ref,
        significant_error,
        contributing_error,
    )


def print_results(method, error_type, results):
    """
    Print the analysis results in a formatted way.

    Args:
        method (sd.Method): Method used
        error_type (sd.Error): Error type used
        results (tuple): Results from analyze_digits
    """
    (
        significant,
        contributing,
        significant_ref,
        contributing_ref,
        significant_error,
        contributing_error,
    ) = results

    print(f"Method: {method.name}")
    print(f"Error: {error_type.name}")
    print(
        f"Significant digits:  {significant:.17e}, reference: {significant_ref:.17e}, "
        f"error: {significant_error:+g}"
    )
    print(
        f"Contributing digits: {contributing:.17e}, reference: {contributing_ref:.17e}, "
        f"error: {contributing_error:+g}"
    )
    print()


def parse_args():
    """
    Parse command line arguments.

    Returns:
        Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Significant and Contributing Digits Analysis"
    )
    parser.add_argument(
        "--file",
        type=str,
        default="data/cramer-x0-10000.txt",
        help="Path to the data file",
    )
    return parser.parse_args()


def main():
    """Main function to run the digit analysis."""

    args = parse_args()

    # Define file path
    file_path = args.file

    # Load data
    try:
        x = load_data(file_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Calculate mean
    y = np.mean(x, axis=0)

    # Methods and error types to analyze
    methods = [sd.Method.CNH, sd.Method.General]
    error_types = [sd.Error.Absolute, sd.Error.Relative]

    # Run analysis for each method and error type
    for method in methods:
        for error_type in error_types:
            results = analyze_digits(x, y, method, error_type)
            print_results(method, error_type, results)


if __name__ == "__main__":
    main()
