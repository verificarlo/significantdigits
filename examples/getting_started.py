#!/usr/bin/env python3
"""
Simple recreation of the significantdigits examples
"""

import significantdigits as sd
import numpy as np
from numpy.random import uniform as U


def main():
    # Set seed for reproducibility
    np.random.seed(0)

    # Machine epsilon
    eps = 2**-52
    print(f"Machine epsilon: {eps}")

    # Simulate results with epsilon differences
    X = [1 + U(-1, 1) * eps for _ in range(10)]
    print(f"Generated {len(X)} values with small perturbations")

    # Example 1: Basic usage with known reference
    print("\n1. Basic usage with reference=1:")
    result1 = sd.significant_digits(X, reference=1)
    print(f"   Significant digits: {result1}")

    # Example 2: Using sample average as reference
    print("\n2. Using sample average as reference:")
    result2 = sd.significant_digits(X, reference=np.mean(X))
    print(f"   Sample mean: {np.mean(X)}")
    print(f"   Significant digits: {result2}")

    # Example 3: Print formatted results
    print("\n3. Formatted output:")
    fmt = sd.format_uncertainty(X, reference=1)
    print("   Formatted significant digits:")
    for line in fmt:
        print(f"   {line}")

    # Example 4: File-based analysis (simulating CLI)
    print("\n4. File-based analysis:")

    # Write data to file
    with open("test.txt", "w") as f:
        for x in X:
            f.write(f"{x}\n")

    # Read data from file
    with open("test.txt", "r") as f:
        file_data = [float(line.strip()) for line in f]

    result3 = sd.significant_digits(file_data, reference=1)
    print(f"   From file: {result3}")

    # CLI equivalent (as comment):
    print("   # CLI equivalent:")
    print(
        '   # significantdigits --metric significant -i "$(cat test.txt)" --input-format stdin --reference 1'
    )

    # Clean up
    import os

    os.remove("test.txt")

    print(f"\nAll results should be approximately: {result1}")


if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("Error: significantdigits library not found!")
        print("Install with: pip install significantdigits")
