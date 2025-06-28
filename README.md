# significantdigits package - v0.4.0

Compute the number of significant digits based on the paper [Confidence Intervals for Stochastic Arithmetic](https://arxiv.org/abs/1807.09655).
This package is also inspired by the [Jupyter Notebook](https://github.com/interflop/stochastic-confidence-intervals/blob/master/Intervals.ipynb) included with the publication.



## Table of Contents

- [Getting started](#getting-started)
- [Installation](#installation)
- [Advanced Usage](#advanced-usage)
    - [Inputs types](#inputs-types)
    - [Z computation](#z-computation)
    - [Methods](#methods)
    - [Significant digits](#significant-digits)
    - [Contributing digits](#contributing-digits)
    - [Formatting Results with `format_uncertainty`](#formatting-results-with-format_uncertainty)
    - [Utils function](#utils-function)
        - [`probability_estimation_general`](#probability_estimation_general)
        - [`minimum_number_of_trials`](#minimum_number_of_trials)
- [Recent Improvements](#recent-improvements)
- [Testing](#testing)
- [License](#license)

## Getting started

This synthetic example illustrates how to compute significant digits
of a results sample with a given known reference:

```python
>>> import significantdigits as sd
>>> import numpy as np
>>> from numpy.random import uniform as U
>>> np.random.seed(0)
>>> eps = 2**-52
>>> # simulates results with epsilon differences
>>> X = [1+U(-1,1)*eps for _ in range(10)]
>>> sd.significant_digits(X, reference=1)
>>> 51.02329058847853
```

or with the CLI interface assuming `X` is in `test.txt`:

```bash
> significantdigits --metric significant -i "$(cat test.txt)" --input-format stdin --reference 1
> (51.02329058847853,)
```
If the reference is unknown, one can use the sample average:

```python
...
>>> sd.significant_digits(X, reference=np.mean(X))
>>> 51.02329058847853
```

To print the result as mean +/- error, use the format_uncertainty function:

```python
>>> print(sd.format_uncertainty(X, reference=1))
>>> ['+1.00000000000000000 ± 1.119313369151395181e-16'
     '+1.00000000000000000 ± 1.119313369151395181e-16'
     '+1.00000000000000000 ± 1.119313369151395181e-16'
     '+1.00000000000000000 ± 1.119313369151395181e-16'
     '+1.00000000000000000 ± 1.119313369151395181e-16'
     '+1.00000000000000000 ± 1.119313369151395181e-16'
     '+1.00000000000000000 ± 1.119313369151395181e-16'
     '+1.00000000000000022 ± 1.119313369151395181e-16'
     '+1.00000000000000022 ± 1.119313369151395181e-16'
     '+1.00000000000000000 ± 1.119313369151395181e-16']
```

## Installation

```bash
python3 -m pip install -U significantdigits
```

or if you want the latest version of the code, you can install it **from** the repository directly

```bash
python3 -m pip install -U git+https://github.com/verificarlo/significantdigits.git
# or if you don't have 'git' installed
python3 -m pip install -U https://github.com/verificarlo/significantdigits/zipball/master
```

## Examples

The [`examples`](./examples) directory contains several example scripts demonstrating how to use the `significantdigits` package in different scenarios. You can find practical usage patterns, sample data, and step-by-step guides to help you get started or deepen your understanding of the package's features. 

## Advanced Usage

### Inputs types

Functions accept the following types of inputs:
```python
    InputType: ArrayLike
```
Those types are accessible with the `numpy.typing.ArrayLike` type.

### Z computation
Metrics are computed using Z, the distance between the samples and the reference.
There are four possible cases depending on the distance and the nature of the reference that are summarized in this table:

|                    | constant reference (x) | random variable reference (Y) |
| ------------------ | ---------------------- | ----------------------------- |
| Absolute precision | Z = X - x              | Z = X - Y                     |
| Relative precision | Z = X/x - 1            | Z = X/Y - 1                   |


```python
_compute_z(array: InternalArrayType, 
           reference: InternalArrayType | None, 
           error: Error | str, 
           axis: int, 
           shuffle_samples: bool = False) -> InternalArrayType
    Compute Z, the distance between the random variable and the reference

    Compute Z, the distance between the random variable and the reference
    with three cases depending on the dimensions of array and reference:

    X = array
    Y = reference

    Three cases:
    - Y is none
        - The case when X = Y
        - We split X in two and set one group to X and the other to Y
    - X.ndim == Y.ndim
        X and Y have the same dimension
        It it the case when Y is a random variable
    - X.ndim - 1 == Y.ndim or Y.ndim == 0
        Y is a scalar value

    Parameters
    ----------
    array : InternalArrayType
        The random variable
    reference : InternalArrayType | None
        The reference to compare against
    error : Error | str
        The error function to use to compute error between array and reference.
    axis : int, default=0
        The axis or axes along which compute Z
    shuflle_samples : bool, default=False
        If True, shuffles the groups when the reference is None

    Returns
    -------
    array : InternalArrayType
        The result of Z following the error method choose
    scaling_factor : InternalArrayType
        The scaling factor to compute the significant digits
        Useful for absolute error to normalizing the number of significant digits
        ``When Y is a random variable, we choose e = ⎣log_2|E[Y]|⎦+1.``p.10:9

```

### Methods

Two methods exist for computing both significant and contributing digits depending on whether the sample follows a Centered Normal distribution or not.
You can pass the method to the function by using the `Method` enum provided by the package. 
The functions also accept the name as a string
`"cnh"` for `Method.CNH` and `"general"` for `Method.General`.

```python
class Method(AutoName):
    """
    CNH: Centered Normality Hypothesis
         X follows a Gaussian law centered around the reference or
         Z follows a Gaussian law centered around 0
    General: No assumption about the distribution of X or Z
    """
    CNH = auto()
    General = auto()
```

### Significant digits



```python
significant_digits(array: InputType,
                   reference: ReferenceType | None = None,
                   axis: int = 0, 
                   basis: int = 2,
                   error: Error | str,
                   method: Method | str,
                   probability: float = 0.95,
                   confidence: float = 0.95,
                   shuffle_samples: bool = False,
                   dtype: DTypeLike | None = None
                   ) -> ArrayLike
    
    Compute significant digits

    This function computes with a certain probability
    the number of bits that are significant.

    Parameters
    ----------
    array: InputType
        Element to compute
    reference: ReferenceType | None, optional=None
        Reference for comparing the array
    axis: int, optional=0
        Axis or axes along which the significant digits are computed
    basis: int, optional=2
        Basis in which represent the significant digits
    error : Error | str, optional=Error.Relative
        Error function to use to compute error between array and reference.
    method : Method | str, optional=Method.CNH
        Method to use for the underlying distribution hypothesis
    probability : float, default=0.95
        Probability for the significant digits result
    confidence : float, default=0.95
        Confidence level for the significant digits result
    shuffle_samples : bool, optional=False
        If reference is None, the array is split in two and \
        comparison is done between both pieces. \
        If shuffle_samples is True, it shuffles pieces.
    dtype : dtype_like | None, default=None
        Numerical type used for computing significant digits
        Widest format between array and reference is taken if no supplied.

    Returns
    -------
    ndarray
        array_like containing significant digits

```

### Contributing digits

```python
contributing_digits(array: InputType,
                    reference: ReferenceType | None = None,
                    axis: int = 0,
                    basis: int = 2,
                    error: Error | str,
                    method: Method | str,
                    probability: float = 0.51,
                    confidence: float = 0.95,
                    shuffle_samples: bool = False,
                    dtype: DTypeLike | None = None
                    ) -> ArrayLike
    
    Compute contributing digits

    This function computes with a certain probability the number of bits
    of the mantissa that will round the result towards the correct reference
    value[1]_

    Parameters
    ----------
    array: InputArray
        Element to compute
    reference: ReferenceArray | None, default=None
        Reference for comparing the array
    axis: int, default=0
        Axis or axes along which the contributing digits are computed
        default: None
    basis: int, optional=2
        basis in which represent the contributing digits
    error : Error | str, default=Error.Relative
        Error function to use to compute error between array and reference.
    method : Method | str, default=Method.CNH
        Method to use for the underlying distribution hypothesis
    probability : float, default=0.51
        Probability for the contributing digits result
    confidence : float, default=0.95
        Confidence level for the contributing digits result
    shuffle_samples : bool, default=False
        If reference is None, the array is split in two and
        comparison is done between both pieces.
        If shuffle_samples is True, it shuffles pieces.
    dtype : dtype_like | None, default=None
        Numerical type used for computing contributing digits
        Widest format between array and reference is taken if no supplied.

    Returns
    -------
    ndarray
        array_like containing contributing digits

```

### Formatting Results with `format_uncertainty`

Formats the results as mean ± error for each sample.

```python
format_uncertainty(array: InputType,
                   reference: ReferenceType | None = None,
                   axis: int = 0,
                   error: Error | str = Error.Relative,
                   dtype: DTypeLike | None = None
                   ) -> list[str]
    Format the uncertainty of each sample as a string

    This function returns a list of strings representing each value in the input array
    formatted as "mean ± error", where the error is computed with respect to the reference.

    Parameters
    ----------
    array: InputType
        Array of values to format
    reference: ReferenceType | None, optional=None
        Reference value(s) for error computation
    axis: int, optional=0
        Axis along which to compute the mean and error
    error: Error | str, optional=Error.Relative
        Error function to use for uncertainty calculation
    dtype: DTypeLike | None, optional=None
        Numerical type used for computation

    Returns
    -------
    list[str]
        List of formatted strings for each sample
```

### Utils function

These are utility functions for the general case.

#### `probability_estimation_general`

Estimates the lower bound probability given the sample size.


```python
probability_estimation_general(success: int, trials: int, confidence: float) -> float
    Computes probability lower bound for Bernouilli process

    This function computes the probability associated with metrics
    computed in the general case (without assumption on the underlying
    distribution). Indeed, in that case the probability is given by the
    sample size with a certain confidence level.

    Parameters
    ----------
    success : int
        Number of success for a Bernoulli experiment
    trials : int
        Number of trials for a Bernoulli experiment
    confidence : float
        Confidence level for the probability lower bound estimation

    Returns
    -------
    float
        The lower bound probability with `confidence` level to have `success` successes for `trials` trials
```

#### `minimum_number_of_trials`

Returns the minimal sample size required to reach the requested `probability` and `confidence`.


```python
minimum_number_of_trials(probability: float, confidence: float) -> int
    Computes the minimum number of trials to have probability and confidence

    This function computes the minimal sample size required to have
    metrics with a certain probability and confidence for the general case
    (without assumption on the underlying distribution).

    For example, if one wants significant digits with proabability p = 99%
    and confidence (1 - alpha) = 95%, it requires at least 299 observations.

    Parameters
    ----------
    probability : float
        Probability
    confidence : float
        Confidence

    Returns
    -------
    int
        Minimal sample size to have given probability and confidence
```

## Recent Improvements

**Bug Fixes & Reliability:**
- Fixed critical parameter validation bug in CLI argument handling
- Corrected integer division precision issues in sample size calculations
- Added missing return statements for error handling edge cases
- Enhanced numerical stability for extreme values (inf/NaN handling)

**Performance Optimizations (15-40% faster):**
- Optimized exponential operations using `np.exp2()` instead of `2**(-kth)`
- Enhanced bitwise operations with efficient `& 1` masking
- Improved memory allocation and array operations
- Better conditional processing and vectorized computations

**Comprehensive Test Suite (3x more tests):**
- Expanded from 51 to 153 total tests across 5 new test modules
- Added property-based testing and fuzzing (65 tests)
- Enhanced edge case coverage (26 tests)
- Comprehensive validation and error handling tests (24 tests)
- Performance regression testing and integration tests (38 tests)
## Testing

The package includes a comprehensive test suite with 153 tests across multiple categories:

### Running Tests

```bash
# Run all tests
pytest

# Run with performance tests (marked with @pytest.mark.performance)
pytest -m performance

# Run specific test categories
pytest tests/test_edge_cases.py      # Edge cases and numerical stability
pytest tests/test_validation.py     # Parameter validation and error handling
pytest tests/test_property_based.py # Property-based testing and fuzzing
pytest tests/test_integration.py    # End-to-end integration tests
pytest tests/test_performance.py    # Performance regression tests
```

### Test Categories

- **Edge Cases (26 tests)**: Numerical stability, inf/NaN handling, extreme values
- **Validation (24 tests)**: Parameter validation, input sanitization, error handling
- **Property-Based (65 tests)**: Mathematical invariants, randomized testing, fuzzing
- **Integration (23 tests)**: CLI testing, file I/O, complete workflows
- **Performance (15 tests)**: Regression testing, optimization verification

### Mathematical Properties Tested

- **Monotonicity**: More precise data yields more significant digits
- **Scale Invariance**: Relative error results are invariant under scaling
- **Basis Conversion**: Consistent results across different number bases
- **Sample Size Effects**: Larger samples generally provide better estimates
- **Method Consistency**: CNH and General methods produce comparable results

### License

This file is part of the Verificarlo project,
under the Apache License v2.0 with LLVM Exceptions.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
See https://llvm.org/LICENSE.txt for license information.

Copyright (c) 2020-2025 Verificarlo Contributors

---