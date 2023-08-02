# significantdigits package - v0.1.3

Compute the number of significant digits based on the paper [Confidence Intervals for Stochastic Arithmetic](https://arxiv.org/abs/1807.09655).
This package is also inspired by the [Jupyter Notebook](https://github.com/interflop/stochastic-confidence-intervals/blob/master/Intervals.ipynb) included with the publication.

## Getting started

This synthetic example illustrates how to compute significant digits
of a results sample with a given known reference:

```python
>>> import significantdigits as sig
>>> import numpy as np
>>> from numpy.random import uniform as U
>>> np.random.seed(0)
>>> eps = 2**-52
>>> # simulates results with epsilon differences
>>> X = [1+U(-1,1)*eps for _ in range(10)]
>>> sig.significant_digits(X, reference=1)
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
>>> sig.significant_digits(X, reference=np.mean(X))
>>> 51.02329058847853
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

## Advanced Usage

### Inputs types

Functions accept the following types of inputs:
```python
    InputType: np.ndarray | tuple | list
```
Those types are accessible with the `get_input_type` function.

### Z computation
Metrics are computed using Z, the distance between the samples and the reference.
They are four possible cases depending on the distance and the nature of the reference that is summarized in this table:

|                    | constant reference (x) | random variable reference (Y) |
| ------------------ | ---------------------- | ----------------------------- |
| Absolute precision | Z = X - x              | Z = X - Y                     |
| Relative precision | Z = X/x - 1            | Z = X/Y - 1                   |


```python
compute_z(array: ~InputType, reference: Optional[~ReferenceType], error: significantdigits._significantdigits.Error | str, axis: int, shuffle_samples: bool = False) -> ~InputType
    Compute Z, the distance between the random variable and the reference

    Compute Z, the distance between the random variable and the reference
    with three cases depending on the dimensions of array and reference:

    X = array
    Y = reference
    Three cases:
        - Y is none
            The case when X = Y
            We split X in two and set one group to X and the other to Y
        - X.ndim == Y.ndim
            X and Y have the same dimension
            It it the case when Y is a random variable
        - X.ndim - 1 == Y.ndim or Y.ndim == 0
            Y is a scalar value

    Parameters
    ----------
    array : InputType
        The random variable
    reference : Optional[ReferenceType]
        The reference to compare against
    error : Method.Error | str
        The error function to compute Z
    axis : int, default=0
        The axis or axes along which compute Z
    shuflle_samples : bool, default=False
        If True, shuffles the groups when the reference is None

    Returns
    -------
    array : numpy.ndarray
        The result of Z following the error method choose
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
significant_digits(array: ~InputType,
                   reference: Optional[~ReferenceType] = None,
                   axis: int = 0, base: int = 2,
                   error: str | significantdigits._significantdi
    Compute significant digits

    This function computes with a certain probability
    the number of bits that are significant.

    Parameters
    ----------
    array: InputType
        Element to compute
    reference: Optional[ReferenceType], optional=None
        Reference for comparing the array
    base: int, optional=2
        Base in which represent the significant digits
    axis: int, optional=0
        Axis or axes along which the significant digits are computed
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
    dtype : np.dtype, default=None
        Numerical type used for computing contributing digits
        Widest format between array and reference is taken if no supplied.

    Returns
    -------
    ndarray
        array_like containing contributing digits

```

### Contributing digits

```python
contributing_digits(array: ~InputType, reference: Optional[~ReferenceType] = None, axis: int = 0, base: int = 2, error: str | significantdigits._significantdigits.Error = <Error.Re$
    Compute contributing digits


    This function computes with a certain probability the number of bits
    of the mantissa that will round the result towards the correct reference
    value[1]_

    Parameters
    ----------
    array: InputArray
        Element to compute
    reference: Optional[ReferenceArray], default=None
        Reference for comparing the array
    axis: int, default=0
        Axis or axes along which the contributing digits are computed
        default: None
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
    dtype : np.dtype, default=None
        Numerical type used for computing contributing digits
        Widest format between array and reference is taken if no supplied.

    Returns
    -------
    ndarray
        array_like containing contributing digits

```
### Utils function

These are utility functions for the general case.

`probability_estimation_general`
allows having an estimation
on the lower bound probability given the sample size.

`minimum_number_of_trials` gives the minimal sample size
required to reach the requested `probability` and `confidence`.

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

### License

This file is part of the Verificarlo project,
under the Apache License v2.0 with LLVM Exceptions.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
See https://llvm.org/LICENSE.txt for license information.

Copyright (c) 2020-2023 Verificarlo Contributors

