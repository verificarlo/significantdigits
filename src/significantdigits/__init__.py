"""
.. include:: ../README.md
"""

from ._significantdigits import (
    InputType,
    ReferenceType,
    Method,
    Metric,
    Error,
    SignificantDigitsException,
    format_uncertainty,
    significant_digits,
    contributing_digits,
    change_basis,
    probability_estimation_bernoulli,
    minimum_number_of_trials,
)

__all__ = [
    "InputType",
    "ReferenceType",
    "Method",
    "Metric",
    "Error",
    "SignificantDigitsException",
    "significant_digits",
    "contributing_digits",
    "change_basis",
    "probability_estimation_bernoulli",
    "minimum_number_of_trials",
    "format_uncertainty",
]
