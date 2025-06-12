from __future__ import annotations

import os
import typing
import warnings
from enum import Enum, auto
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy
import scipy.stats
from icecream import ic


def _get_verbose_mode():
    """@private"""
    verbose = os.getenv("SD_VERBOSE", "0")
    if verbose.lower() in ("1", "true", "yes", "on"):
        return True
    return False


_VERBOSE_MODE = _get_verbose_mode()
ic.configureOutput(
    includeContext=True,
    prefix="sd| ",
)

# Configure ic for debugging
if _VERBOSE_MODE:
    ic.enable()
else:
    ic.disable()

__all__ = [
    "significant_digits",
    "contributing_digits",
    "change_basis",
    "probability_estimation_bernoulli",
    "minimum_number_of_trials",
    "InputType",
    "ReferenceType",
    "Metric",
    "Method",
    "Error",
    "SignificantDigitsException",
]


class SignificantDigitsException(Exception):
    pass


class AutoName(Enum):
    """@private"""

    names: list[str]
    """@private"""
    map: dict[str, AutoName]
    """@private"""

    def _generate_next_value_(name, start, count, last_values):
        return name


class Metric(AutoName):
    """
    Different metrics to compute

    - Significant: Compute the number of significant digits
    - Contributing: Compute the number of contributing digits
    """

    Significant = auto()
    Contributing = auto()

    @classmethod
    def is_significant(cls, error):
        """@private"""
        if isinstance(error, cls):
            return error == cls.Significant
        if isinstance(error, str):
            return error.lower() == cls.Significant.name

    @classmethod
    def is_contributing(cls, error):
        """@private"""
        if isinstance(error, cls):
            return error == cls.Contributing
        if isinstance(error, str):
            return error.lower() == cls.Contributing.name


class Method(AutoName):
    """
    Methods for underlying distribution hypothesis

    - CNH: Centered Normality Hypothesis
        - X follows a Gaussian law centered around the reference or
        - Z follows a Gaussian law centered around 0
    - General: No assumption about the distribution of X or Z
    """

    CNH = auto()
    General = auto()

    @classmethod
    def is_cnh(cls, error):
        """@private"""
        if isinstance(error, cls):
            return error == cls.CNH
        if isinstance(error, str):
            return error.lower() == cls.CNH.name

    @classmethod
    def is_general(cls, error):
        """@private"""
        if isinstance(error, cls):
            return error == cls.General
        if isinstance(error, str):
            return error.lower() == cls.General.name


class Error(AutoName):
    r"""
    Errors between random variable and reference

    Reference is either a random variable ($Y$) or a constant ($x$)

    .. math::
    \begin{array}{|c|c|c|}
    \hline
                & \text{Reference } x & \text{Reference } Y \newline
    \hline
    \text{Absolute}    &  Z = X - x   &  Z = X - Y  \newline
    \text{Relative}    &  Z = X/x - 1  &  Z = X/Y - 1 \newline
    \hline
    \end{array}
    """

    Absolute = auto()
    Relative = auto()

    @classmethod
    def is_absolute(cls, error):
        """@private"""
        if isinstance(error, cls):
            return error == cls.Absolute
        if isinstance(error, str):
            return error.lower() == cls.Absolute.name

    @classmethod
    def is_relative(cls, error):
        """@private"""
        if isinstance(error, cls):
            return error == cls.Relative
        if isinstance(error, str):
            return error.lower() == cls.Relative.name


@typing.overload
def _lower_map(x: list[str]) -> list[str]: ...


@typing.overload
def _lower_map(x: dict[str, AutoName]) -> dict[str, AutoName]: ...


def _lower_map(x):
    if isinstance(x, list):
        return list(map(str.lower, x))
    if isinstance(x, dict):
        return {k.lower(): v for k, v in x.items()}


Metric.names = _lower_map(vars(Metric)["_member_names_"])
Metric.map = _lower_map(vars(Metric)["_value2member_map_"])

Method.names = _lower_map(vars(Method)["_member_names_"])
Method.map = _lower_map(vars(Method)["_value2member_map_"])

Error.names = _lower_map(vars(Error)["_member_names_"])
Error.map = _lower_map(vars(Error)["_value2member_map_"])

_internal_dtype = np.dtype(np.float64)
_default_probability = {Metric.Significant: 0.95, Metric.Contributing: 0.51}
_default_confidence = {Metric.Significant: 0.95, Metric.Contributing: 0.95}

InputType = npt.ArrayLike
r"""Valid random variable inputs type (np.ndarray, tuple, list)

Types allowing for `array` in significant_digits and contributing_digits functions

"""

OutputType = npt.NDArray[np.number]
r"""Valid output type"""

ReferenceType = Union[npt.ArrayLike, np.number]
r"""Valid reference inputs type (np.ndarray, tuple, list, float, int)

Types allowing for `reference` in significant_digits and contributing_digits functions

"""

InternalArrayType = npt.NDArray[np.number]
r"""Internal array type used for computation"""


def _assert_is_valid_metric(metric: Union[Metric, str]) -> None:  # type: ignore
    if Metric.is_significant(metric) or Metric.is_contributing(metric):
        return

    raise TypeError(f"provided invalid metric {metric}: must be one of {list(Metric)}")


def _assert_is_valid_method(method: Union[Method, str]) -> None:
    if Method.is_cnh(method) or Method.is_general(method):
        return

    raise TypeError(f"provided invalid method {method}: must be one of {list(Method)}")


def _assert_is_valid_error(error: Union[Error, str]) -> None:
    if Error.is_absolute(error) or Error.is_relative(error):
        return

    raise TypeError(f"provided invalid error {error}: must be one of {list(Error)}")


def _assert_is_probability(probability: float) -> None:
    if not (0 <= probability <= 1):
        raise TypeError("probability must be between 0 and 1")


def _assert_is_confidence(confidence: float) -> None:
    if not (0 <= confidence <= 1):
        raise TypeError("confidence must be between 0 and 1")


def _assert_is_valid_inputs(array: InputType) -> None:
    if not isinstance(array, (np.ndarray, list, tuple)):
        raise TypeError(
            f"array must be of type {InputType}, not {type(array).__name__}"
        )
    if isinstance(array, np.ndarray) and array.ndim == 0:
        raise TypeError("array must be at least 1D")


def _preprocess_inputs(
    array: InputType,
    reference: Optional[ReferenceType],
) -> Tuple[InternalArrayType, Optional[InternalArrayType]]:
    preprocessed_array = np.asanyarray(array)
    preprocessed_reference = np.asanyarray(reference) if reference is not None else None
    return (preprocessed_array, preprocessed_reference)


def change_basis(array: InputType, basis: int) -> OutputType:
    """Changes basis from binary to `basis` representation

    Parameters
    ----------
    array : np.ndarray
        array_like containing significant or contributing bits
    basis : int
        output basis

    Returns
    -------
    np.ndarray
        Array convert to basis `basis`
    """
    (preprocessed_array, _) = _preprocess_inputs(array, None)
    factor = np.log(2) / np.log(basis)
    return preprocessed_array * factor


def _operator_along_axis(operator, x, y, axis):
    shape = list(y.shape)
    shape.insert(axis, 1)
    y_reshaped = np.reshape(y, shape)
    return operator(x, y_reshaped)


def _divide_along_axis(x, y, axis):
    return _operator_along_axis(np.divide, x, y, axis)


def _substract_along_axis(x, y, axis):
    return _operator_along_axis(np.subtract, x, y, axis)


def _get_significant_size(
    z: InternalArrayType, dtype: Optional[npt.DTypeLike] = None
) -> int:
    if dtype is None:
        dtype = z.dtype

    return np.finfo(dtype).nmant  # type: ignore


def _fill_where(
    x: InternalArrayType, fill_value: InternalArrayType, mask: npt.NDArray
) -> InternalArrayType:
    """Fill x with fill_value where mask is True"""
    if x.ndim == 0:
        if mask:
            x = fill_value
    elif fill_value.ndim == 0:
        x[mask] = fill_value
    else:
        x[mask] = fill_value[mask]
    return x


def _compute_scaling_factor(
    y: InternalArrayType, axis: int, reference_is_random_variable: bool
) -> InternalArrayType:
    """Compute the scaling factor for the error

    The scaling factor is computed as the number of significant digits
    of the reference. It is used to normalize the number of significant digits
    when using absolute error.

    Parameters
    ----------
    y : InternalArrayType
        The reference to compare against
    axis : int
        The axis or axes along which compute the scaling factor
    reference_is_random_variable : bool
        If True, the reference is a random variable

    Returns
    -------
    InternalArrayType
        The scaling factor to compute the significant digits

    """
    if reference_is_random_variable:
        # The reference is a random variable
        y = y.mean(axis=axis)
    y_masked = np.ma.masked_equal(y, 0)
    e_y = np.ma.floor(np.ma.log2(np.ma.abs(y_masked)))
    return np.ma.filled(e_y, 0) + 1


def _compute_z(
    array: InternalArrayType,
    reference: Optional[InternalArrayType],
    error: Union[Error, str],
    axis: int,
    shuffle_samples: bool = False,
) -> Tuple[InternalArrayType, InternalArrayType]:
    r"""Compute Z, the distance between the random variable and the reference

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

    See Also
    --------
    significantdigits.InternalArrayType : Type used for internal computations


    """
    _assert_is_valid_inputs(array)

    nb_samples = array.shape[axis]

    if reference is None:
        # No reference provided
        # X = Y
        reference_is_random_variable = True
        if nb_samples % 2 != 0:
            error_msg = "Number of samples must be a multiple of 2"
            raise SignificantDigitsException(error_msg)
        nb_samples /= 2
        if shuffle_samples:
            np.random.shuffle(array)
        x, y = np.split(array, 2, axis=axis)
    elif reference.ndim == array.ndim:
        # X and Y have the same dimension
        # It is the case when Y is a random variable
        reference_is_random_variable = True
        x = array
        y = reference
        if shuffle_samples:
            np.random.shuffle(x)
            np.random.shuffle(y)
    elif reference.ndim == array.ndim - 1:
        # Y has one less dimension than X
        # Y is a constant
        reference_is_random_variable = False
        x = array
        y = reference
    elif reference.ndim == 0:
        # Y is a scalar value
        # Y is a constant
        reference_is_random_variable = False
        x = array
        y = reference
    else:
        raise TypeError("No comparison found for X and reference:")

    x = np.asanyarray(x)
    y = np.asanyarray(y)

    if Error.is_absolute(error):
        z = _substract_along_axis(x, y, axis=axis)
        e = _compute_scaling_factor(
            y, axis=axis, reference_is_random_variable=reference_is_random_variable
        )
    elif Error.is_relative(error):
        if np.any(y[y == 0]):
            warn_msg = "error is set to relative and the reference has 0 leading to NaN"
            warnings.warn(warn_msg)
        z = _divide_along_axis(x, y, axis=axis) - 1
        e = np.full_like(y, 1)
    else:
        raise SignificantDigitsException(f"Unknown error {error}")
    return z, e


def _significant_digits_cnh(
    array: InternalArrayType,
    reference: Optional[InternalArrayType],
    axis: int,
    error: Union[Error, str],
    probability: float,
    confidence: float,
    shuffle_samples: bool = False,
    dtype: Optional[npt.DTypeLike] = None,
) -> InternalArrayType:
    r"""Compute significant digits for Centered Normality Hypothesis (CNH)

    Parameters
    ----------
    array: InternalArrayType
        Element to compute
    reference: InternalArrayType | None
        Reference for comparing the array
    axis: int
        Axis or axes along which the significant digits are computed
    error : Error | str
        The error function to use to compute error between array and reference.
    probability : float
        Probability for the significant digits result
    confidence : float
        Confidence level for the significant digits result
    shuffle_samples : bool, default=False
        If reference is None, the array is split in two and
        comparison is done between both pieces.
        If shuffle_samples is True, it shuffles pieces.
    dtype : dtype_like | None, default=None
        Numerical type used for computing significant digits
        Widest format between array and reference is taken if not supplied.

    Returns
    -------
    ndarray
        array_like containing significant digits

    Notes
    -----
    .. [1] Sohier, D., Castro, P. D. O., Févotte, F.,
    Lathuilière, B., Petit, E., & Jamond, O. (2021).
    Confidence intervals for stochastic arithmetic.
    ACM Transactions on Mathematical Software (TOMS), 47(2), 1-33.

    .. math::

    s >= -log_2(std) - [\frac{1}{2} log_2( \frac{n-1}{ Chi^2_{1-\frac{\alpha}{2}} }) ) + log_2(F^{-1}(\frac{p+1}{2})]
    """
    z, e = _compute_z(
        array, reference, error, axis=axis, shuffle_samples=shuffle_samples
    )
    nb_samples = z.shape[axis]
    std = z.std(axis=axis, dtype=_internal_dtype)
    # if std == 0, we set it to the maximum value of z
    # to avoid returning the maximum number of bits depending on the dtype
    # while it can be lower (cf. Cramer example)
    z_eps = np.max(np.abs(z), axis=axis)
    std = _fill_where(std, fill_value=z_eps, mask=std == 0)
    # We need to mask the std where z_eps == 0
    # In that case, we have no variance and z = 0
    std0 = np.ma.array(std, mask=(z_eps == 0))
    chi2 = scipy.stats.chi2.interval(confidence, nb_samples - 1)[0]
    inorm = scipy.stats.norm.ppf((probability + 1) / 2)
    delta_chn = 0.5 * np.log2((nb_samples - 1) / chi2) + np.log2(inorm)
    significant = -np.ma.log2(std0) + (e - 1) - delta_chn
    std0 = np.ma.array(std, mask=std == 0)
    max_bits = _get_significant_size(z, dtype=dtype)
    if significant.ndim == 0:
        significant = np.ma.array(significant, mask=std0.mask)
    significant = significant.filled(fill_value=max_bits - delta_chn)
    return significant


def _significant_digits_general(
    array: InternalArrayType,
    reference: Optional[InternalArrayType],
    axis: int,
    error: Union[Error, str],
    shuffle_samples: bool = False,
    dtype: Optional[npt.DTypeLike] = None,
) -> InternalArrayType:
    r"""Compute significant digits for unknown underlying distribution

    For the general case, the probability is not parametrizable but
    can be estimated by the sample size. By setting `return_probability` to
    True, the function returns a tuple with the estimated probability
    lower bound for the given `confidence`.

    Parameters
    ----------
    array: InternalArrayType
        Element to compute
    reference: InternalArrayType | None
        Reference for comparing the array
    axis: int
        Axis or axes along which the significant digits are computed
    error : Error | str
        The error function to use to compute error between array and reference.
    shuffle_samples : bool, optional=False
        If reference is None, the array is split in two and
        comparison is done between both pieces.
        If shuffle_samples is True, it shuffles pieces.
    dtype : dtype_like | None, default=None
        Numerical type used for computing significant digits
        Widest format between array and reference is taken if not supplied.

    Returns
    -------
    out : ndarray
        array_like containing significant digits

    Notes
    -----
    .. [1] Sohier, D., Castro, P. D. O., Févotte, F.,
    Lathuilière, B., Petit, E., & Jamond, O. (2021).
    Confidence intervals for stochastic arithmetic.
    ACM Transactions on Mathematical Software (TOMS), 47(2), 1-33.

    .. math::
        s = max{k \in [1,mant], st \forall i \in [1,n], |Z_i| <= 2^{-k}}
    """
    z, e = _compute_z(
        array, reference, error, axis=axis, shuffle_samples=shuffle_samples
    )
    sample_shape = tuple(dim for i, dim in enumerate(z.shape) if i != axis)
    max_bits = _get_significant_size(z, dtype=dtype)
    significant = np.full(shape=sample_shape, fill_value=0)
    mask = np.full_like(significant, True, dtype=bool)
    z = np.abs(z)

    ic(z, reference, e)

    # Compute successes
    for k in range(0, max_bits + 1):
        ic(k, significant, mask, z.max(axis=axis))

        kth = k - (e - 1.0)
        successes = np.min(z <= 2 ** (-kth), axis=axis)
        mask = np.logical_and(mask, successes)
        significant[mask] = k

        ic(kth, 2**-kth, successes, significant, mask)

        if ~mask.all():
            break

    return significant


def significant_digits(
    array: InputType,
    reference: Optional[ReferenceType] = None,
    axis: int = 0,
    basis: int = 2,
    error: Union[str, Error] = Error.Relative,
    method: Union[str, Method] = Method.CNH,
    probability: float = _default_probability[Metric.Significant],
    confidence: float = _default_confidence[Metric.Significant],
    shuffle_samples: bool = False,
    dtype: Optional[npt.DTypeLike] = None,
) -> OutputType:
    r"""Compute significant digits

    This function calculates with a certain probability the number of
    significant bits with, in comparison to a correct reference value[1]_.

    .. math::
        \begin{array}{ll}
        & \text{Significant digits formulae for both cases} \newline
        \text{CNH:} & \hat{s}_{CNH} = -\log_2(\hat{\sigma}_Z) - \left[ \dfrac{1}{2} \log_2\left( \dfrac{n-1}{ \chi^2_{1-\alpha/2 } }\right) + \log_2 \left( F^{-1}\left( \dfrac{p+1}{2} \right) \right) \right] \newline
        \text{General:} & \hat{s}_B = \max \left\\{ k \in [0,53] \text{ s.t. } \forall i \in [1, n], |Z_i| \leq 2^{-k} \right\\}
        \end{array}

    - X = array
    - Y = reference

    Three cases:
    - Y is None
        - Divide X equally between variables 'X' and 'Y'
        - The case when X = Y
    - X.ndim == Y.ndim
        - X and Y have the same dimension
        - The case when Y is a random variable
    - X.ndim - 1 == Y.ndim or Y.ndim == 0
        - Y is a scalar value


    Parameters
    ----------
    array: InputType
        Element to compute
    reference: ReferenceType | None, optional=None
        Reference for comparing the array
    axis: int, optional=0
        Axis or axes along which the significant digits are computed
    basis: int, optional=2
        basis in which represent the significant digits
    error : Error | str, optional=Error.Relative
        The error function to use to compute error between array and reference.
    method : Method | str, optional=Method.CNH
        Method to use for the underlying distribution hypothesis
    probability : float, default=0.95
        Probability for the significant digits result
    confidence : float, default=0.95
        Confidence level for the significant digits result
    shuffle_samples : bool, optional=False
        If reference is None, the array is split in two and
        comparison is done between both pieces.
        If shuffle_samples is True, it shuffles pieces.
    dtype : dtype_like | None, default=None
        Numerical type used for computing significant digits
        Widest format between array and reference is taken if no supplied.

    Returns
    -------
    ndarray
        array_like containing significant digits

    See Also
    --------
    significantdigits.contributing_digits : Computes the contributing digits
    significantdigits.Error : Errors between random variable and reference
    significantdigits.Method : Methods for underlying distribution hypothesis
    significantdigits.InputType : get InputType types
    significantdigits.ReferenceType : get ReferenceType types

    Notes
    -----
    .. [1] Sohier, D., Castro, P. D. O., Févotte, F.,
        Lathuilière, B., Petit, E., & Jamond, O. (2021).
        Confidence intervals for stochastic arithmetic.
        ACM Transactions on Mathematical Software (TOMS), 47(2), 1-33.


    """
    _assert_is_probability(probability)
    _assert_is_confidence(confidence)
    _assert_is_valid_method(method)
    _assert_is_valid_error(error)

    significant = None

    preproc_array, preproc_reference = _preprocess_inputs(array, reference)

    if method == Method.CNH:
        significant = _significant_digits_cnh(
            array=preproc_array,
            reference=preproc_reference,
            error=error,
            probability=probability,
            confidence=confidence,
            axis=axis,
            shuffle_samples=shuffle_samples,
            dtype=dtype,
        )

    elif method == Method.General:
        significant = _significant_digits_general(
            array=preproc_array,
            reference=preproc_reference,
            error=error,
            axis=axis,
            shuffle_samples=shuffle_samples,
            dtype=dtype,
        )
    else:
        raise SignificantDigitsException(f"Unknown method {method}")

    if basis != 2:
        significant = change_basis(significant, basis)

    return significant


def _contributing_digits_cnh(
    array: InternalArrayType,
    reference: Optional[InternalArrayType],
    axis: int,
    error: Union[Error, str],
    probability: float,
    confidence: float,
    shuffle_samples: bool = False,
    dtype: Optional[npt.DTypeLike] = None,
) -> InternalArrayType:
    r"""Compute contributing digits for Centered Hypothesis Normality

    Parameters
    ----------
    array: InternalArrayType
        Element to compute
    reference: InternalArrayType | None
        Reference for comparing the array
    axis: int
        Axis or axes along which the contributing digits are computed
    error : Error | str
        The error function to use to compute error between array and reference.
    probability : float
        Probability for the contributing digits result
    confidence : float
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

    Notes
    -----
    .. [1] Sohier, D., Castro, P. D. O., Févotte, F.,
    Lathuilière, B., Petit, E., & Jamond, O. (2021).
    Confidence intervals for stochastic arithmetic.
    ACM Transactions on Mathematical Software (TOMS), 47(2), 1-33.

    .. math::
        c >= -log_2(std) - [\frac{1}{2} log_2( \frac{n-1} / \frac{ Chi^2_{1-\frac{alpha}{2}} }) ) + log_2(p+\frac{1}{2}) + log_2(2\sqrt{2\pi})]
    """
    z, e = _compute_z(
        array, reference, error, axis=axis, shuffle_samples=shuffle_samples
    )
    nb_samples = z.shape[axis]
    std = z.std(axis=axis, dtype=_internal_dtype)
    # if std == 0, we set it to the maximum value of z
    # to avoid returning the maximum number of bits depending on the dtype
    # while it can be lower (cf. Cramer example)
    z_eps = np.max(np.abs(z), axis=axis)
    std = _fill_where(std, fill_value=z_eps, mask=std == 0)
    # We need to mask the std where z_eps == 0
    # In that case, we have no variance and z = 0
    std0 = np.ma.array(std, mask=(z_eps == 0))
    chi2 = scipy.stats.chi2.interval(confidence, nb_samples - 1)[0]
    delta_chn = (
        0.5 * np.log2((nb_samples - 1) / chi2)
        + np.log2(probability - 0.5)
        + np.log2(2 * np.sqrt(2 * np.pi))
    )
    contributing = -np.ma.log2(std0) + (e - 1) - delta_chn
    max_bits = _get_significant_size(z, dtype=dtype) + (e - 1)
    if contributing.ndim == 0:
        contributing = np.ma.array(contributing, mask=std0.mask)
    contributing = np.ma.filled(contributing, fill_value=max_bits - delta_chn)
    return contributing


def _contributing_digits_general(
    array: InternalArrayType,
    reference: Optional[InternalArrayType],
    axis: int,
    error: Union[Error, str],
    shuffle_samples: bool = False,
    dtype: Optional[npt.DTypeLike] = None,
) -> InternalArrayType:
    r"""Computes contributing digits for unknown underlying distribution

    This function computes with a certain probability the number of bits
    of the mantissa that will round the result towards the correct reference
    value[1]_

    Parameters
    ----------
    array: InternalArrayType
        Element to compute
    reference: InternalArrayType | None
        Reference for comparing the array
    axis: int
        Axis or axes along which the contributing digits are computed
    error : Error | str
        The error function to use to compute error between array and reference.
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

    See Also
    --------
    significantdigits.significant_digits_cnh : Computes the significant digits under CNH
    significantdigits.compute_z : Computes the error between random variable and reference
    significantdigits.get_input_type : get InputType types
    significantdigits.get_output_type : get ReferenceType types

    Notes
    -----
    .. [1] Sohier, D., Castro, P. D. O., Févotte, F.,
    Lathuilière, B., Petit, E., & Jamond, O. (2021).
    Confidence intervals for stochastic arithmetic.
    ACM Transactions on Mathematical Software (TOMS), 47(2), 1-33.

    .. math::
        C^{i_k} = "\lfloor 2^k|Z_i|  \rfloor is even"
        c = (\frac{#success}{#trials} > p)
    """

    z, e = _compute_z(
        array, reference, error, axis=axis, shuffle_samples=shuffle_samples
    )
    sample_shape = tuple(dim for i, dim in enumerate(z.shape) if i != axis)
    max_bits = _get_significant_size(z, dtype=dtype)
    contributing = np.full(shape=sample_shape, fill_value=1)
    mask = np.full_like(contributing, True, dtype=bool)

    for k in range(1, max_bits + 1):
        kth = k - (e - 1.0)
        kth_bit_z = np.floor(np.abs(z) * 2**kth).astype(np.int64)

        successes = np.sum(np.mod(kth_bit_z, 2), axis=axis) == 0
        mask = np.logical_and(mask, successes)
        contributing[mask] = k

        if ~mask.all():
            break

    return contributing


def contributing_digits(
    array: InputType,
    reference: Optional[ReferenceType] = None,
    axis: int = 0,
    basis: int = 2,
    error: Union[str, Error] = Error.Relative,
    method: Union[str, Method] = Method.CNH,
    probability: float = _default_probability[Metric.Contributing],
    confidence: float = _default_confidence[Metric.Contributing],
    shuffle_samples: bool = False,
    dtype: Optional[npt.DTypeLike] = None,
) -> OutputType:
    r"""Compute contributing digits

    This function computes with a certain probability the number of bits
    that will round the result towards the correct reference
    value [1]_

    - X = array
    - Y = reference

    Three cases:
    - Y is None
        - Divide X equally between variables 'X' and 'Y'
        - The case when X = Y
    - X.ndim == Y.ndim
        - X and Y have the same dimension
        - The case when Y is a random variable
    - X.ndim - 1 == Y.ndim or Y.ndim == 0
        - Y is a scalar value

    .. math::
        \begin{array}{ll}
        & \text{Contributing digits formulae for both cases} \newline
        \text{CNH} & \hat{c}_{cnh} = -\log_2(\hat{\sigma}_Z) - \left[ \dfrac{1}{2} \log_2\left( \dfrac{n-1}{ \chi^2_{1-\alpha/2}} \right) + \log_2 \left (p-\dfrac{1}{2} \right) + \log_2( 2\sqrt{2\pi}) \right] \newline
        \text{General:} & \hat{c}_B = \max \left\\{ k \in [0,53] \text{ s.t. } \forall i \in [1, n], \lfloor 2^k|Z_i|  \rfloor \text{ is even } \right\\}
        \end{array}

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

    See Also
    --------
    significantdigits.significant_digits : Computes the significant digits
    significantdigits.Error : Errors between random variable and reference
    significantdigits.Method : Methods for underlying distribution hypothesis
    significantdigits.InputType : get InputType types
    significantdigits.ReferenceType : get ReferenceType types

    Notes
    -----
    .. [1] Sohier, D., Castro, P. D. O., Févotte, F.,
        Lathuilière, B., Petit, E., & Jamond, O. (2021).
        Confidence intervals for stochastic arithmetic.
        ACM Transactions on Mathematical Software (TOMS), 47(2), 1-33.

    """

    _assert_is_probability(probability)
    _assert_is_confidence(confidence)
    _assert_is_valid_method(method)
    _assert_is_valid_error(error)

    contributing = None

    array, reference = _preprocess_inputs(array, reference)

    if method == Method.CNH:
        contributing = _contributing_digits_cnh(
            array=array,
            reference=reference,
            error=error,
            axis=axis,
            probability=probability,
            confidence=confidence,
            shuffle_samples=shuffle_samples,
            dtype=dtype,
        )

    elif method == Method.General:
        contributing = _contributing_digits_general(
            array=array,
            reference=reference,
            error=error,
            axis=axis,
            shuffle_samples=shuffle_samples,
            dtype=dtype,
        )
    else:
        raise SignificantDigitsException(f"Unknown method {method}")

    if basis != 2:
        contributing = change_basis(contributing, basis)

    return contributing


def probability_estimation_bernoulli(
    success: int, trials: int, confidence: float
) -> float:
    r"""Computes probability lower bound for Bernoulli process

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
        The lower bound probability with `confidence` level to have `success`
        successes for `trials` trials


    Notes
    -----
    .. [1] Sohier, D., Castro, P. D. O., Févotte, F.,
    Lathuilière, B., Petit, E., & Jamond, O. (2021).
    Confidence intervals for stochastic arithmetic.
    ACM Transactions on Mathematical Software (TOMS), 47(2), 1-33.

    .. math::
        p =
        \begin{cases}
            1 + \frac{\log{ 1 - \alpha }}{n} & \text{if s=n} \newline
            \frac{s+2}{s+4} - F^{-1}(\frac{p+1}{2}) \sqrt{ \frac{(s+2)(n-s+2)}{n+4}^3  } & \text{else}
        \end{cases}

    """
    _assert_is_confidence(confidence)

    s = success
    n = trials
    coef = scipy.stats.norm.ppf(confidence)

    if s == n:
        # Special case when having only successes
        probability = 1 + np.log(1 - confidence) / n
    else:
        probability = (s + 2) / (n + 4) - coef * np.sqrt(
            (s + 2) * (n - s + 2) / (n + 4) ** 3
        )

    return probability


def minimum_number_of_trials(probability: float, confidence: float) -> int:
    r"""Computes the minimum number of trials to have probability and confidence

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

    Notes
    -----
    .. [1] Sohier, D., Castro, P. D. O., Févotte, F.,
    Lathuilière, B., Petit, E., & Jamond, O. (2021).
    Confidence intervals for stochastic arithmetic.
    ACM Transactions on Mathematical Software (TOMS), 47(2), 1-33.

    .. math::
        n = \left \lceil \frac{\ln{\alpha}}{\ln{p}} \right \rceil

    """
    _assert_is_probability(probability)
    _assert_is_confidence(confidence)

    alpha = 1 - confidence
    n = np.log(alpha) / np.log(probability)
    return int(np.ceil(n))


def format_uncertainty(
    array: InputType,
    reference: Optional[ReferenceType] = None,
    axis: int = 0,
    error: Union[str, Error] = Error.Relative,
    method: Union[str, Method] = Method.CNH,
    probability: float = _default_probability[Metric.Contributing],
    confidence: float = _default_confidence[Metric.Contributing],
    shuffle_samples: bool = False,
    dtype: Optional[npt.DTypeLike] = None,
    as_tuple: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    r"""Format an array with its significant and contributing digits.

    This function computes and formats each element of the input array
    to display its value along with its uncertainty, based on the calculated
    significant and contributing digits. The output provides a human-readable
    representation of numerical precision, using the appropriate number of
    digits and error notation.

    Parameters
    ----------
    array : InputType
        The array of values to format.
    reference : ReferenceType or None, optional
        The reference values for error computation. If None, the array is split
        and compared internally.
    axis : int, default=0
        Axis along which the digits are computed.
    error : Error or str, default=Error.Relative
        The error metric to use ('absolute' or 'relative').
    method : Method or str, default=Method.CNH
        The statistical method for digit estimation.
    probability : float, default=0.51
        Probability for the contributing digits result.
    confidence : float, default=0.95
        Confidence level for the digits result.
    shuffle_samples : bool, default=False
        Whether to shuffle samples when splitting the array.
    dtype : dtype_like or None, default=None
        Data type used for computation.
    as_tuple : bool, default=False
        If True, returns a tuple of value and error.
        If False, returns a formatted string for each element.

    Returns
    -------
    np.ndarray
        An array of formatted strings, each showing the value and its uncertainty.
    or
    Tuple[np.ndarray, np.ndarray]
        If `as_tuple` is True, returns a tuple containing two arrays:
        the first with formatted values and the second with formatted errors.

    Notes
    -----
    The significant and contributing digits are computed according to the
    selected error metric and statistical method. The formatted output
    displays each value with its uncertainty, using either absolute or
    relative error notation, e.g.:
        1.234 ± 0.005
    or
        1.234 ± 0.4e-2

    For absolute error:
        The uncertainty is shown as ± 2^{-s}, where s is the number of significant digits.
    For relative error:
        The uncertainty is shown as ± y·2^{-s}, where y is the reference value.

    References
    ----------
    - Sohier, D., Castro, P. D. O., Févotte, F., Lathuilière, B., Petit, E., & Jamond, O. (2021).
      Confidence intervals for stochastic arithmetic. ACM Transactions on Mathematical Software (TOMS), 47(2), 1-33.
    """

    """Format the array with significant and contributing digits

    """

    sd = significant_digits(
        array,
        reference=reference,
        axis=axis,
        basis=2,
        error=error,
        method=method,
        probability=probability,
        confidence=confidence,
        shuffle_samples=shuffle_samples,
        dtype=dtype,
    )
    cd = contributing_digits(
        array,
        reference=reference,
        axis=axis,
        basis=10,
        error=error,
        method=method,
        probability=probability,
        confidence=confidence,
        shuffle_samples=shuffle_samples,
        dtype=dtype,
    )

    # Format the array with significant and contributing digits
    # sd = np.floor(sd)
    cd = np.ceil(cd)

    ic(sd, cd)

    def print_significant_absolute(x, c):
        return np.format_float_positional(
            x,
            precision=max(0, int(c)),
            min_digits=max(0, int(c)),
            unique=True,
            fractional=False,
            trim="k",
            sign=True,
        )

    def print_error_absolute(x, c):
        return np.format_float_scientific(
            x,
            precision=max(0, int(c)),
            unique=False,
            trim="0",
            sign=False,
        )

    def print_significant_relative(x, c):
        return np.format_float_positional(
            x,
            precision=max(0, int(c)),
            min_digits=max(0, int(c)),
            unique=True,
            fractional=False,
            trim="k",
            sign=True,
        )

    def print_error_relative(x, c):
        return np.format_float_scientific(
            x, precision=max(0, int(c)), unique=False, trim="0", sign=False
        )

    # Create output array with same shape as input
    array, _ = _preprocess_inputs(array, reference)
    formatted_array = np.empty(array.shape, dtype=object)
    value_array = np.empty(array.shape, dtype=_internal_dtype)
    error_array = np.empty(array.shape, dtype=_internal_dtype)

    if Error.is_absolute(error):
        # Use nditer to iterate over all elements while preserving multi-dimensional structure
        with np.nditer(
            [array, sd, cd, formatted_array],
            flags=["multi_index", "refs_ok"],
            op_flags=[["readonly"], ["readonly"], ["readonly"], ["writeonly"]],
        ) as it:
            for x_val, sd_val, cd_val, out_val in it:
                idx = it.multi_index
                significant_str = print_significant_absolute(
                    x_val.item(), cd_val.item()
                )
                error_str = print_error_absolute(2 ** -sd_val.item(), cd_val.item())
                if as_tuple:
                    value_array[idx] = significant_str
                    error_array[idx] = error_str
                else:
                    formatted_array[idx] = f"{significant_str} ± {error_str}"

    elif Error.is_relative(error):
        if reference is None:
            reference = array  # Use array itself as reference if none provided

        reference = np.asarray(reference)

        # Use nditer for relative error formatting
        with np.nditer(
            [array, reference, sd, cd, formatted_array],
            flags=["multi_index", "refs_ok"],
            op_flags=[
                ["readonly"],
                ["readonly"],
                ["readonly"],
                ["readonly"],
                ["writeonly"],
            ],
        ) as it:
            for x_val, ref_val, sd_val, cd_val, out_val in it:
                idx = it.multi_index
                significant_str = print_significant_relative(
                    x_val.item(), cd_val.item()
                )
                error_str = print_error_relative(
                    ref_val.item() * (2 ** -sd_val.item()), cd_val.item()
                )
                if as_tuple:
                    value_array[idx] = significant_str
                    error_array[idx] = error_str
                else:
                    formatted_array[idx] = f"{significant_str} ± {error_str}"
    else:
        raise SignificantDigitsException(f"Unknown error {error}")

    if as_tuple:
        # Return a tuple of value and error arrays
        return value_array, error_array
    else:
        return formatted_array
