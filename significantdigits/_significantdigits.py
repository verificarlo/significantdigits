import warnings
from enum import Enum, auto
from typing import Optional, Tuple, TypeVar, Union

import numpy as np
import scipy
import scipy.stats


class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name


class Metric(AutoName):
    """
    Significant: Compute the number of significant digits
    Contributing: Compute the number of contributing digits
    """

    Significant = auto()
    Contributing = auto()

    @classmethod
    def is_significant(cls, error):
        if isinstance(error, cls):
            return error == cls.Significant
        if isinstance(error, str):
            return error.lower() == cls.Significant.name

    @classmethod
    def is_contributing(cls, error):
        if isinstance(error, cls):
            return error == cls.Contributing
        if isinstance(error, str):
            return error.lower() == cls.Contributing.name


class Method(AutoName):
    """
    CNH: Centered Normality Hypothesis
         X follows a Gaussian law centered around the reference or
         Z follows a Gaussian law centered around 0
    General: No assumption about the distribution of X or Z
    """

    CNH = auto()
    General = auto()

    @classmethod
    def is_cnh(cls, error):
        if isinstance(error, cls):
            return error == cls.CNH
        if isinstance(error, str):
            return error.lower() == cls.CNH.name

    @classmethod
    def is_general(cls, error):
        if isinstance(error, cls):
            return error == cls.General
        if isinstance(error, str):
            return error.lower() == cls.General.name


class Error(AutoName):
    """
    ----------+-------------+-------------
              | Reference x | Reference Y
    ----------+-------------+-------------
    Absolute  | Z = X - x   | Z = X - Y
    Relative  | Z = X/x - 1 | Z = X/Y - 1
    ----------+-------------+-------------
    """

    Absolute = auto()
    Relative = auto()

    @classmethod
    def is_absolute(cls, error):
        if isinstance(error, cls):
            return error == cls.Absolute
        if isinstance(error, str):
            return error.lower() == cls.Absolute.name

    @classmethod
    def is_relative(cls, error):
        if isinstance(error, cls):
            return error == cls.Relative
        if isinstance(error, str):
            return error.lower() == cls.Relative.name


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

internal_dtype = np.dtype(np.float64)
default_probability = {Metric.Significant: 0.95, Metric.Contributing: 0.51}
default_confidence = {Metric.Significant: 0.95, Metric.Contributing: 0.95}

InputType = TypeVar("InputType", np.ndarray, tuple, list)
r"""Valid random variable inputs type (np.ndarray, tuple, list)

Types allowing for `array` in significant_digits and contributing_digits functions

"""

ReferenceType = TypeVar(
    "ReferenceType", np.ndarray, tuple, list, float, int, np.floating
)
r"""Valid reference inputs type (np.ndarray, tuple, list, float, int)

Types allowing for `reference` in significant_digits and contributing_digits functions

"""


def get_input_type():
    r"""Returns InputType subtypes"""
    return InputType.__constraints__


def get_reference_type():
    r"""Returns ReferenceType subtypes"""
    return ReferenceType.__constraints__


def assert_is_valid_metric(metric: Union[Metric, str]) -> None:
    if Metric.is_significant(metric) or Metric.is_contributing(metric):
        return

    raise TypeError(
        f"provided invalid metric {metric}: " f"must be one of {list(Metric)}"
    )


def assert_is_valid_method(method: Union[Method, str]) -> None:
    if Method.is_cnh(method) or Method.is_general(method):
        return

    raise TypeError(
        f"provided invalid method {method}: " f"must be one of {list(Method)}"
    )


def assert_is_valid_error(error: Union[Error, str]) -> None:
    if Error.is_absolute(error) or Error.is_relative(error):
        return

    raise TypeError(f"provided invalid error {error}: " f"must be one of {list(Error)}")


def assert_is_probability(probability: float) -> None:
    if probability < 0 or probability > 1:
        raise TypeError("probability must be between 0 and 1")


def assert_is_confidence(confidence: float) -> None:
    if confidence < 0 or confidence > 1:
        raise TypeError("confidence must be between 0 and 1")


def change_basis(array: InputType, base: int) -> InputType:
    """Changes basis from binary to `base` representation

    Parameters
    ----------
    array : np.ndarray
        array_like containing significant or contributing bits
    basis : int
        output basis

    Returns
    -------
    np.ndarray
        Array convert to base `base`
    """
    pow2 = np.power(2, array, dtype=np.float64)
    array_masked = np.ma.array(pow2, mask=array <= 0)
    return np.emath.logn(base, array_masked).astype(np.int8)


def preprocess_inputs(
    array: Union[np.ndarray, tuple, list], reference: Union[np.ndarray, tuple, list]
) -> Tuple[np.ndarray, np.ndarray]:
    if scipy.sparse.issparse(array[0]):
        array = np.asanyarray([i.toarray() for i in array])

    if not isinstance(array, InputType.__constraints__):
        raise TypeError(
            f"Input array must be "
            f"one of the following types "
            f"{InputType.__constraints__}"
        )

    if not isinstance(array, np.ndarray):
        array = np.array(array)

    if reference is not None:
        if scipy.sparse.issparse(reference):
            reference = reference.toarray()
        if not isinstance(reference, ReferenceType.__constraints__):
            raise TypeError(
                f"Reference must be "
                f"one of the following types "
                f"{ReferenceType.__constraints__}"
            )

        if not isinstance(reference, np.ndarray):
            reference = np.array(reference)

    return (array, reference)


def compute_z(
    array: InputType,
    reference: Optional[ReferenceType],
    error: Union[Error, str],
    axis: int,
    shuffle_samples: bool = False,
) -> InputType:
    r"""Compute Z, the distance between the random variable and the reference

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
        The error function to use to compute error between array and reference.
    axis : int, default=0
        The axis or axes along which compute Z
    shuflle_samples : bool, default=False
        If True, shuffles the groups when the reference is None

    Returns
    -------
    array : numpy.ndarray
        The result of Z following the error method choose

    See Also
    --------
    significantdigits.InputType : Type for random variable
    significantdigits.ReferenceType : Type for reference


    """
    nb_samples = array.shape[axis]

    if reference is None:
        if nb_samples % 2 != 0:
            error_msg = ("Number of samples must be ", "a multiple of 2")
            raise Exception(error_msg)
        nb_samples /= 2
        if shuffle_samples:
            np.random.shuffle(array)
        x, y = np.split(array, 2, axis=axis)
    elif reference.ndim == array.ndim:
        x = array
        y = reference
        if shuffle_samples:
            np.random.shuffle(x)
            np.random.shuffle(y)
    elif reference.ndim == array.ndim - 1:
        x = array
        y = reference
    elif reference.ndim == 0:
        x = array
        y = reference
    else:
        raise TypeError("No comparison found for X and reference:")

    x = np.array(x)
    y = np.array(y)

    if Error.is_absolute(error):
        z = x - y
    elif Error.is_relative(error):
        if np.any(y[y == 0]):
            warn_msg = "error is set to relative and the reference " "0 leading to NaN"
            warnings.warn(warn_msg)
        z = x / y - 1
    else:
        raise Exception(f"Unknown error {error}")
    return z


def significant_digits_cnh(
    array: InputType,
    reference: Optional[ReferenceType],
    axis: int,
    error: Union[Error, str],
    probability: float,
    confidence: float,
    shuffle_samples: bool = False,
    dtype: Optional[np.dtype] = None,
) -> InputType:
    r"""Compute significant digits for Centered Normality Hypothesis (CNH)

    Parameters
    ----------
    array: InputType
        Element to compute
    reference: Optional[ReferenceType]
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
    dtype : np.dtype, default=None
        Numerical type used for computing contributing digits
        Widest format between array and reference is taken if no supplied.

    Returns
    -------
    ndarray
        array_like containing contributing digits

    See Also
    --------
    significantdigits.contributing_digits_general : Computes the contributing digits in general case
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
        s >= -log_2(std) - [\frac{1}{2} log_2( \frac{n-1}{ Chi^2_{1-\frac{\alpha}{2}} }) ) + log_2(F^{-1}(\frac{p+1}{2})]
    """
    z = compute_z(array, reference, error, axis=axis, shuffle_samples=shuffle_samples)
    nb_samples = z.shape[axis]
    std = np.std(z, axis=axis, dtype=internal_dtype)
    std0 = np.ma.array(std, mask=std == 0)
    chi2 = scipy.stats.chi2.interval(confidence, nb_samples - 1)[0]
    inorm = scipy.stats.norm.ppf((probability + 1) / 2)
    delta_chn = 0.5 * np.log2((nb_samples - 1) / chi2) + np.log2(inorm)
    significant = -1 * (np.ma.log2(std0) + delta_chn)
    max_bits = np.finfo(dtype if dtype else z.dtype).nmant
    if significant.ndim == 0:
        significant = np.ma.array(significant, mask=std0.mask)
    significant = significant.filled(fill_value=max_bits - delta_chn)
    return significant


def significant_digits_general(
    array: InputType,
    reference: Optional[ReferenceType],
    axis: int,
    error: Union[Error, str],
    shuffle_samples: bool = False,
    dtype: Optional[np.dtype] = None,
) -> InputType:
    r"""Compute significant digits for unknown underlying distribution

    For the general case, the probability is not parametrizable but
    can be estimated by the sample size. By setting `return_probability` to
    True, the function returns a tuple with the estimated probability
    lower bound for the given `confidence`.

    Parameters
    ----------
    array: InputType
        Element to compute
    reference: Optional[ReferenceType]
        Reference for comparing the array
    axis: int
        Axis or axes along which the significant digits are computed
    error : Error | str
        The error function to use to compute error between array and reference.
    shuffle_samples : bool, optional=False
        If reference is None, the array is split in two and
        comparison is done between both pieces.
        If shuffle_samples is True, it shuffles pieces.
    dtype : np.dtype, default=None
        Numerical type used for computing contributing digits
        Widest format between array and reference is taken if no supplied.

    Returns
    -------
    out : ndarray
        array_like containing contributing digits

    See Also
    --------
    significantdigits.significant_digits_cnh : Computes the contributing digits under CNH
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
        s = max{k \in [1,mant], st \forall i \in [1,n], |Z_i| <= 2^{-k}}
    """
    z = compute_z(array, reference, error, axis=axis, shuffle_samples=shuffle_samples)

    sample_shape = tuple(dim for i, dim in enumerate(z.shape) if i != axis)
    max_bits = np.finfo(dtype if dtype else z.dtype).nmant
    significant = np.ma.MaskedArray(
        data=np.full(shape=sample_shape, fill_value=0, dtype=np.int8), mask=False
    )
    zz = np.ma.array(np.abs(z), mask=np.abs(z) <= 0, fill_value=max_bits)
    if np.all(zz.mask):
        return zz.filled()

    z2 = np.ma.log2(zz)

    # Compute successes
    for k in range(0, max_bits + 1):
        # min(bool) <=> logical and
        successes = np.ma.min(z2 <= -k, axis=axis)
        significant.mask |= np.ma.logical_not(successes)
        significant[np.logical_not(significant.mask)] = k
        if np.all(significant.mask):
            break

    return significant.data


def significant_digits(
    array: InputType,
    reference: Optional[ReferenceType] = None,
    axis: int = 0,
    base: int = 2,
    error: Union[str, Error] = Error.Relative,
    method: Union[str, Method] = Method.CNH,
    probability: float = default_probability[Metric.Significant],
    confidence: float = default_confidence[Metric.Significant],
    shuffle_samples: bool = False,
    dtype: Optional[np.dtype] = None,
) -> InputType:
    r"""Compute significant digits

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
        The error function to use to compute error between array and reference.
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

    See Also
    --------
    significantdigits.contributing_digits : Computes the contributing digits
    significantdigits.compute_z : Computes the error between random variable and reference
    significantdigits.get_input_type : get InputType types
    significantdigits.get_output_type : get ReferenceType types

    Notes
    -----
    .. [1] Sohier, D., Castro, P. D. O., Févotte, F.,
    Lathuilière, B., Petit, E., & Jamond, O. (2021).
    Confidence intervals for stochastic arithmetic.
    ACM Transactions on Mathematical Software (TOMS), 47(2), 1-33.

    """
    assert_is_probability(probability)
    assert_is_confidence(confidence)
    assert_is_valid_method(method)
    assert_is_valid_error(error)

    significant = None

    array, reference = preprocess_inputs(array, reference)

    if method == Method.CNH:
        significant = significant_digits_cnh(
            array=array,
            reference=reference,
            error=error,
            probability=probability,
            confidence=confidence,
            axis=axis,
            shuffle_samples=shuffle_samples,
            dtype=dtype,
        )

    elif method == Method.General:
        significant = significant_digits_general(
            array=array,
            reference=reference,
            error=error,
            axis=axis,
            shuffle_samples=shuffle_samples,
            dtype=dtype,
        )

    if base != 2:
        significant = change_basis(significant, base)

    return significant


def contributing_digits_cnh(
    array: InputType,
    reference: Optional[ReferenceType],
    axis: int,
    error: Union[Error, str],
    probability: float,
    confidence: float,
    shuffle_samples: bool = False,
    dtype: Optional[np.dtype] = None,
) -> InputType:
    r"""Compute contributing digits for Centered Hypothesis Normality

    Parameters
    ----------
    array: InputType
        Element to compute
    reference: Optional[ReferenceType]
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
    dtype : np.dtype, default=None
        Numerical type used for computing contributing digits
        Widest format between array and reference is taken if no supplied.

    Returns
    -------
    ndarray
        array_like containing contributing digits

    See Also
    --------
    significantdigits.significant_general : Computes the significant digits for general case
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
        c >= -log_2(std) - [\frac{1}{2} log_2( \frac{n-1} / \frac{ Chi^2_{1-\frac{alpha}{2}} }) ) + log_2(p+\frac{1}{2}) + log_2(2\sqrt{2\pi})]
    """
    z = compute_z(array, reference, error, axis=axis, shuffle_samples=shuffle_samples)
    nb_samples = z.shape[axis]
    std = np.std(z, axis=axis, dtype=internal_dtype)
    std0 = np.ma.masked_array(std, mask=std == 0)
    chi2 = scipy.stats.chi2.interval(confidence, nb_samples - 1)[0]
    delta_chn = (
        0.5 * np.log2((nb_samples - 1) / chi2)
        + np.log2(probability - 0.5)
        + np.log2(2 * np.sqrt(2 * np.pi))
    )
    contributing = -np.ma.log2(std0) - delta_chn
    max_bits = np.finfo(dtype if dtype else z.dtype).nmant
    contributing = contributing.filled(fill_value=max_bits - delta_chn)
    return contributing


def contributing_digits_general(
    array: InputType,
    reference: Optional[ReferenceType],
    axis: int,
    error: Union[Error, str],
    probability: float,
    shuffle_samples: bool = False,
    dtype: Optional[np.dtype] = None,
) -> InputType:
    r"""Computes contributing digits for unknown underlying distribution

    This function computes with a certain probability the number of bits
    of the mantissa that will round the result towards the correct reference
    value[1]_

    Parameters
    ----------
    array: InputType
        Element to compute
    reference: Optional[ReferenceType]
        Reference for comparing the array
    axis: int
        Axis or axes along which the contributing digits are computed
    error : Error | str
        The error function to use to compute error between array and reference.
    probability : float
        Probability for the contributing digits result
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

    z = compute_z(array, reference, error, axis=axis, shuffle_samples=shuffle_samples)
    sample_shape = tuple(dim for i, dim in enumerate(z.shape) if i != axis)
    max_bits = np.finfo(dtype if dtype else z.dtype).nmant
    contributing = np.ma.MaskedArray(
        data=np.full(shape=sample_shape, fill_value=1, dtype=np.int8), mask=False
    )

    for k in range(1, max_bits + 1):
        # scale = ldexp(x,n) = x * 2^n
        # floor(scale) & 1 : returns 1 if scale is even
        # taking the max to check if at least one result is even
        # Get the negation to have success as boolean
        successes = np.logical_not(
            np.max(
                np.bitwise_and(np.floor(np.abs(np.ldexp(z, k))).astype(np.int64), 1),
                axis=axis,
            )
        )
        contributing.mask |= np.ma.logical_not(successes)
        contributing[np.logical_not(contributing.mask)] = k
        if np.all(contributing.mask):
            break

    return contributing.data


def contributing_digits(
    array: InputType,
    reference: Optional[ReferenceType] = None,
    axis: int = 0,
    base: int = 2,
    error: Union[str, Error] = Error.Relative,
    method: Union[str, Method] = Method.CNH,
    probability: float = default_probability[Metric.Contributing],
    confidence: float = default_confidence[Metric.Contributing],
    shuffle_samples: bool = False,
    dtype: Optional[np.dtype] = None,
) -> InputType:
    r"""Compute contributing digits


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

    See Also
    --------
    significantdigits.significant_digits : Computes the significant digits
    significantdigits.compute_z : Computes the error between random variable and reference
    significantdigits.get_input_type : get InputType types
    significantdigits.get_output_type : get ReferenceType types

    Notes
    -----
    .. [1] Sohier, D., Castro, P. D. O., Févotte, F.,
    Lathuilière, B., Petit, E., & Jamond, O. (2021).
    Confidence intervals for stochastic arithmetic.
    ACM Transactions on Mathematical Software (TOMS), 47(2), 1-33.

    """

    assert_is_probability(probability)
    assert_is_confidence(confidence)
    assert_is_valid_method(method)
    assert_is_valid_error(error)

    contributing = None

    array, reference = preprocess_inputs(array, reference)

    if method == Method.CNH:
        contributing = contributing_digits_cnh(
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
        contributing = contributing_digits_general(
            array=array,
            reference=reference,
            error=error,
            axis=axis,
            probability=probability,
            shuffle_samples=shuffle_samples,
            dtype=dtype,
        )

    if base != 2:
        contributing = change_basis(contributing, base)

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
        p = 1 + \frac{\log{ 1 - \alpha }}{n} if success=sample\_size
        p = \frac{s+2}{s+4} - F^{-1}(\frac{p+1}{2}) \sqrt{ \frac{(s+2)(n-s+2)}{n+4}^3  } else
    """
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
        n = \lceil \frac{\ln{\alpha}}{\ln{p}}

    """
    alpha = 1 - confidence
    n = np.log(alpha) / np.log(probability)
    return int(np.ceil(n))
