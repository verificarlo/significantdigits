
from typing import TypeVar
from typing import Tuple
import math
import warnings
from enum import Enum, auto
from typing import List, Optional

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


class Method(AutoName):
    """
    CNH: Centered Normality Hypothesis
         X follows a Gaussian law centered around the reference or
         Z follows a Gaussian law centered around 0
    General: No assumption about the distribution of X or Z
    """
    CNH = auto()
    General = auto()


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


_Metric_names = vars(Metric)['_member_names_']
_Metric_map = vars(Metric)['_value2member_map_']

_Method_names = vars(Method)['_member_names_']
_Method_map = vars(Method)['_value2member_map_']

_Error_names = vars(Error)['_member_names_']
_Error_map = vars(Error)['_value2member_map_']

internal_dtype = np.dtype(np.float64)
default_probability = {Metric.Significant: 0.95,
                       Metric.Contributing: 0.51}
default_confidence = {Metric.Significant: 0.95,
                      Metric.Contributing: 0.95}

InputType = TypeVar('InputType', np.ndarray, tuple, list)
ReferenceType = TypeVar(
    'ReferenceType', np.ndarray | tuple | list | float)
_valid_input_types = (tuple, list, np.ndarray)
_valid_reference_types = (float, tuple, list, np.ndarray)


def assert_is_valid_method(method: Method) -> None:
    if method not in Method:
        raise TypeError(
            f"provided invalid method {method}: must be one of {list(Method)}")


def assert_is_probability(probability: float) -> None:
    if probability < 0 or probability > 1:
        raise TypeError("probability must be between 0 and 1")


def assert_is_confidence(confidence: float) -> None:
    if confidence < 0 or confidence > 1:
        raise TypeError("confidence must be between 0 and 1")


def change_base(sig: InputType, base: int) -> InputType:
    sig_power2 = np.power(2, sig)

    def to_base(x):
        return math.log(x, base)
    np_to_base = np.frompyfunc(to_base, 1, 1)
    x = np_to_base(sig_power2)
    return x


def preprocess_inputs(array: np.ndarray | tuple | list,
                      reference: np.ndarray | tuple | list
                      ) -> Tuple[np.ndarray, np.ndarray]:

    if scipy.sparse.issparse(array[0]):
        array = np.asanyarray([i.toarray() for i in array])

    if not isinstance(array, _valid_input_types):
        raise TypeError(f'Input array must be '
                        f'one of the following types '
                        f'{_valid_input_types}')

    if not isinstance(array, np.ndarray):
        array = np.array(array)

    if reference is not None:
        if scipy.sparse.issparse(reference):
            reference = reference.toarray()
        if not isinstance(reference, _valid_reference_types):
            raise TypeError(f'Reference must be '
                            f'one of the following types '
                            f'{_valid_reference_types}')

        if not isinstance(reference, np.ndarray):
            reference = np.array(reference)

    return (array, reference)


def compute_z(array: InputType,
              reference: ReferenceType,
              error: Error,
              axis: int = 0,
              shuffle_samples: bool = False) -> InputType:
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
        - X.ndim - 1 == Y.ndim
            Y is a scalar value

    Parameters
    ----------
    array : numpy.ndarray
        The random variable
    reference : None | float | numpy.ndarray
        The reference to compare against
    error : Method.Error | str
        The error function to compute Z
    axis : int
        The axis or axes along which compute Z
        default: 0
    shuflle_samples : bool
        If True, shuffles the groups when the reference is None

    Returns
    -------
    array : numpy.ndarray
        The result of Z following the error method choose

    """
    nb_samples = array.shape[axis]

    if reference is None:
        if nb_samples % 2 != 0:
            error_msg = ("Number of samples must be ",
                         "a multiple of 2")
            raise Exception(error_msg)
        nb_samples /= 2
        if shuffle_samples:
            np.random.shuffle(array)
        x, y = np.split(array, 2)
    elif reference.ndim == array.ndim:
        x = array
        y = reference
        if shuffle_samples:
            np.random.shuffle(x)
            np.random.shuffle(y)
    elif reference.ndim == array.ndim - 1:
        x = array
        y = reference
    else:
        raise TypeError("No comparison found for X and reference:")

    if error == Error.Absolute or error.lower() == Error.Absolute.name.lower():
        z = x - y
    elif error == Error.Relative or error.lower() == Error.Relative.name.lower():
        if np.any(y[y == 0]):
            warn_msg = ('error is set to relative and the reference '
                        '0 leading to NaN')
            warnings.warn(warn_msg)
        z = x / y - 1
    else:
        raise Exception(f"Unknown error {error}")

    return z


def significant_digits_cnh(array: InputType,
                           reference: ReferenceType,
                           error: Error,
                           probability: float,
                           confidence: float,
                           axis: int = 0,
                           shuffle_samples: bool = False) -> InputType:
    r'''Compute significant digits for Centered Normality Hypothesis (CNH)

    Parameters
    ----------
    array: numpy.ndarray
        Element to compute
    reference: Optional[float|numpy.ndarray]
        Reference for comparing the array
    base: int
        Base in which represent the significant digits
    axis: int | tuple(int)
        Axis or axes along which the significant digits are computed
        default: None
    error : Error | str
        Name of the error function to use to compute Z
        default: Error.Relative
    method : Method | str
        Name of the method for the underlying distribution hypothesis
        default: Method.CNH (Centered Normality Hypothesis)
    probability : float
        Probability for the significant digits result
        default: 0.95
    confidence : float
        Confidence level for the significant digits result
        default: 0.95
    shuffle_samples : bool
        If reference is None, the array is split in two and
        comparison is done between both pieces.
        If shuffle_samples is True, it shuffles pieces.

    Returns
    -------
    ndarray
        array_like containing contributing digits

    See Also
    --------
    significantdigits.contributing_digits_general : Computes the contributing digits in general case
    significantdigits.compute_z : Computes the error between random variable and reference

    Notes
    -----
    .. [1] Sohier, D., Castro, P. D. O., Févotte, F.,
    Lathuilière, B., Petit, E., & Jamond, O. (2021).
    Confidence intervals for stochastic arithmetic.
    ACM Transactions on Mathematical Software (TOMS), 47(2), 1-33.

    .. math::
        s >= -log_2(std) - [\frac{1}{2} log_2( \frac{n-1}{ Chi^2_{1-\frac{\alpha}{2}} }) ) + log_2(F^{-1}(\frac{p+1}{2})]
    '''
    z = compute_z(array, reference, error, axis=axis,
                  shuffle_samples=shuffle_samples)
    nb_samples = z.shape[axis]
    std = np.std(z, axis=axis, dtype=internal_dtype)
    std0 = np.ma.masked_array(std == 0)
    chi2 = scipy.stats.chi2.interval(confidence, nb_samples - 1)[0]
    inorm = scipy.stats.norm.ppf((probability + 1) / 2)
    delta_chn = 0.5 * np.log2((nb_samples - 1) / chi2) + np.log2(inorm)
    significant = -np.log2(std) - delta_chn
    if significant.ndim != 0:
        significant[std0] = np.finfo(z.dtype).nmant - delta_chn
    elif std0:
        significant = np.finfo(z.dtype).nmant - delta_chn
    return significant


def significant_digits_general(array: InputType,
                               reference: ReferenceType,
                               error: Error,
                               axis: int = 0,
                               shuffle_samples: bool = False) -> InputType:
    r'''Compute significant digits for unknown underlying distribution

    For the general case, the probability is not parametrizable but
    can be estimated by the sample size. By setting `return_probability` to
    True, the function returns a tuple with the estimated probability
    lower bound for the given `confidence`.

    Parameters
    ----------
    array: numpy.ndarray
        Element to compute
    reference: Optional[float|numpy.ndarray]
        Reference for comparing the array
    base: int
        Base in which represent the significant digits
    axis: int | tuple(int)
        Axis or axes along which the significant digits are computed
        default: None
    error : Error | str
        Name of the error function to use to compute Z
        default: Error.Relative
    method : Method | str
        Name of the method for the underlying distribution hypothesis
        default: Method.CNH (Centered Normality Hypothesis)
    shuffle_samples : bool
        If reference is None, the array is split in two and
        comparison is done between both pieces.
        If shuffle_samples is True, it shuffles pieces.

    Returns
    -------
    out : ndarray
        array_like containing contributing digits

    See Also
    --------
    significantdigits.significant_digits_cnh : Computes the contributing digits under CNH
    significantdigits.compute_z : Computes the error between random variable and reference

    Notes
    -----
    .. [1] Sohier, D., Castro, P. D. O., Févotte, F.,
    Lathuilière, B., Petit, E., & Jamond, O. (2021).
    Confidence intervals for stochastic arithmetic.
    ACM Transactions on Mathematical Software (TOMS), 47(2), 1-33.

    .. math::
        s = max{k \in [1,mant], st \forall i \in [1,n], |Z_i| <= 2^{-k}}
    '''
    z = compute_z(array, reference, error, axis=axis,
                  shuffle_samples=shuffle_samples)

    sample_shape = tuple(dim for i, dim in enumerate(z.shape) if i != axis)
    max_bits = np.finfo(z.dtype).nmant
    significant = np.full(sample_shape, max_bits, dtype=np.float64)
    z_mask = np.full(sample_shape, False)
    for k in range(max_bits, -1, -1):
        pow2minusk = np.power(2, -np.float64(k))
        successess = np.abs(z) <= pow2minusk
        _z = np.all(successess, axis=axis)
        if z.ndim == 0 and _z:
            significant = k
            break

        z_mask = np.ma.masked_array(data=_z, mask=_z)
        if np.all(_z):
            break

        significant[~z_mask] = k

    return significant


def significant_digits(array: InputType,
                       reference: Optional[ReferenceType] = None,
                       axis: int = 0,
                       base: int = 2,
                       error: Error = Error.Relative,
                       method: Method = Method.CNH,
                       probability: float = default_probability[Metric.Significant],
                       confidence: float = default_confidence[Metric.Significant],
                       shuffle_samples: bool = False) -> InputType:
    r'''Compute significant digits

    Parameters
    ----------
    array: numpy.ndarray
        Element to compute
    reference: Optional[float|numpy.ndarray]
        Reference for comparing the array
    base: int
        Base in which represent the significant digits
    axis: int | tuple(int)
        Axis or axes along which the significant digits are computed
        default: None
    error : Error | str
        Name of the error function to use to compute Z
        default: Error.Relative
    method : Method | str
        Name of the method for the underlying distribution hypothesis
        default: Method.CNH (Centered Normality Hypothesis)
    probability : float
        Probability for the significant digits result
        default: 0.95
    confidence : float
        Confidence level for the significant digits result
        default: 0.95
    shuffle_samples : bool
        If reference is None, the array is split in two and
        comparison is done between both pieces.
        If shuffle_samples is True, it shuffles pieces.

    Returns
    -------
    ndarray
        array_like containing contributing digits

    See Also
    --------
    significantdigits.contributing_digits : Computes the contributing digits
    significantdigits.compute_z : Computes the error between random variable and reference

    Notes
    -----
    .. [1] Sohier, D., Castro, P. D. O., Févotte, F.,
    Lathuilière, B., Petit, E., & Jamond, O. (2021).
    Confidence intervals for stochastic arithmetic.
    ACM Transactions on Mathematical Software (TOMS), 47(2), 1-33.

    '''

    assert_is_probability(probability)
    assert_is_confidence(confidence)

    significant = None

    array, reference = preprocess_inputs(array, reference)

    if method == Method.CNH:
        significant = significant_digits_cnh(array=array,
                                             reference=reference,
                                             error=error,
                                             probability=probability,
                                             confidence=confidence,
                                             axis=axis,
                                             shuffle_samples=shuffle_samples)

    elif method == Method.General:
        significant = significant_digits_general(array=array,
                                                 reference=reference,
                                                 error=error,
                                                 axis=axis,
                                                 shuffle_samples=shuffle_samples)

    if base != 2:
        significant = change_base(significant, base)

    return significant


def contributing_digits_cnh(array: InputType,
                            reference: ReferenceType,
                            error: Error,
                            probability: float,
                            confidence: float,
                            axis: int = 0,
                            shuffle_samples: bool = False) -> InputType:
    r'''Compute contributing digits for Centered Hypothesis Normality

    Parameters
    ----------
    array: numpy.ndarray
        Element to compute
    reference: Optional[float|numpy.ndarray]
        Reference for comparing the array
    axis: Optional[int|tuple(int)]
        Axis or axes along which the contributing digits are computed
        default: None
    error : Error | str
        Name of the error function to use to compute E(array, reference).
        default: Error.Relative
    method : Method | str
        Name of the method for the underlying distribution hypothesis
        default: Method.CNH (Centered Normality Hypothesis)
    probability : float
        Probability for the contributing digits result
        default: 0.51
    confidence : float
        Confidence level for the contributing digits result
        default: 0.95
    shuffle_samples : bool
        If reference is None, the array is split in two and
        comparison is done between both pieces.
        If shuffle_samples is True, it shuffles pieces.

    Returns
    -------
    ndarray
        array_like containing contributing digits

    See Also
    --------
    significantdigits.significant_general : Computes the significant digits for general case
    significantdigits.compute_z : Computes the error between random variable and reference

    Notes
    -----
    .. [1] Sohier, D., Castro, P. D. O., Févotte, F.,
    Lathuilière, B., Petit, E., & Jamond, O. (2021).
    Confidence intervals for stochastic arithmetic.
    ACM Transactions on Mathematical Software (TOMS), 47(2), 1-33.

    .. math::
        c >= -log_2(std) - [\frac{1}{2} log_2( \frac{n-1} / \frac{ Chi^2_{1-\frac{alpha}{2}} }) ) + log_2(p+\frac{1}{2}) + log_2(2\sqrt{2\pi})]
    '''
    z = compute_z(array, reference, error, axis=axis,
                  shuffle_samples=shuffle_samples)
    nb_samples = z.shape[axis]
    std = np.std(z, axis=axis, dtype=internal_dtype)
    std0 = np.ma.masked_array(std == 0)
    chi2 = scipy.stats.chi2.interval(confidence, nb_samples-1)[0]
    delta_chn = 0.5*np.log2((nb_samples - 1)/chi2) + \
        np.log2(probability-0.5) + np.log2(2*np.sqrt(2*np.pi))
    contributing = -np.log2(std) - delta_chn
    if contributing.ndim != 0:
        contributing[std0] = np.finfo(z.dtype).nmant - delta_chn
    elif std0:
        contributing = np.finfo(z.dtype).nmant - delta_chn

    return contributing


def contributing_digits_general(array: InputType,
                                reference: ReferenceType,
                                error: Error,
                                probability: float,
                                axis: int = 0,
                                shuffle_samples: bool = False) -> InputType:
    r'''Computes contributing digits for unknown underlying distribution

    This function computes with a certain probability the number of bits
    of the mantissa that will round the result towards the correct reference
    value[1]_

    Parameters
    ----------
    array: numpy.ndarray
        Element to compute
    reference: Optional[float|numpy.ndarray]
        Reference for comparing the array
    axis: Optional[int|tuple(int)]
        Axis or axes along which the contributing digits are computed
        default: None
    error : Error | str
        Name of the error function to use to compute E(array, reference).
        default: Error.Relative
    method : Method | str
        Name of the method for the underlying distribution hypothesis
        default: Method.CNH (Centered Normality Hypothesis)
    probability : float
        Probability for the contributing digits result
        default: 0.51
    shuffle_samples : bool
        If reference is None, the array is split in two and
        comparison is done between both pieces.
        If shuffle_samples is True, it shuffles pieces.

    Returns
    -------
    ndarray
        array_like containing contributing digits

    See Also
    --------
    significantdigits.significant_digits_cnh : Computes the significant digits under CNH
    significantdigits.compute_z : Computes the error between random variable and reference

    Notes
    -----
    .. [1] Sohier, D., Castro, P. D. O., Févotte, F.,
    Lathuilière, B., Petit, E., & Jamond, O. (2021).
    Confidence intervals for stochastic arithmetic.
    ACM Transactions on Mathematical Software (TOMS), 47(2), 1-33.

    .. math::
        C^{i_k} = "\lfloor 2^k|Z_i|  \rfloor is even"
        c = (\frac{#success}{#trials} > p)
    '''

    z = compute_z(array, reference, error, axis=axis,
                  shuffle_samples=shuffle_samples)
    nb_samples = z.shape[axis]
    sample_shape = tuple(dim for i, dim in enumerate(z.shape) if i != axis)
    contributing = np.zeros(sample_shape)
    contributing = np.zeros(
        [dim for i, dim in enumerate(z.shape) if i != axis])
    max_bits = np.finfo(z.dtype).nmant
    z_mask = np.full(sample_shape, fill_value=True)

    for k in range(1, max_bits + 1):
        pow2k = np.power(2, k)
        successes = np.floor(pow2k * np.abs(z)) % 2 == 0
        _z = np.sum(successes, axis=axis) / nb_samples
        z_mask = z_mask & (_z > probability)
        contributing[z_mask] = k

    return contributing


def contributing_digits(array: InputType,
                        reference: ReferenceType = None,
                        axis: int = 0,
                        base: int = 2,
                        error: Error = Error.Relative,
                        method: Method = Method.CNH,
                        probability: float = default_probability[Metric.Contributing],
                        confidence: float = default_confidence[Metric.Contributing],
                        shuffle_samples: bool = False) -> InputType:
    r'''Compute contributing digits


    This function computes with a certain probability the number of bits
    of the mantissa that will round the result towards the correct reference
    value[1]_

    Parameters
    ----------
    array: numpy.ndarray
        Element to compute
    reference: Optional[float|numpy.ndarray]
        Reference for comparing the array
    axis: Optional[int|tuple(int)]
        Axis or axes along which the contributing digits are computed
        default: None
    error : Error | str
        Name of the error function to use to compute E(array, reference).
        default: Error.Relative
    method : Method | str
        Name of the method for the underlying distribution hypothesis
        default: Method.CNH (Centered Normality Hypothesis)
    probability : float
        Probability for the contributing digits result
        default: 0.51
    confidence : float
        Confidence level for the contributing digits result
        default: 0.95
    shuffle_samples : bool
        If reference is None, the array is split in two and
        comparison is done between both pieces.
        If shuffle_samples is True, it shuffles pieces.

    Returns
    -------
    ndarray
        array_like containing contributing digits

    See Also
    --------
    significantdigits.significant_digits : Computes the significant digits
    significantdigits.compute_z : Computes the error between random variable and reference

    Notes
    -----
    .. [1] Sohier, D., Castro, P. D. O., Févotte, F.,
    Lathuilière, B., Petit, E., & Jamond, O. (2021).
    Confidence intervals for stochastic arithmetic.
    ACM Transactions on Mathematical Software (TOMS), 47(2), 1-33.

    '''

    assert_is_probability(probability)
    assert_is_confidence(confidence)

    contributing = None

    array, reference = preprocess_inputs(array, reference)

    if method == Method.CNH:
        contributing = contributing_digits_cnh(array=array,
                                               reference=reference,
                                               error=error,
                                               axis=axis,
                                               probability=probability,
                                               confidence=confidence,
                                               shuffle_samples=shuffle_samples)

    elif method == Method.General:
        contributing = contributing_digits_general(array=array,
                                                   reference=reference,
                                                   error=error,
                                                   axis=axis,
                                                   probability=probability,
                                                   shuffle_samples=shuffle_samples)

    if base != 2:
        contributing = change_base(contributing, base)

    return contributing


def probability_estimation_general(success: int,
                                   trials: int,
                                   confidence: float) -> float:
    r'''Computes probability lower bound for Bernouilli process

    This function computes the probability associated with metrics
    computed in the general case (without assumption on the underlying
    distribution). Indeed, in that case the probability is given by the
    sample size with a certain confidence level.


    Notes
    -----
    .. [1] Sohier, D., Castro, P. D. O., Févotte, F.,
    Lathuilière, B., Petit, E., & Jamond, O. (2021).
    Confidence intervals for stochastic arithmetic.
    ACM Transactions on Mathematical Software (TOMS), 47(2), 1-33.

    .. math::
        p = 1 + \frac{\log{ 1 - \alpha }}{n} if success=sample\_size
        p = \frac{s+2}{s+4} - F^{-1}(\frac{p+1}{2}) \sqrt{ \frac{(s+2)(n-s+2)}{n+4}^3  } else
    '''
    s = success
    n = trials
    coef = scipy.stats.norm.ppf(confidence)

    if s == n:
        # Special case when having only successes
        probability = 1 + np.log(1 - confidence) / n
    else:
        probability = (s + 2) / (n + 4) - coef * \
            np.sqrt((s + 2) * (n - s + 2) / (n + 4)**3)

    return probability


def minimum_number_of_trials(probability: float, confidence: float) -> int:
    r'''Computes the minimum number of trials to have probability and confidence

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

    '''
    alpha = 1 - confidence
    n = np.log(alpha) / np.log(probability)
    return int(np.ceil(n))
