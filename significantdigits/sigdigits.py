
import math
import warnings
from enum import Enum, auto

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


def assert_is_valid_method(method):
    if method not in Method:
        raise TypeError(
            f"provided invalid method {method}: must be one of {list(Method)}")


def assert_is_probability(probability):
    if probability < 0 or probability > 1:
        raise TypeError("probability must be between 0 and 1")


def assert_is_confidence(confidence):
    if confidence < 0 or confidence > 1:
        raise TypeError("confidence must be between 0 and 1")


def change_base(sig, base):
    sig_power2 = np.power(2, sig)

    def to_base(x):
        return math.log(x, base)
    np_to_base = np.frompyfunc(to_base, 1, 1)
    x = np_to_base(sig_power2)
    return x


_valid_input_types = (tuple, list, np.ndarray)
_valid_reference_types = (float, tuple, list, np.ndarray)


def preprocess_inputs(array, reference):

    if not isinstance(array, _valid_input_types):
        raise TypeError(f'Input array must be '
                        f'one of the following types {_valid_input_types}')

    if not isinstance(array, np.ndarray):
        array = np.array(array)

    if reference is not None:
        if not isinstance(reference, _valid_reference_types):
            raise TypeError(f'Reference must be '
                            f'one of the following types {_valid_reference_types}')

        if not isinstance(reference, np.ndarray):
            reference = np.array(reference)

    return (array, reference)


def compute_z(array, reference, error, axis=0, shuffle_samples=False):
    """
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
    """
    nb_samples = array.shape[axis]

    if reference is None:
        if nb_samples % 2 != 0:
            raise Exception(
                "Number of samples must be a multiple of 2 without reference provided")
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

    if error == Error.Absolute:
        z = x - y
    elif error == Error.Relative:
        if np.any(y[y == 0]):
            warnings.warn(
                "error is set to relative and the reference contains 0 leading to NaN")
        z = x/y - 1
    else:
        raise Exception(f"Unknown error {error}")

    return z


def significant_digits_cnh(array,
                           reference,
                           error,
                           probability,
                           confidence,
                           axis=0,
                           shuffle_samples=False):
    """
    s >= -log_2(std) - [1/2 log2( (n-1)/(Chi^2_{1-alpha/2}) ) + log2( F^{-1}((p+1)/2)]
    """
    z = compute_z(array, reference, error, axis=axis,
                  shuffle_samples=shuffle_samples)
    nb_samples = z.shape[axis]
    std = np.std(z, axis=axis, dtype=internal_dtype)
    std0 = np.ma.masked_array(std == 0)
    chi2 = scipy.stats.chi2.interval(confidence, nb_samples-1)[0]
    inorm = scipy.stats.norm.ppf((probability+1)/2)
    delta_chn = 0.5*np.log2((nb_samples - 1)/chi2) + np.log2(inorm)
    significant = -np.log2(std) - delta_chn
    if significant.ndim != 0:
        significant[std0] = np.finfo(z.dtype).nmant - delta_chn
    elif std0:
        significant = np.finfo(z.dtype).nmant - delta_chn
    return significant


def significant_digits_general(array,
                               reference,
                               error,
                               probability,
                               confidence,
                               axis=0,
                               shuffle_samples=False):
    """
    s = max{k in [1,mant], st forall i in [1,n], |Z_i| <= 2^{-k}}
    """
    z = compute_z(array, reference, error, axis=axis,
                  shuffle_samples=shuffle_samples)

    z_nan_mask = np.ma.masked_invalid(z)
    sample_shape = tuple(dim for i, dim in enumerate(z.shape) if i != axis)
    max_bits = np.finfo(z.dtype).nmant
    significant = np.full(sample_shape, max_bits, dtype=np.float64)
    z_mask = np.full(sample_shape, False)
    for k in range(max_bits, -1, -1):
        pow2minusk = np.power(2, -np.float64(k))
        _z = np.all(np.abs(z_nan_mask) <= pow2minusk, axis=axis)
        if z.ndim == 0 and _z:
            significant = k
            break
        else:
            z_mask = np.ma.masked_array(data=_z, mask=_z)
            if np.all(_z):
                break
            significant[~z_mask] = k

    # sig[z_mask.mask] = np.nan
    return significant


def significant_digits(array,
                       reference=None,
                       axis=0,
                       base=2,
                       error=Error.Relative,
                       method=Method.CNH,
                       probability=default_probability[Metric.Significant],
                       confidence=default_confidence[Metric.Significant],
                       shuffle_samples=False):

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
                                                 probability=probability,
                                                 confidence=confidence,
                                                 axis=axis,
                                                 shuffle_samples=shuffle_samples)

    if base != 2:
        significant = change_base(significant, base)

    return significant


def contributing_digits_cnh(array,
                            reference,
                            error,
                            probability,
                            confidence,
                            axis=0,
                            shuffle_samples=False):
    """
    c >= -log_2(std) - [1/2 log2( (n-1)/(Chi^2_{1-alpha/2}) ) + log2(p+1/2) + log2(2.sqrt(2.pi))]
    """
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


def contributing_digits_general(array,
                                reference,
                                error,
                                probability,
                                confidence,
                                axis=0,
                                shuffle_samples=False):
    """
    C^i_k = "floor( 2^k|Z_i| ) is even"
    c = (#success/#trials > 0.5)
    """
    z = compute_z(array, reference, error, axis=axis,
                  shuffle_samples=shuffle_samples)

    nb_samples = z.shape[axis]
    sample_shape = tuple(dim for i, dim in enumerate(z.shape) if i != axis)
    contributing = np.zeros(sample_shape)
    contributing = np.zeros(
        [dim for i, dim in enumerate(z.shape) if i != axis])
    max_bits = np.finfo(z.dtype).nmant
    z_mask = np.full(sample_shape, fill_value=True)
    for k in range(1, max_bits+1):
        pow2k = np.power(2, k)
        _z = np.sum(np.floor(pow2k*np.abs(z)) % 2 == 0, axis=axis)/nb_samples
        z_mask = z_mask & (_z > probability)
        contributing[z_mask] = k

    return contributing


def contributing_digits(array,
                        reference=None,
                        axis=0,
                        base=2,
                        error=Error.Relative,
                        method=Method.CNH,
                        probability=default_probability[Metric.Contributing],
                        confidence=default_confidence[Metric.Contributing],
                        shuffle_samples=False):

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
                                                   confidence=confidence,
                                                   shuffle_samples=shuffle_samples)

    if base != 2:
        contributing = change_base(contributing, base)

    return contributing