
import math
from enum import Enum

import numpy as np
import scipy
import scipy.stats
import warnings


class Method(Enum):
    """
    CNH: Centered Normality Hypothesis
         X follows a Gaussian law centered around the reference or
         Z follows a Gaussian law centered around 0
    General: No assumption about the distribution of X or Z
    """
    CNH = "Centered Normality Hypothese"
    General = "General"


class Precision(Enum):
    """
    ----------+-------------+-------------
              | Reference x | Reference Y
    ----------+-------------+-------------
    Absolute  | Z = X - x   | Z = X - Y
    Relative  | Z = X/x - 1 | Z = X/Y - 1
    ----------+-------------+-------------
    """
    Absolute = "Absolute"
    Relative = "Relative"


internal_dtype = np.dtype(np.float64)
default_probability = 0.95
default_confidence = 0.95
default_contributing_probability = 0.51


def assert_is_valid_method(method):
    if method in Method:
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


def compute_z(array, reference, precision, axis=0, shuffle_samples=False):
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

    if precision == Precision.Absolute:
        z = x - y
    elif precision == Precision.Relative:
        if np.any(np.ma.masked_equal(y, 0)):
            warnings.warn(
                "Precision is set to relative and the reference contains 0 leading to NaN")
        z = x/y - 1
    else:
        raise Exception(f"Unknown precision {precision}")

    return z


def significant_digits_cnh(array,
                           reference,
                           precision,
                           probability,
                           confidence,
                           axis=0,
                           shuffle_samples=False):
    """
    s >= -log_2(std) - [1/2 log2( (n-1)/(Chi^2_{1-alpha/2}) ) + log2( F^{-1}((p+1)/2)]
    """
    z = compute_z(array, reference, precision, axis=axis,
                  shuffle_samples=shuffle_samples)
    nb_samples = z.shape[axis]
    std = np.std(z, axis=axis, dtype=internal_dtype)
    std0 = np.ma.masked_array(std == 0)
    chi2 = scipy.stats.chi2.interval(confidence, nb_samples-1)[0]
    inorm = scipy.stats.norm.ppf((probability+1)/2)
    delta_chn = 0.5*np.log2((nb_samples - 1)/chi2) + np.log2(inorm)
    sig = -np.log2(std) - delta_chn
    sig[std0] = np.finfo(z.dtype).nmant - delta_chn
    return sig


def significant_digits_general(array,
                               reference,
                               precision,
                               probability,
                               confidence,
                               axis=0,
                               shuffle_samples=False):
    """
    s = max{k in [1,mant], st forall i in [1,n], |Z_i| <= 2^{-k}}
    """
    z = compute_z(array, reference, precision, axis=axis,
                  shuffle_samples=shuffle_samples)

    z_nan_mask = np.ma.masked_invalid(z)
    sample_shape = tuple(dim for i, dim in enumerate(z.shape) if i != axis)
    max_bits = np.finfo(z.dtype).nmant
    sig = np.full(sample_shape, max_bits, dtype=np.float64)
    for k in range(max_bits, -1, -1):
        pow2minusk = np.power(2, -np.float(k))
        _z = np.all(np.abs(z_nan_mask) <= pow2minusk, axis=axis)
        z_mask = np.ma.masked_array(_z, fill_value=k)
        sig[~z_mask] = k
        if np.all(z_mask):
            break

    sig[z_mask.mask] = np.nan
    return sig


def significant_digits(array,
                       reference=None,
                       axis=0,
                       base=2,
                       precision=Precision.Relative,
                       method=Method.CNH,
                       probability=default_probability,
                       confidence=default_confidence,
                       shuffle_samples=False):

    assert_is_probability(probability)
    assert_is_confidence(confidence)

    sig = None

    if method == Method.CNH:
        sig = significant_digits_cnh(array=array,
                                     reference=reference,
                                     precision=precision,
                                     probability=probability,
                                     confidence=confidence,
                                     axis=axis,
                                     shuffle_samples=shuffle_samples)

    elif method == Method.General:
        sig = significant_digits_general(array=array,
                                         reference=reference,
                                         precision=precision,
                                         probability=probability,
                                         confidence=confidence,
                                         axis=axis,
                                         shuffle_samples=shuffle_samples)

    if base != 2:
        sig = change_base(sig, base)

    return sig


def contributing_digits_cnh(array,
                            reference,
                            precision,
                            axis=0,
                            probability=0.51,
                            confidence=default_confidence,
                            shuffle_samples=False):
    """
    c >= -log_2(std) - [1/2 log2( (n-1)/(Chi^2_{1-alpha/2}) ) + log2(p+1/2) + log2(2.sqrt(2.pi))]
    """
    z = compute_z(array, reference, precision, axis=axis,
                  shuffle_samples=shuffle_samples)
    nb_samples = z.shape[axis]
    std = np.std(z, axis=axis, dtype=internal_dtype)
    std0 = np.ma.masked_array(std == 0)
    chi2 = scipy.stats.chi2.interval(confidence, nb_samples-1)[0]
    delta_chn = 0.5*np.log2((nb_samples - 1)/chi2) + \
        np.log2(probability-0.5) + np.log2(2*np.sqrt(2*np.pi))
    con = -np.log2(std) - delta_chn
    con[std0] = np.finfo(z.dtype).nmant - delta_chn
    return con


def contributing_digits_general(array,
                                reference,
                                precision,
                                axis=0,
                                probability=default_contributing_probability,
                                confidence=default_confidence,
                                shuffle_samples=False):
    """
    C^i_k = "floor( 2^k|Z_i| ) is even"
    c = (#success/#trials > 0.5)
    """

    z = compute_z(array, reference, precision, axis=axis,
                  shuffle_samples=shuffle_samples)

    nb_samples = z.shape[axis]
    sample_shape = tuple(dim for i, dim in enumerate(z.shape) if i != axis)
    con = np.zeros(sample_shape)
    con = np.zeros([dim for i, dim in enumerate(z.shape) if i != axis])
    max_bits = np.finfo(z.dtype).nmant
    z_mask = np.full(sample_shape, fill_value=True)
    for k in range(1, max_bits+1):
        pow2k = np.power(2, k)
        _z = np.sum(np.floor(pow2k*np.abs(z)) % 2 == 0, axis=axis)/nb_samples
        z_mask = z_mask & (_z > probability)
        con[z_mask] = k

    return con


def contributing_digits(array,
                        reference=None,
                        axis=0,
                        base=2,
                        precision=Precision.Relative,
                        method=Method.CNH,
                        probability=default_contributing_probability,
                        confidence=default_confidence,
                        shuffle_samples=False):

    assert_is_probability(probability)
    assert_is_confidence(confidence)

    con = None

    if method == Method.CNH:
        con = contributing_digits_cnh(array=array,
                                      reference=reference,
                                      precision=precision,
                                      axis=axis,
                                      probability=probability,
                                      confidence=confidence,
                                      shuffle_samples=shuffle_samples)

    elif method == Method.General:
        con = contributing_digits_general(array=array,
                                          reference=reference,
                                          precision=precision,
                                          axis=axis,
                                          probability=probability,
                                          confidence=confidence,
                                          shuffle_samples=shuffle_samples)

    if base != 2:
        con = change_base(con, base)

    return con
