
import math
from enum import Enum

import numpy as np
import scipy
import scipy.stats


class Method(Enum):
    """
    Parker: Parker's formula
            Particular case of CHN where
            - Precision: Relative
            - Reference: mean_X
    CNH: Centered Normality Hypothesis
         X follows a Gaussian law centered around the reference or
         Z follows a Gaussian law centered around 0
    General: No assumption about the distribution of X or Z
    """
    Parker = "Parker"
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
default_contributing_probability = 0.51
default_confidence = 0.95


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
    sig_power2 = np.power(2, -sig)
    to_base = np.frompyfunc(lambda x: math.log(x, base))
    return to_base(sig_power2)


def compute_z(array, reference, precision, shuffle_samples=False):
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

    nb_samples = array.shape[0]

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
        z = array - reference
    elif precision == Precision.Relative:
        z = array/reference - 1
    else:
        raise Exception(f"Unknown precision {precision}")

    return z


def significant_digits_cnh(array,
                           reference,
                           precision,
                           probability,
                           confidence,
                           shuffle_samples=False):
    """
    s >= -log_2(std) - [1/2 log2( (n-1)/(Chi^2_{1-alpha/2}) ) + log2( F^{-1}((p+1)/2)]
    """
    z = compute_z(array, reference, precision, shuffle_samples=shuffle_samples)
    nb_samples = z.shape[0]
    std = np.std(z, axis=0, dtype=internal_dtype)
    chi2 = scipy.stats.chi2.ppf(1-confidence/2, nb_samples)
    inorm = scipy.stats.norm.ppf((probability+1)/2)
    delta_chn = 0.5*np.log2((nb_samples - 1)/chi2) + np.log2(inorm)
    sig = -np.log2(std) - delta_chn

    return sig


def significant_digits_general(array,
                               reference,
                               precision,
                               probability,
                               confidence,
                               shuffle_samples=False):
    """
    s = max{k in [1,mant], st forall i in [1,n], |Z_i| <= 2^{-k}}
    """
    z = compute_z(array, reference, precision, shuffle_samples=shuffle_samples)

    sig = np.zeros(z.shape[1:])
    max_bits = np.finfo(z.dtype).nmant
    for k in range(max_bits, 0, -1):
        pow2minusk = np.power(2, -np.float(k))
        _z = np.all(np.abs(z) <= pow2minusk, axis=0)
        z_mask = np.ma.masked_array(_z, axis=0, fill_value=k)
        sig[~z_mask] = k
        if np.all(z_mask):
            break

    return sig


def significant_digits(array,
                       reference=None,
                       base=2,
                       precision=Precision.Relative,
                       method=Method.CNH,
                       probability=default_probability,
                       confidence=default_confidence,
                       shuffle_samples=False):

    assert_is_probability(probability)
    assert_is_confidence(confidence)

    sig = None

    if method == Method.Parker:
        reference = np.mean(array, axis=0, dtype=internal_dtype)
        sig = significant_digits_cnh(array,
                                     reference,
                                     Precision.Relative,
                                     probability,
                                     confidence,
                                     False)

    elif method == Method.CNH:
        sig = significant_digits_cnh(array,
                                     reference,
                                     precision,
                                     probability,
                                     confidence,
                                     shuffle_samples)

    elif method == Method.General:
        sig = significant_digits_general(array,
                                         reference,
                                         precision,
                                         probability,
                                         confidence,
                                         shuffle_samples)

    if base != 2:
        sig = change_base(sig, base)

    return sig


def contributing_digits_cnh(array,
                            reference,
                            precision,
                            probability=0.51,
                            confidence=default_confidence,
                            shuffle_samples=False):
    """
    c >= -log_2(std) - [1/2 log2( (n-1)/(Chi^2_{1-alpha/2}) ) + log2(p+1/2) + log2(2.sqrt(2.pi))]
    """
    z = compute_z(array, reference, precision, shuffle_samples=shuffle_samples)
    nb_samples = z.shape[0]
    std = np.std(z, axis=0, dtype=internal_dtype)
    chi2 = scipy.stats.chi2.ppf(1-confidence/2, nb_samples)
    delta_chn = 0.5*np.log2((nb_samples - 1)/chi2) + \
        np.log2(probability-0.5) + np.log2(2*np.sqrt(2*np.pi))
    con = -np.log2(std) - delta_chn

    return con


def contributing_digits_general(array,
                                reference,
                                precision,
                                probability=default_contributing_probability,
                                confidence=default_confidence,
                                shuffle_samples=False):
    """
    C^i_k = "floor( 2^k|Z_i| ) is even"
    c = (#success/#trials > 0.5)
    """

    z = compute_z(array, reference, precision, shuffle_samples=shuffle_samples)

    con = np.zeros(z.shape[1:])
    max_bits = np.finfo(z.dtype).nmant
    z_mask = np.full(z.shape[1:], fill_value=True)
    for k in range(1, max_bits+1):
        pow2k = np.power(2, k)
        _z = np.sum(np.floor(pow2k*np.abs(z)) % 2 == 0, axis=0)/z.shape[0]
        z_mask = z_mask & (_z > probability)
        con[z_mask] = k

    return con


def contributing_digits(array,
                        reference=None,
                        base=2,
                        precision=Precision.Relative,
                        method=Method.CNH,
                        probability=default_contributing_probability,
                        confidence=default_confidence,
                        shuffle_samples=False):

    assert_is_probability(probability)
    assert_is_confidence(confidence)

    con = None

    if method == Method.Parker:
        reference = np.mean(array, axis=0, dtype=internal_dtype)
        con = contributing_digits_cnh(array,
                                      reference,
                                      Precision.Relative,
                                      probability,
                                      confidence,
                                      False)

    elif method == Method.CNH:
        con = contributing_digits_cnh(array,
                                      reference,
                                      precision,
                                      probability,
                                      confidence,
                                      shuffle_samples)

    elif method == Method.General:
        con = contributing_digits_general(array,
                                          reference,
                                          precision,
                                          probability,
                                          confidence,
                                          shuffle_samples)

    if base != 2:
        con = change_base(con, base)

    return con
