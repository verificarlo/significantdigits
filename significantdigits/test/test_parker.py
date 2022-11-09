#!/usr/bin/python3

import math

import numpy as np
import scipy.linalg
from significantdigits.test.utils import (
    get_numpy_data_path, get_report_data_path)

# Set of tests taken from:
# Parker, Douglas Stott, Brad Pierce, and Paul R. Eggert.
# "Monte Carlo arithmetic: how to gamble with floating point and win."
# Computing in Science & Engineering 2.4 (2000): 58-68.


class TestDiscriminant:
    '''Solving quadratic equation using discriminant
        due to W. Kahan
    '''
    filename_npy = get_numpy_data_path('parker_descriminant.npy')
    filename_csv = get_report_data_path('parker_descriminant.csv')
    args = (7, -8686, 2)

    def discriminant(self, a, b, c):
        return (-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)

    def test_fuzzy(self, run_fuzzy, nsamples, save):
        samples = run_fuzzy.run(nsamples, self.discriminant, *self.args)
        save.save(self.filename_npy, samples)

    def test_significant_digits(self, load, run_significant_digits):
        x = load.load(self.filename_npy)
        ref = np.mean(x, axis=0)
        run_significant_digits.run(self.filename_csv, x, ref)

    def test_contributing_digits(self, load, run_contributing_digits):
        x = load.load(self.filename_npy)
        ref = np.mean(x, axis=0)
        run_contributing_digits.run(self.filename_csv, x, ref)


class TestCancellation:
    '''Simple example illustrating catastrophic cancellation
    due to non-associativity.
    '''

    filename_npy = get_numpy_data_path('parker_cancellation.npy')
    filename_csv = get_report_data_path('parker_cancellation.csv')

    def cancellation(self):
        x = (11111113 - 11111111) + 7.5111111
        y = 11111113 + (-11111111 + 7.5111111)
        return np.array([x, y])

    def test_fuzzy(self, run_fuzzy, nsamples, save):
        samples = run_fuzzy.run(nsamples, self.cancellation)
        save.save(self.filename_npy, samples)

    def test_significant_digits(self, load, run_significant_digits):
        x = load.load(self.filename_npy)
        ref = [9.5111111, 9.5111111]
        run_significant_digits.run(self.filename_csv, x, ref)

    def test_contributing_digits(self, load, run_contributing_digits):
        x = load.load(self.filename_npy)
        ref = [9.5111111, 9.5111111]
        run_contributing_digits.run(self.filename_csv, x, ref)


class TestMuller:

    filename_npy = get_numpy_data_path('parker_muller.npy')
    filename_csv = get_report_data_path('parker_muller.csv')

    args = (1.510005072139, 30)

    def muller(self, x0, n):

        def sequence(x_k):
            num = (3 * x_k ** 4 - 20 * x_k ** 3 + 35 * x_k ** 2 - 24)
            den = (4 * x_k ** 3 - 30 * x_k ** 2 + 70 * x_k - 50)
            return num / den

        x_k0 = x0
        x_k1 = x_k0

        results = []

        for _ in range(1, n + 1):
            results.append(x_k1)
            x_k1, x_k0 = sequence(x_k0), x_k1

        return results

    def test_fuzzy(self, run_fuzzy, nsamples, save):
        samples = run_fuzzy.run(nsamples, self.muller, *self.args)
        save.save(self.filename_npy, samples)

    def test_significant_digits(self, load, run_significant_digits):
        x = load.load(self.filename_npy)
        ref = np.mean(x, axis=0)
        run_significant_digits.run(self.filename_csv, x, ref)

    def test_contributing_digits(self, load, run_contributing_digits):
        x = load.load(self.filename_npy)
        ref = np.mean(x, axis=0)
        run_contributing_digits.run(self.filename_csv, x, ref)


class TestTchebycheff:

    filename_npy = get_numpy_data_path('parker_tchebycheff.npy')
    filename_csv = get_numpy_data_path('parker_tchebycheff.csv')
    args = np.linspace(0, 1, 100)

    def tchebycheff(self, z):
        coeffs = np.array([524288, -2621440, 5570560, -6553600,
                           4659200, -2050048, 549120, -84480, 6600, -200, 1])
        power = np.array([20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0])
        def t_20(x): return np.sum(coeffs * x ** power)
        return np.array(list(map(t_20, z)))

    def test_fuzzy(self, run_fuzzy, nsamples, save):
        samples = run_fuzzy.run(nsamples, self.tchebycheff, self.args)
        save.save(self.filename_npy, samples)

    def test_significant_digits(self, load, run_significant_digits):
        x = load.load(self.filename_npy)
        ref = np.cos(20 * np.arccos(self.args))
        run_significant_digits.run(self.filename_csv, x, ref)

    def test_contributing_digits(self, load, run_contributing_digits):
        args = self.args
        x = load.load(self.filename_npy)
        ref = np.cos(20 * np.arccos(args))
        run_contributing_digits.run(self.filename_csv, x, ref)


class TestGaussian:

    filename_npy = get_numpy_data_path('parker_gaussian.npy')
    filename_csv = get_numpy_data_path('parker_gaussian.csv')
    args = 24

    def turing_matrix(self, n):
        T = np.eye(n)
        T[:, -1] = np.array([2**i for i in range(n)])
        return T

    def reference(self, n):
        return np.array([2**-i for i in range(self.args)])

    def gaussian(self, n):
        A = self.turing_matrix(n)
        b = np.full(n, 1)
        return scipy.linalg.solve(A, b)

    def test_fuzzy(self, run_fuzzy, nsamples, save):
        samples = run_fuzzy.run(nsamples, self.gaussian, self.args)
        save.save(self.filename_npy, samples)

    def test_significant_digits(self, load, run_significant_digits):
        x = load.load(self.filename_npy)
        ref = self.reference(self.args)
        run_significant_digits.run(self.filename_csv, x, ref)

    def test_contributing_digits(self, load, run_contributing_digits):
        x = load.load(self.filename_npy)
        ref = self.reference(self.args)
        run_contributing_digits.run(self.filename_csv, x, ref)
