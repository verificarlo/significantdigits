#!/usr/bin/python3

import numpy as np
from significantdigits.test.utils import (
    get_numpy_data_path, get_report_data_path)


class TestCramer:
    '''Solving 2x2 system a.x=b with Cramer's rule'''

    filename_npy = get_numpy_data_path('cramer.npy')
    filename_csv = get_report_data_path('cramer.csv')
    a = np.array([[0.2161, 0.1441], [1.2969, 0.8648]])
    b = np.array([0.1440, 0.8642])
    args = (a, b)

    def cramer(self, a, b):
        det = a[0, 0]*a[1, 1] - a[1, 0]*a[0, 1]
        det0 = b[0]*a[1, 1] - b[1]*a[0, 1]
        det1 = a[0, 0]*b[1] - a[1, 0]*b[0]
        return np.array([det0/det, det1/det])

    def test_fuzzy(self, run_fuzzy, nsamples, save):
        samples = run_fuzzy.run(nsamples, self.cramer, *self.args)
        save.save(self.filename_npy, samples)

    def test_significant_digits(self, load, run_significant_digits):
        x = load.load(self.filename_npy)
        ref = np.array([2, -2])
        run_significant_digits.run(self.filename_csv, x, ref)

    def test_contributing_digits(self, load, run_contributing_digits):
        x = load.load(self.filename_npy)
        ref = np.array([2, -2])
        run_contributing_digits.run(self.filename_csv, x, ref)
