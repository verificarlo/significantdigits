#!/usr/bin/python3

import numpy as np
from significantdigits.test.utils import (
    get_numpy_data_path, get_report_data_path)


class TestDotProduct:

    filename_csv = get_numpy_data_path('higham_dot_product.csv')
    filename_npy = get_report_data_path('higham_dot_product.npy')
    args = [10**i for i in range(2, 6)]

    def dot_product(self, sizes):
        results = []
        for size in sizes:
            x = np.random.random_sample(size)
            y = np.random.random_sample(size)
            z = np.dot(x, y)
            results.append(z)
        return results

    def test_fuzzy(self, run_fuzzy, nsamples, save):
        samples = run_fuzzy.run(nsamples, self.dot_product, self.args)
        save.save(self.filename_npy, samples)

    def test_significant_digits(self, load, run_significant_digits):
        x = load.load(self.filename_npy)
        ref = np.mean(x, axis=0)
        run_significant_digits.run(self.filename_csv, x, ref)

    def test_contributing_digits(self, load, run_contributing_digits):
        x = load.load(self.filename_npy)
        ref = np.mean(x, axis=0)
        run_contributing_digits.run(self.filename_csv, x, ref)
