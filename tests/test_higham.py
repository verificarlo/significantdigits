#!/usr/bin/python3

import numpy as np
from tests.utils import Setup

setup = Setup()


class TestDotProduct:
    filename = "higham_dot_product"
    args = [10**i for i in range(2, 6)]

    def dot_product(self, sizes):
        results = []
        for size in sizes:
            x = setup.rng.random_sample(size)
            y = setup.rng.random_sample(size)
            z = np.dot(x, y)
            results.append(z)
        return results

    def test_fuzzy(self, run_fuzzy, nsamples, save):
        samples = run_fuzzy.run(nsamples, self.dot_product, self.args)
        save.save(self.filename, samples)

    def test_significant_digits(self, load, run_significant_digits):
        x = load.load(self.filename)
        ref = np.mean(x, axis=0)
        run_significant_digits.run(self.filename, x, ref)

    def test_contributing_digits(self, load, run_contributing_digits):
        x = load.load(self.filename)
        ref = np.mean(x, axis=0)
        run_contributing_digits.run(self.filename, x, ref)
