#!/usr/bin/python3

import numpy as np
from significantdigits.test.utils import Setup

setup = Setup()


class TestScalar:
    """Test scalar value"""

    filename = "scalar"
    n = 100
    a = np.ones(n)

    args = (a,)

    def noise(self, a):
        eps = setup.rng.uniform(-(10**-14), 10**-14, size=a.size)
        return a + eps

    def test_fuzzy(self, run_fuzzy, nsamples, save):
        samples = run_fuzzy.run(nsamples, self.noise, *self.args)
        save.save(self.filename, samples)

    def test_significant_digits(self, load, run_significant_digits):
        x = load.load(self.filename)
        ref = 1
        run_significant_digits.run(self.filename, x, ref)

    def test_contributing_digits(self, load, run_contributing_digits):
        x = load.load(self.filename)
        ref = 1
        run_contributing_digits.run(self.filename, x, ref)


class TestScalarIdentical(TestScalar):
    """Test identical result"""

    filename = "scalar_identical"
    n = 100
    a = np.ones(n)

    args = (a,)

    def noise(self, a):
        return a
