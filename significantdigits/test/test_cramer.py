#!/usr/bin/python3

import numpy as np
import significantdigits as sd


class TestCramer:
    """Solving 2x2 system a.x=b with Cramer's rule"""

    filename = "cramer"
    a = np.array([[0.2161, 0.1441], [1.2969, 0.8648]])
    b = np.array([0.1440, 0.8642])
    args = (a, b)

    def cramer(self, a, b):
        det = a[0, 0] * a[1, 1] - a[1, 0] * a[0, 1]
        det0 = b[0] * a[1, 1] - b[1] * a[0, 1]
        det1 = a[0, 0] * b[1] - a[1, 0] * b[0]
        return np.array([det0 / det, det1 / det])

    def test_fuzzy(self, run_fuzzy, nsamples, save):
        samples = run_fuzzy.run(nsamples, self.cramer, *self.args)
        save.save(self.filename, samples)

    def test_significant_digits(self, load, run_significant_digits):
        x = load.load(self.filename)
        ref = np.array([2, -2])
        run_significant_digits.run(self.filename, x, ref)

    def test_contributing_digits(self, load, run_contributing_digits):
        x = load.load(self.filename)
        ref = np.array([2, -2])
        run_contributing_digits.run(self.filename, x, ref)


class TestCramerPaper:
    """Comparing with the Cramer example in the paper"""

    filename = "cramer_paper"

    confidence = 0.95
    probability_significant = 0.99
    probability_contributing = 0.51

    sd_reference_values = {
        ("CNH", "Absolute"): 26.094599393993263,
        ("CNH", "Relative"): 27.094599370144074,
        ("General", "Absolute"): 25,
        ("General", "Relative"): 26,
    }

    cd_reference_values = {
        ("CNH", "Absolute"): 31.777744510355838,
        ("CNH", "Relative"): 32.77774448650665,
        ("General", "Absolute"): 25,
        ("General", "Relative"): 26,
    }

    def _get_values(self):
        x = np.loadtxt("data/cramer-x0-10000.txt")
        y = np.mean(x, axis=0)
        return x, y

    def test_cnh_absolute(self):
        """Test CNH method with absolute error"""
        x, y = self._get_values()
        s = sd.significant_digits(
            x,
            reference=y,
            method=sd.Method.CNH,
            error=sd.Error.Absolute,
            probability=self.probability_significant,
            confidence=self.confidence,
        )
        assert np.isclose(s, self.sd_reference_values[("CNH", "Absolute")], rtol=1e-5)

    def test_cnh_relative(self):
        """Test CNH method with relative error"""
        x, y = self._get_values()
        s = sd.significant_digits(
            x,
            reference=y,
            method=sd.Method.CNH,
            error=sd.Error.Relative,
            probability=self.probability_significant,
            confidence=self.confidence,
        )
        assert np.isclose(s, self.sd_reference_values[("CNH", "Relative")], rtol=1e-5)

    def test_general_absolute(self):
        """Test General method with absolute error"""
        x, y = self._get_values()
        s = sd.significant_digits(
            x,
            reference=y,
            method=sd.Method.General,
            error=sd.Error.Absolute,
            probability=self.probability_significant,
            confidence=self.confidence,
        )
        assert np.isclose(
            s, self.sd_reference_values[("General", "Absolute")], rtol=1e-5
        )

    def test_general_relative(self):
        """Test General method with relative error"""
        x, y = self._get_values()
        s = sd.significant_digits(
            x,
            reference=y,
            method=sd.Method.General,
            error=sd.Error.Relative,
            probability=self.probability_significant,
            confidence=self.confidence,
        )
        assert np.isclose(
            s, self.sd_reference_values[("General", "Relative")], rtol=1e-5
        )
