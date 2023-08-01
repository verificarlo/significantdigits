import csv
import itertools
import os
import sys

import numpy as np
import pytest

import significantdigits
import significantdigits.args
from significantdigits.test.utils import Setup


class Save:
    def __init__(self):
        self.setup = Setup()

    def save(self, outputs, samples):
        filename = self.setup.get_numpy_data_path(outputs)
        np.save(filename, samples)
        assert os.path.isfile(filename)


class Load:
    def __init__(self):
        self.setup = Setup()

    def load(self, inputs):
        filename = self.setup.get_numpy_data_path(inputs)
        assert os.path.isfile(filename)
        return np.load(filename)


class RunFuzzyTest:
    def run(self, samples, function, *args, **kwargs):
        return [function(*args, **kwargs) for _ in range(samples)]


class RunMetricDigitsTest:
    available_metrics = ["significant", "contributing"]

    def __init__(self, metric):
        self.setup = Setup()
        self.metric = self.check_metric(metric)
        self.bases = [2, 10]
        self.methods = significantdigits.Method
        self.errors = significantdigits.Error
        self.writer = None

    def check_metric(self, metric):
        if metric in self.available_metrics:
            return metric
        else:
            print(
                (
                    f"Unknown metric {metric}."
                    f" Must be one of {self.available_metrics}"
                )
            )
            sys.exit(1)

    def compute_significant_digits(self, x, ref, error, method, base):
        return significantdigits.significant_digits(
            x, ref, error=error, method=method, base=base
        )

    def compute_contributing_digits(self, x, ref, error, method, base):
        return significantdigits.contributing_digits(
            x, ref, error=error, method=method, base=base
        )

    def compute_metric(self, *args, **kwargs):
        if self.metric == "significant":
            return self.compute_significant_digits(*args, **kwargs)
        else:
            return self.compute_contributing_digits(*args, **kwargs)

    def save_result(self, metric, error, method, base):
        result = {"Method": method, "Error": error, "Base": base, "Metric": metric}
        self.writer.writerow(result.values())

    def run(self, output, x, ref):
        header = ["Method", "Error", "Base", self.metric.capitalize()]
        filename = self.setup.get_report_path(output)
        with open(filename, "a", encoding="utf-8") as fo:
            self.writer = csv.writer(fo)
            self.writer.writerow(header)
            configurations = itertools.product(self.bases, self.methods, self.errors)
            for base, method, error in configurations:
                metric = self.compute_metric(x, ref, error, method, base)
                self.save_result(metric, error, method, base)


def pytest_addoption(parser):
    parser.addoption(
        "--nsamples",
        action="store",
        type=int,
        default=3,
        help="Number of samples to run",
    )


@pytest.fixture
def nsamples(request):
    return request.config.getoption("--nsamples")


@pytest.fixture
def run_fuzzy(request):
    return RunFuzzyTest()


@pytest.fixture
def run_significant_digits(request):
    return RunMetricDigitsTest("significant")


@pytest.fixture
def run_contributing_digits(request):
    return RunMetricDigitsTest("contributing")


@pytest.fixture
def save(request):
    return Save()


@pytest.fixture
def load(request):
    return Load()


@pytest.fixture
def parser(request):
    return significantdigits.args.create_parser()
