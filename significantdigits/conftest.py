import csv
import itertools
import os
import sys

import numpy as np
import pytest

import significantdigits
import significantdigits.args


class Save:

    def save(self, outputs, samples):
        np.save(outputs, samples)
        assert(os.path.isfile(outputs))


class Load:

    def load(self, inputs):
        assert(os.path.isfile(inputs))
        return np.load(inputs)


class RunFuzzyTest:

    def run(self, nsamples, function, *args, **kwargs):
        return [function(*args, **kwargs) for _ in range(nsamples)]


class RunMetricDigitsTest:

    available_metrics = ['significant', 'contributing']

    def __init__(self, metric):
        self.metric = self.check_metric(metric)
        self.bases = [2, 10]
        self.methods = significantdigits.significantdigits.Method
        self.errors = significantdigits.significantdigits.Error

    def check_metric(self, metric):
        if metric in self.available_metrics:
            return metric
        else:
            print((f'Unknown metric {metric}.'
                   f' Must be one of {self.available_metrics}'))
            sys.exit(1)

    def compute_significant_digits(self, x, ref, error, method, base):
        return significantdigits.significantdigits.significant_digits(
            x, ref, error=error, method=method, base=base)

    def compute_contributing_digits(self, x, ref, error, method, base):
        return significantdigits.significantdigits.contributing_digits(
            x, ref, error=error, method=method, base=base)

    def compute_metric(self, *args, **kwargs):
        if self.metric == 'significant':
            return self.compute_significant_digits(*args, **kwargs)
        else:
            return self.compute_contributing_digits(*args, **kwargs)

    def save_result(self, metric, error, method, base):
        result = dict(Method=method,
                      Error=error,
                      Base=base,
                      Metric=metric)
        self.writer.writerow(result.values())

    def run(self, output, x, ref):
        header = ['Method', 'Error', 'Base', self.metric.capitalize()]
        with open(output, 'w') as fo:
            self.writer = csv.writer(fo)
            self.writer.writerow(header)
            configurations = itertools.product(self.bases,
                                               self.methods,
                                               self.errors)
            for (base, method, error) in configurations:
                metric = self.compute_metric(x,
                                             ref,
                                             error,
                                             method,
                                             base)
                self.save_result(metric,
                                 error,
                                 method,
                                 base)


def pytest_addoption(parser):
    parser.addoption(
        "--nsamples", action="store", type=int, default=3,
        help="Number of samples to run"
    )


@pytest.fixture
def nsamples(request):
    return request.config.getoption("--nsamples")


@pytest.fixture
def run_fuzzy(request):
    return RunFuzzyTest()


@pytest.fixture
def run_significant_digits(request):
    return RunMetricDigitsTest('significant')


@pytest.fixture
def run_contributing_digits(request):
    return RunMetricDigitsTest('contributing')


@pytest.fixture
def save(request):
    return Save()


@pytest.fixture
def load(request):
    return Load()


@pytest.fixture
def parser(request):
    return significantdigits.args.create_parser()
