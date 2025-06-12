import significantdigits as sd
import numpy as np


def test_format_uncertainty_absolute():
    print("Testing format_uncertainty with absolute error...")
    x = np.loadtxt("data/cramer-x0-10000.txt")
    fmt = sd.format_uncertainty(
        x, reference=np.mean(x), error=sd.Error.Absolute, probability=0.99
    )
    print(fmt)
    for i in fmt[:10]:
        print(i)


def test_format_uncertainty_relative():
    print("Testing format_uncertainty with relative error...")
    x = np.loadtxt("data/cramer-x0-10000.txt")
    fmt = sd.format_uncertainty(x, reference=np.mean(x))
    print(fmt)
    for i in fmt[:10]:
        print(i)


def test_format_uncertainty_absolute_tuple():
    print("Testing format_uncertainty with absolute error as tuple...")
    x = np.loadtxt("data/cramer-x0-10000.txt")
    value, error = sd.format_uncertainty(
        x,
        reference=np.mean(x),
        error=sd.Error.Absolute,
        probability=0.99,
        as_tuple=True,
    )
    print(value, error)
    for i, j in zip(value[:10], error[:10]):
        print(f"Value: {i}, Error: {j}")


def test_format_uncertainty_relative_tuple():
    print("Testing format_uncertainty with relative error as tuple...")
    x = np.loadtxt("data/cramer-x0-10000.txt")
    value, error = sd.format_uncertainty(x, reference=np.mean(x), as_tuple=True)
    print(value, error)
    for i, j in zip(value[:10], error[:10]):
        print(f"Value: {i}, Error: {j}")


if __name__ == "__main__":
    test_format_uncertainty_absolute()
    test_format_uncertainty_relative()
    test_format_uncertainty_absolute_tuple()
    test_format_uncertainty_relative_tuple()
