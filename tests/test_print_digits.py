import significantdigits as sd
import numpy as np


def test_print_digits_absolute():
    x = np.loadtxt("data/cramer-x0-10000.txt")
    fmt = sd.print_digits(
        x, reference=np.mean(x), error=sd.Error.Absolute, probability=0.99
    )
    print(fmt)
    for i in fmt[:10]:
        print(i)


def test_print_digits_relative():
    x = np.loadtxt("data/cramer-x0-10000.txt")
    fmt = sd.print_digits(x, reference=np.mean(x))
    print(fmt)
    for i in fmt[:10]:
        print(i)


if __name__ == "__main__":
    test_print_digits_absolute()
    test_print_digits_relative()
