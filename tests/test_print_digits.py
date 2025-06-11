import significantdigits as sd
import numpy as np


def test_print_digits():
    x = np.loadtxt("data/cramer-x0-10000.txt")
    fmt = sd.print_digits(x, reference=np.mean(x))
    print(fmt)


if __name__ == "__main__":
    test_print_digits()
