import sys

import numpy as np


# Solving 2x2 system a.x=b with Cramer's rule
def cramer(a, b):
    det = a[0, 0]*a[1, 1] - a[1, 0]*a[0, 1]
    det0 = b[0]*a[1, 1] - b[1]*a[0, 1]
    det1 = a[0, 0]*b[1] - a[1, 0]*b[0]
    return np.array([det0/det, det1/det])


if __name__ == '__main__':

    n_samples = sys.argv[1]

    a = np.array([[0.2161, 0.1441], [1.2969, 0.8648]])
    b = np.array([0.1440, 0.8642])
    x = cramer(a, b)

    samples = [cramer(a, b) for _ in range(n_samples)]
    np.save("cramer", samples)
