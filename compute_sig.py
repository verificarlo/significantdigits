import sigdigits
import sys
import numpy as np

if __name__ == '__main__':
    xf = sys.argv[1]
    x = np.load(xf)
    ref = np.array([2, -2])

    for method in sigdigits.Method:
        for precision in sigdigits.Precision:
            sig = sigdigits.significant_digits(
                x, ref, precision=precision, method=method)
            print(f"[{method.name:7}] {precision.name:9} significant:", sig)

        for precision in sigdigits.Precision:
            con = sigdigits.contributing_digits(
                x, ref, precision=precision, method=method)
            print(f"[{method.name:7}] {precision.name:8} contributing:", con)
