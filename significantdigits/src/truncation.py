import numpy as np
import ctypes

truncation_clib_name = "truncation.so"
truncation_clib = ctypes.CDLL(truncation_clib_name)
truncation_clib.truncate_binary32.restype = ctypes.c_float
truncation_clib.truncate_binary64.restype = ctypes.c_double


def truncate(x, n):
    if n < 0:
        raise ArithmeticError("n must be positive")

    mantissa_size = np.finfo(type(x)).nmant
    if n > mantissa_size:
        return x

    fp_trunc = None
    if isinstance(x, (np.float16, np.float32)):
        fp_f32 = ctypes.c_float(x)
        fp_trunc = truncation_clib.truncate_binary32(fp_f32, n)
    elif isinstance(x, (float, np.float64)):
        fp_f64 = ctypes.c_double(x)
        fp_trunc = truncation_clib.truncate_binary64(fp_f64, n)
    else:
        raise TypeError(type(x))
    return fp_trunc


if __name__ == "__main__":
    x16 = np.float16(0.9995)
    print("Truncation for float16:", x16)
    for i in range(11):
        print(i, f"{truncate(x16, i):.4f}")

    x32 = np.float32(0.99999994)
    print("Truncation for float32:", x32)
    print(x32)
    for i in range(24):
        print(i, f"{truncate(x32, i):.8f}")

    x64 = np.float64(0.9999999999999999)
    print("Truncation for float64:", x64)
    print(x64)
    for i in range(53):
        print(i, truncate(x64, i))
