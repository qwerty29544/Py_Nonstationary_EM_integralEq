import numpy as np
import numba


# Префикс для компилятора питона numba
@numba.jit(nopython=True, parallel=True, nogil=True)
def G3_func(x, y, rb):
    zr = 1e-12
    r = np.sqrt((x - y) @ np.conj(x - y))
    c = 3 * 1e8
    if r >= rb:
        result = - 1 / r * (1 / (c ** 2))
    elif r >= zr:
        t = r / rb
        result = - 1 / r * (3 * t ** 2 - 2 * t ** 3) * (1 / (c ** 2))
    else:
        result = 0.0
    return result


# Префикс для компилятора питона numba
@numba.jit(nopython=True, parallel=True, nogil=True)
def G2_func(x, y, rb):
    zr = 1e-8
    r = np.sqrt((x - y) @ np.conj(x - y))
    if r >= rb:
        result = (x - y) * (1 / (r ** 3))
    elif r >= zr:
        t = r / rb
        result = (x - y) * (1 / (r ** 3)) * (3 * t ** 2 - 2 * t ** 3)
    else:
        result = np.zeros(3)
    return result


# Префикс для компилятора питона numba
@numba.jit(nopython=True, parallel=True, nogil=True)
def G1_func(x, y, rb):
    zr = 1e-12
    r = np.sqrt((x - y) @ np.conj(x - y))
    c = 3 * 1e8
    if r >= rb:
        result = (x - y) * (1 / r) * (1 / c)
    elif r >= zr:
        t = r / rb
        result = (x - y) * (1 / r) * (3 * t ** 2 - 2 * t ** 3) * (1 / c)
    else:
        result = np.zeros(3)
    return result
