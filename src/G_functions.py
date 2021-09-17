import numpy as np                              # Библиотека обработки векторов, матриц и тензоров numpy.array()
import numba                                    # Библиотека параллелизации и компиляции кода numpy python
import customalgebra as ca                  # Собственные функции для вычисления линейных преобразований


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


@numba.jit(nopython=True, parallel=True, nogil=True)
def integr_G3(frame, point, num_of_frame, num_of_collocation, *args):
    if (num_of_frame == num_of_collocation):
        # Треугольник OAB - O точка центра рамки
        result = 0
        for i in range(4):
            AB = frame[(i + 1) % 4] - frame[i]
            # Поворот на Пи относительно срединного перпендикуляра
            H = ca.rotation_XYZ(point=point, vector_begin=(point + frame[i]) / 2, vector_end=(point + frame[(i + 1) % 4]) / 2, theta=np.pi)
            mod_OH = ca.L2(H - point)
            HA = frame[i] - H
            HB = frame[(i + 1) % 4] - H

            alpha_min = np.arctan(ca.L2(HA) / mod_OH)
            alpha_max = np.arctan(ca.L2(HB) / mod_OH)

            # Решение какой угол отрицательный
            if (HA @ AB) <= 0:
                alpha_min = -alpha_min

            if (HB @ AB) <= 0:
                alpha_max = -alpha_max

            square = mod_OH * (np.log(np.abs((1 + np.tan(alpha_max / 2))/(1 - np.tan(alpha_max / 2)))) -
                               np.log(np.abs((1 + np.tan(alpha_min / 2))/(1 - np.tan(alpha_min / 2)))))
            result += square
    else:
        result = 0
    return result