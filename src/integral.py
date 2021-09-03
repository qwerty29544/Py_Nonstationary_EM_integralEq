import numpy as np
import numba


# Префикс компилятора питона numba
@numba.jit(nopython=True, parallel=True, nogil=True)
def integrate_func(collocation,
                   frame,
                   n_vertex,
                   num_slices,
                   integral_function,
                   ndim,
                   *args):
    res = np.zeros(ndim)
    for i in range(num_slices):
        for j in range(num_slices):
            p = (i + 0.) / num_slices
            q = (j + 0.) / num_slices
            p1 = (i + 1.) / num_slices
            q1 = (j + 1.) / num_slices

            a = q * frame[1] + (1 - q) * frame[0]
            b = q * frame[2] + (1 - q) * frame[3]
            a1 = p * b + (1 - p) * a
            a4 = p1 * b + (1 - p1) * a

            a = q1 * frame[1] + (1 - q1) * frame[0]
            b = q1 * frame[2] + (1 - q1) * frame[3]
            a2 = p * b + (1 - p) * a
            a3 = p1 * b + (1 - p1) * a

            rc = (a1 + a2 + a3 + a4) / 4.

            m1 = ((a2 + a3) - (a1 + a4)) / 2.
            m2 = ((a3 + a4) - (a1 + a2)) / 2.
            rn = np.cross(m1, m2)
            s = np.sqrt(rn @ rn)

            ff = integral_function(collocation, rc, *args)
            res += ff * s
    return res

