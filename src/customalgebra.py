import numpy as np
import numba


@numba.jit(nopython=True, parallel=True, nogil=True)
def L2(vector):
    return np.sqrt(vector @ vector)


@numba.jit(nopython=True, parallel=True, nogil=True)
def point_to_rot(point):
    V_rot = np.ones(4)
    if point.shape[0] == 3:
        V_rot[0:3] = point
        return V_rot
    else:
        return V_rot


@numba.jit(nopython=True, parallel=True, nogil=True)
def rot_to_point(rot):
    return rot[0:3]


@numba.jit(nopython=True, parallel=True, nogil=True)
def shift_matrix(point):
    T = np.diag(np.ones(4) * 1.0)
    if point.shape[0] == 3:
        T[3, 0:3] = point
        return T
    else:
        return T


@numba.jit(nopython=True, parallel=True, nogil=True)
def resize_matrix(point):
    d = point_to_rot(point)
    S = np.diag(d) * 1.0
    return S


@numba.jit(nopython=True, parallel=True, nogil=True)
def rot_matrix(theta, c):
    result = np.zeros((4, 4))
    result[0, 0] = np.cos(theta) + (c[0] ** 2) * (1 - np.cos(theta))
    result[0, 1] = c[0] * c[1] * (1 - np.cos(theta)) - c[2] * np.sin(theta)
    result[0, 2] = c[0] * c[2] * (1 - np.cos(theta)) + c[1] * np.sin(theta)
    result[1, 0] = c[0] * c[1] * (1 - np.cos(theta)) + c[2] * np.sin(theta)
    result[1, 1] = np.cos(theta) + (c[1] ** 2) * (1 - np.cos(theta))
    result[1, 2] = c[1] * c[2] * (1 - np.cos(theta)) - c[0] * np.sin(theta)
    result[2, 0] = c[0] * c[2] * (1 - np.cos(theta)) - c[1] * np.sin(theta)
    result[2, 1] = c[1] * c[2] * (1 - np.cos(theta)) + c[0] * np.sin(theta)
    result[2, 2] = np.cos(theta) + (c[2] ** 2) * (1 - np.cos(theta))
    result[3, 3] = 1
    return result


@numba.jit(nopython=True, parallel=True, nogil=True)
def rotation_XYZ(point, vector_begin, vector_end, theta):
    c = (vector_end - vector_begin) / L2(vector_end - vector_begin)
    rot = point_to_rot(point)
    rot_shift = rot @ shift_matrix(-1 * vector_begin)
    rot_shift_rot = rot_shift @ rot_matrix(theta, c)
    rot_rot = rot_shift_rot @ shift_matrix(vector_begin)
    point_l = rot_to_point(rot_rot)
    for i in range(point_l.shape[0]):
        if np.abs(point_l[i]) < 1e-14:
            point_l[i] = np.round(point_l[i])
    return point_l
