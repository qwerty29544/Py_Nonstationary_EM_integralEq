import numpy as np


def L2(vector):
    return np.sqrt(vector @ vector)


def point_to_rot(point):
    if len(point) == 3:
        V_rot = np.ones(4)
        V_rot[0:3] = point
        return V_rot
    else:
        return None


def rot_to_point(rot):
    return rot[0:3]


def shift_matrix(point):
    if len(point) == 3:
        T = np.diag(np.ones(4))
        T[3, 0:3] = point
        return T
    else:
        return None


def resize_matrix(point):
    d = point_to_rot(point)
    S = np.diag(d)
    return S


def rot_matrix(theta, c):
    a11 = np.cos(theta) + (c[0]**2) * (1 - np.cos(theta))
    a12 = c[0] * c[1] * (1 - np.cos(theta)) - c[2] * np.sin(theta)
    a13 = c[0] * c[2] * (1 - np.cos(theta)) + c[1] * np.sin(theta)
    a21 = c[0] * c[1] * (1 - np.cos(theta)) + c[2] * np.sin(theta)
    a22 = np.cos(theta) + (c[1]**2) * (1 - np.cos(theta))
    a23 = c[1] * c[2] * (1 - np.cos(theta)) - c[0] * np.sin(theta)
    a31 = c[0] * c[2] * (1 - np.cos(theta)) - c[1] * np.sin(theta)
    a32 = c[1] * c[2] * (1 - np.cos(theta)) + c[0] * np.sin(theta)
    a33 = np.cos(theta) + (c[2]**2) * (1 - np.cos(theta))
    return np.array([[a11, a12, a13, 0],
                     [a21, a22, a23, 0],
                     [a31, a32, a33, 0],
                     [0, 0, 0, 1]])


def rotation_XYZ(point, vector_begin, vector_end, theta):
    c = (vector_end - vector_begin) / L2(vector_end - vector_begin)
    rot = point_to_rot(point)
    point = rot_to_point(rot @ shift_matrix(-1 * vector_begin) @ rot_matrix(theta, c) @ shift_matrix(vector_begin))
    for i in range(len(point)):
        if np.abs(point[i]) < 1e-14:
            point[i] = np.round(point[i])
    return point