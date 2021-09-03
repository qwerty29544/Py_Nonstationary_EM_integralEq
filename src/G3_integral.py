import numpy as np
import src.rotation as rot
import numba


@numba.jit(nopython=True, parallel=True, nogil=True)
def integr_G3(frame, point, num_of_frame, num_of_collocation, *args):
    if (num_of_frame == num_of_collocation):
        # Треугольник OAB - O точка центра рамки
        result = 0
        for i in range(4):
            AB = frame[(i + 1) % 4] - frame[i]
            # Поворот на Пи относительно срединного перпендикуляра
            H = rot.rotation_XYZ(point=point, vector_begin=(point + frame[i]) / 2, vector_end=(point + frame[(i + 1) % 4]) / 2, theta=np.pi)
            mod_OH = rot.L2(H - point)
            HA = frame[i] - H
            HB = frame[(i + 1) % 4] - H

            alpha_min = np.arctan(rot.L2(HA) / mod_OH)
            alpha_max = np.arctan(rot.L2(HB) / mod_OH)

            # Решение какой угол отрицательный
            if (HA @ AB) > 0:
                alpha_min = alpha_min
                alpha_max = -alpha_max
            else:
                alpha_min = -alpha_min
                alpha_max = alpha_max

            square = mod_OH * (np.log(np.abs((1 + np.tan(alpha_max / 2))/(1 - np.tan(alpha_max / 2)))) -
                               np.log(np.abs((1 + np.tan(alpha_min / 2))/(1 - np.tan(alpha_min / 2)))))
            result += square
    else:
        result = 0
    return result