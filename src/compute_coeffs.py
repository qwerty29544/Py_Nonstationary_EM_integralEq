import numpy as np
import numba
from G_functions import G3_func, integr_G3


# Префикс компилятора питона numba
@numba.jit(nopython=True, parallel=True, nogil=True)
def compute_coeffs(frame,                   # Все рамки объекта в формате (N x 4 x 3) - массив numpy
                   collocation,             # Все точки коллокации объекта в формате (N x 3) - массив numpy
                   number_of_frames,        # Общее количество разбиений объекта - число
                   integration_method,      # Метод интегрирования (доразбиения)
                   integral_function,       # Функция, которую мы интегрируем
                   max_diameter,            # Максимальный диаметр разбиения на объекте
                   ndim,                    # Размерность вектора результата
                   n_vertex=4,              # Количество углов у фигуры разбиения
                   num_slices=10,           # Количество подразбиений для внедиагональных элементов
                   num_slices_diag=5):     # Количество подразбиений для диагональных элементов
    rb = max_diameter / num_slices          # Epsilon - коэффициент для сглаживающей функции
    coeffs = np.zeros((number_of_frames, number_of_frames, ndim))   # Массив для коэффициентов (N x N x ndim)
    for i in range(number_of_frames):
        for j in range(number_of_frames):
            if i == j:
                slices = num_slices_diag    # Если точка коллокации в рамке - num_slices_diag разбиений
            else:
                slices = num_slices         # Если точка коллокации вне рамки - num_slices разбиений
            coeffs[i][j] = integration_method(collocation[i],
                                              frame[j],
                                              n_vertex,
                                              slices,
                                              integral_function,
                                              ndim,
                                              rb)

    return coeffs


@numba.jit(nopython=True, parallel=True, nogil=True)
def compute_G3_coefficients(frame,                   # Все рамки объекта в формате (N x 4 x 3) - массив numpy
                            collocation,             # Все точки коллокации объекта в формате (N x 3) - массив numpy
                            number_of_frames,        # Общее количество разбиений объекта - число
                            integration_method,      # Метод интегрирования (доразбиения)
                            max_diameter,            # Максимальный диаметр разбиения на объекте
                            n_vertex=4,              # Количество углов у фигуры разбиения
                            num_slices=10):          # Количество подразбиений для внедиагональных элементов
    rb = max_diameter / num_slices          # Epsilon - коэффициент для сглаживающей функции
    coeffs = np.zeros((number_of_frames, number_of_frames, 1))   # Массив для коэффициентов (N x N x ndim)
    slices = num_slices
    for i in range(number_of_frames):
        for j in range(number_of_frames):
            if i == j:
                coeffs[i][j] = integr_G3(frame[j], collocation[i], j, i)
            else:
                coeffs[i][j] = integration_method(collocation[i], frame[j], n_vertex, slices, G3_func, 1, rb)
    coeffs.reshape((number_of_frames, number_of_frames))
    return coeffs


def coeffs_save(coeffs, filename):
    def coeffs_print(frame, file):
        if len(frame.shape) == 1:
            file.write("\t".join(map(str, frame)) + "\n")
            return frame
        else:
            for i in range(frame.shape[0]):
                coeffs_print(frame[i], file)

    file = open(filename, "w")
    file.write(str(len(coeffs.shape)) + "\n")
    file.write("\t".join(map(str, coeffs.shape)) + "\n")
    coeffs_print(coeffs, file)
    file.close()


def coeffs_load(filename):
    file = open(filename, "r")
    n_dims = int(file.readline())
    tensor_shape = tuple(map(int, file.readline().split()))
    tensor_input_shape = [np.prod(tensor_shape[:(n_dims - 1)]), tensor_shape[n_dims - 1]]
    input_tensor = []
    for row in range(tensor_input_shape[0]):
        input_tensor.append(list(map(float, file.readline().split())))
    input_tensor = np.array(input_tensor).reshape(tensor_shape)
    file.close()
    return input_tensor


