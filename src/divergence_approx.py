import numpy as np
import numba


@numba.jit(nopython=True, parallel=True, nogil=True)
def div_vec_approx(collocation,         # Точка коллокации  x[3]
                   vec_collocation,     # Значение вектора под дивергенцией в точке коллокации gx[3]
                   vec,                 # Весь вектор под дивергенцией g(x_i, t_{n+1}) g[N][3]
                   collocations,        # Точки коллокаций всех разбиений   rkt[N][3]
                   squares,             # Площади разбиений   Squares[N]
                   jmax,                # Размер всего вектора под дивергенцией N
                   eps,                 # Радиус подсчёта дивергенции
                   appr_degree,         # Степень аппроскимации
                   grad_Function):      # Фукнция, реализующая подсчёт градиента в точке коллокации
    """
    Функция рассчета поверхностной дивергенции в точке
    на основе ядра свёртки интегральной функции,
    аппроскимируемой полиномом
    """
    div_vec_appr = 0            # Итоговая дивергенция
    for j in range(jmax):
        grad_psi = grad_Function(collocation, collocations[j], eps, appr_degree)
        div_vec_appr += ((vec[j] - vec_collocation) @  grad_psi) * squares[j]
    return div_vec_appr


@numba.jit(nopython=True, parallel=True, nogil=True)
def gradient_vec(x,                     # Точка коллокации которую мы наблюдаем
                 y,                     # Относительная точка коллокации
                 eps,                   # 2 диаметра разбиения
                 appr_degree=2):        # Степень аппроксимации
    x_y = x - y
    r_2 = (x_y @ x_y) / (eps**2)
    # Граничное условие
    if r_2 > 50:
        grad = np.zeros(3)
    else:
        if appr_degree == 1:
            grad = -2 * x_y * np.exp(-r_2) / np.pi / (eps**4)
        elif appr_degree == 2:
            grad = (-6 + 2 * r_2) * (x_y) * np.exp(-r_2) / np.pi / (eps**4)
        elif appr_degree == 3:
            grad = (-12 + 8 * r_2 - (r_2) ** 2) * (x_y) * np.exp(-r_2) / np.pi / (eps ** 4)
        else:
            grad = (-20 + 20 * r_2 - 5 * ((r_2)**2) + ((r_2)**3) / 3) * (x_y) * np.exp(-r_2) / np.pi / (eps**4)

    return grad


@numba.jit(nopython=True, parallel=True, nogil=True)
def div_element_surface(vec_element_k: np.array,
                        frame: np.array,
                        neighbors_vecs: np.array,
                        square: float,
                        norm: np.array) -> float:
    """
    Функция, которая считает дивергенцию относительно одной точки центра

    :param vec_element_k: Значение вектора в точке коллокации, массив (3), значение дивергенции которого мы хотим посчитать по этой ячейке
    :param frame: Массив (4, 3) для точек нашего прямоугольного разбиения (единичной поверхности)
    :param neighbors_vecs: массив (4, 3) из значений векторов по которому считаем дивергецию, в соседних ячейках
    :param square: Площадь ячейки, число
    :param norm: Нормаль к ячейке массив (3)
    :return: Значение дивергенции - число
    """
    result = 0
    m = frame.shape[0]
    for i in range(m):
        vec_prod = np.cross((frame[(i+1) % m] - frame[i]), norm)
        sum_vecs = (vec_element_k + neighbors_vecs[i]) / 2.
        dot_prod = sum_vecs @ vec_prod
        result += dot_prod
    return result / square


@numba.jit(nopython=True, parallel=True, nogil=True)
def div_surface(vec: np.array,
                frames: np.array,
                neighbors_inds: np.array,
                squares: np.array,
                norms: np.array) -> np.array:
    """
    Функция, которая считает дивергенцию вектора по всей поверхности разбиения

    :param vec: Вектор, дивергенцию которого мы ищем в каждой точке разбиения, массив (N, 3)
    :param frames: Массив разбиений, состоящий из структур типа ячеек, массив (N, 4, 3)
    :param neighbors_inds: Массив индексов соседних элементов разбиений поверхности, относительно текущей точки, массив (N, 4)
    :param squares: Массив площадей разбиений, массив (N)
    :param norms: Массив нормалей к ячейкам по всей поверхности разбиения (N, 4)
    :return: Массив (N) - дивергенция вектора в каждой точке разбиения
    """
    N = frames.shape[0]     # Общее число разбиений модуля
    div_vec = np.zeros(N)   # Итоговый массив дивергенций в точках

    for k in range(N):
        neighbors_vecs = np.zeros((4, 3))
        for i in range(4):
            idx = int(neighbors_inds[k][i])
            if idx == (-1):
                continue
            else:
                neighbors_vecs[i, :] = vec[idx, :]
        div_vec[k] = div_element_surface(vec_element_k=vec[k],
                                         frame=frames[k],
                                         neighbors_vecs=neighbors_vecs,
                                         square=squares[k],
                                         norm=norms[k])
    return div_vec

