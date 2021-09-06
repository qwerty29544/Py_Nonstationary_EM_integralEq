import numpy as np

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
