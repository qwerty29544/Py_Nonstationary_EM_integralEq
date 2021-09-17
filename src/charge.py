import numpy as np
from divergence_approx import div_vec_approx, gradient_vec


def f_function(rho, D):
    if rho == 0 or rho == D:
        return 0
    if np.abs(1 / rho * (D - rho)) >= 20:
        return 0
    if D > rho > 0:
        return np.exp(-((1) / (rho * (D - rho))))
    else:
        return 0


class Charge:
    def __init__(self,
                 time_step,
                 max_time,
                 max_diameter,
                 G1,
                 G2,
                 G3,
                 tau,
                 norms,
                 frames,
                 squares,
                 collocations,
                 total_number_of_frames,
                 div_approx=3,
                 orientation=None):

        # Первоначальная ориентация вектора E_{inc}
        if orientation is None:
            orientation = [0., 1., 0.]

        self.time_step = time_step  # Шаг по времени
        self.max_time = max_time  # Максмимальное время наблюдения
        self.timeline_vec = np.arange(0, max_time, time_step)  # Дискретный вектор времени
        self.max_d = max_diameter  # Максимальный диаметр разбиения
        self.total_number_of_frames = total_number_of_frames  # Количество разбиений  N
        self.G1_coeffs = G1  # Коэффициенты схемы    N x N x 3
        self.G2_coeffs = G2  # Коэффициенты схемы    N x N x 3
        self.G3_coeffs = G3  # Коэффициенты схемы    N x N
        self.tau = tau / (3 * 1e8)  # Евклидово расстояние между точками коллокации на скорость света
        self.norms = norms  # Нормы     N x 3
        self.frames = frames  # Разбиения или рамки N x 4 x 3
        self.collocations = collocations  # Точки коллокаций Nx3
        self.div_approx = div_approx  # Степень аппроксимации поверхностной дивергенции
        self.squares = squares  # Площади разбиений
        self.orientation = np.array(orientation)    # Ориентация E_{inc}

        self.Q = np.zeros((self.total_number_of_frames, 2))  # Плотность зарядов на поверхности тела
        self.P = np.zeros((self.total_number_of_frames, 2))  # Изменение плотности зарядов на поверхности тела
        self.G = np.zeros((self.total_number_of_frames, 2, 3))  # Вектор электрического тока на поверхности тела
        self.D = np.zeros((self.total_number_of_frames, 1, 3))

        # Индексная зависимость массивов Q, P, D в каждой точке коллокации
        self.indexes_upper = np.array(np.ceil(self.tau / time_step), dtype='int32')
        self.indexes_lower = self.indexes_upper + 1

        self.current_time = 1  # Текущий счётчик индекса времени

    def _alpha_beta_(self,
                     time_k,  # Время T(k)
                     time_star):  # Время между (time - tau)
        beta = (time_star - time_k) / self.time_step
        return 1 - beta, beta

    # ВСЕ ГЕТТЕРЫ НИЖЕ МОЖНО СВЕРНУТЬ, КРОМЕ ГЕТТЕРА D --------------------------------------------------
    def get_Q(self, item):
        if item < 0:
            return self.Q[:, 0]
        elif item <= self.current_time:
            return self.Q[:, item]
        else:
            return self.Q[:, self.current_time]

    def get_P(self, item):
        if item < 0:
            return self.P[:, 0]
        elif item <= self.current_time:
            return self.P[:, item]
        else:
            return self.P[:, self.current_time]

    def get_G(self, item):
        if item < 0:
            return self.G[:, 0]
        elif item <= self.current_time:
            return self.G[:, item]
        else:
            return self.G[:, self.current_time]

    def get_D(self, item):
        if item < 0:
            return self.D[:, 0]
        elif item <= self.current_time - 1:
            return self.D[:, item]
        else:
            return self.D[:, self.current_time - 1]

    def get_P_element(self, row, col):
        if row >= 0 and row < self.total_number_of_frames:
            if col < 0:
                return self.P[row][0]
            elif col <= self.current_time:
                return self.P[row][col]
            else:
                return self.P[row][self.current_time]
        else:
            return 0

    def get_Q_element(self, row, col):
        if row >= 0 and row < self.total_number_of_frames:
            if col < 0:
                return self.Q[row][0]
            elif col <= self.current_time:
                return self.Q[row][col]
            else:
                return self.Q[row][self.current_time]
        else:
            return 0

    def get_G_element(self, row, col):
        if row >= 0 and row < self.total_number_of_frames:
            if col < 0:
                return self.G[row][0]
            elif col <= self.current_time:
                return self.G[row][col]
            else:
                return self.G[row][self.current_time]
        else:
            return 0

    def get_D_element(self, row, col):
        if row >= 0 and row < self.total_number_of_frames:
            if col < 0:
                return self.D[row][0]
            elif col <= self.current_time - 1:
                return self.D[row][col]
            else:
                return self.D[row][self.current_time - 1]
        else:
            return np.array([0, 0, 0])

    # ----------------------------------------------------------------------------------------------------------
    def compute_Q_past(self, row, col):
        time_now = self.timeline_vec[self.current_time]
        tau_in_past = self.tau[row][col]
        time_star = time_now - tau_in_past
        upper_index = self.indexes_upper[row][col]
        lower_index = self.indexes_lower[row][col]
        Q_upper = self.get_Q_element(row, self.current_time - upper_index)
        Q_lower = self.get_Q_element(row, self.current_time - lower_index)
        if time_star < 0:
            return 0
        else:
            alpha, beta = self._alpha_beta_(self.timeline_vec[self.current_time - lower_index], time_star)
        return Q_upper * beta - Q_lower * alpha

    def compute_P_past(self, row, col):
        time_now = self.timeline_vec[self.current_time]
        tau_in_past = self.tau[row][col]
        time_star = time_now - tau_in_past
        upper_index = self.indexes_upper[row][col]
        lower_index = self.indexes_lower[row][col]
        P_upper = self.get_P_element(row, self.current_time - upper_index)
        P_lower = self.get_P_element(row, self.current_time - lower_index)
        if time_star < 0:
            return 0
        else:
            alpha, beta = self._alpha_beta_(self.timeline_vec[self.current_time - lower_index], time_star)
        return P_upper * beta - P_lower * alpha

    def compute_D_past(self, row, col):
        time_now = self.timeline_vec[self.current_time]
        tau_in_past = self.tau[row][col]
        time_star = time_now - tau_in_past
        upper_index = self.indexes_upper[row][col]
        lower_index = self.indexes_lower[row][col]
        D_upper = self.get_D_element(row, self.current_time - upper_index)
        D_lower = self.get_D_element(row, self.current_time - lower_index)
        if time_star < 0:
            return np.array([0., 0., 0.])
        else:
            alpha, beta = self._alpha_beta_(self.timeline_vec[self.current_time - lower_index], time_star)
        return D_upper * beta - D_lower * alpha

    def write_step_Q_file(self, tensor, filename):
        f = open(filename, 'w')
        f.write(str(self.total_number_of_frames) + "\n")
        for i in range(self.total_number_of_frames):
            f.write(str(tensor[i]) + "\n")
        f.close()

    def write_step_D_file(self, tensor, filename):
        f = open(filename, "w")
        f.write(str(self.total_number_of_frames) + "\n")
        for i in range(self.total_number_of_frames):
            f.write("\t".join(map(str, tensor[i])) + "\n")
        f.close()

    def step_in_time(self):
        # Подсчитать S -> подсчитать P(x(j), t(n) - tau(i,j), Q(x(j), t(n) - tau(i,j), D(x(j), t(n) - tau(i,j)
        time_now = self.timeline_vec[self.current_time]
        # Вычисление вектора D на шаге n
        d_step = []         # Планируется N x 3
        S_step = []         # Планируется N x 3
        for i in range(self.total_number_of_frames):
            P_past = []
            Q_past = []
            D_past = []
            for j in range(self.total_number_of_frames):
                # Вычисляю значения векторов Q, P, D для каждой пары ячеек i, j
                Q_past.append(self.compute_Q_past(i, j))
                P_past.append(self.compute_P_past(i, j))
                # это условие можно вынести в саму функцию выше
                if i != j:
                    D_past.append(list(self.compute_D_past(i, j)))
                else:
                    D_past.append([0, 0, 0])
            # Превратил полученные массивы в np.array Nx1
            P_past = np.array(P_past)
            Q_past = np.array(Q_past)
            # Этот массив Nx3
            D_past = np.array(D_past)
            # Подсчитал S для каждого i-го D
            S_step.append(
                list(self.G1_coeffs[i].T @ P_past + self.G2_coeffs[i].T @ Q_past + self.G3_coeffs[i] @ D_past))
        # Получил массив Nx3
        S_step = np.array(S_step)

        # Вычисление каждого i-го D
        for i in range(self.total_number_of_frames):
            # Вектор правой части для каждой i-ой точки коллокации
            F = f_function(3 * 1e8 * self.timeline_vec[self.current_time] - self.collocations[i][2], 1)
            f_vec = self.orientation * F
            # Считаем D на шаге
            d_step.append(list(
                (-np.cross(np.cross(self.norms[i], S_step[i]), self.norms[i]) + np.cross(f_vec, self.norms[i])) /
                self.G3_coeffs[i][i]))
        # Подсчитаный D на шаге морфим в Nx1x3 для конкатенации с итоговым тензором (N x t x 3)
        d_step = np.array(d_step).reshape((self.total_number_of_frames, 1, 3))

        # Конкатенация D и логгирование в файл нового шага
        self.D = np.concatenate((self.D, d_step), axis=1)
        print("D computes! on step " + str(self.current_time))
        file = "d:\\Python\\MaxwellIntegralEq\\logs\\plate_12_12\\D\\plate_12_12_step_" + str(self.current_time) + ".dat"
        self.write_step_D_file(tensor=d_step.reshape((self.total_number_of_frames, 3)),
                               filename=file)


        # Вычисление Q, P, G на n + 1 ----------------------------------------------------------------
        # Вычисление G на шаге
        new_G = self.get_G(self.current_time - 1) + 2 * self.time_step * self.D[:, self.current_time]
        # Морфинг в N x 1 x 3
        new_G = new_G.reshape((self.total_number_of_frames, 1, 3))
        # Конкатенация с итоговым
        self.G = np.concatenate((self.G, new_G), axis=1)
        # Логгирование и запись в файл
        print("G computes! on step " + str(self.current_time))
        file = "d:\\Python\\MaxwellIntegralEq\\logs\\plate_12_12\\G\\plate_12_12_step_" + str(self.current_time) + ".dat"
        self.write_step_D_file(tensor=new_G.reshape((self.total_number_of_frames, 3)),
                               filename=file)

        # Вычисление P на шаге
        new_P = []
        for i in range(self.total_number_of_frames):
            # Дивергенция на каждой точке коллокации возвращает число
            div = div_vec_approx(collocation=self.collocations[i],                  # Коллокация i
                                 vec_collocation=self.G[:, self.G.shape[1] - 1][i], # Последний посчитаный G в i
                                 vec=self.G[:, self.G.shape[1] - 1],                # Последний посчитаный G
                                 collocations=self.collocations,                    # Тензор коллокаций вообще
                                 squares=self.squares,                              # Вектор площадей
                                 jmax=self.total_number_of_frames,                  # Количество разбиений
                                 eps=2 * self.max_d,                                # 2 диаметра
                                 appr_degree=self.div_approx,                       # Степень аппроксимации
                                 grad_Function=gradient_vec)                        # Функция подсчёта градиента
            new_P.append(div)   # Конкатенация в список
        # Массив np.array размером N x 1
        new_P = np.array(new_P).reshape((self.total_number_of_frames, 1))
        # Конкатенация с (N x t)
        self.P = np.concatenate((self.P, new_P), axis=1)
        # Логгирование и запись в файл
        print("P computes! on step " + str(self.current_time))
        file = "d:\\Python\\MaxwellIntegralEq\\logs\\plate_12_12\\P\\plate_12_12_step_" + str(self.current_time) +".dat"
        self.write_step_Q_file(new_P[:, 0], file)

        # Подсчёт нового вектора Q (N x 1)
        new_Q = self.Q[:, self.current_time - 1] + 2 * self.time_step * self.P[:, self.current_time]
        # Конкатенация с (N x t)
        self.Q = np.concatenate((self.Q, new_Q.reshape((self.total_number_of_frames, 1))), axis=1)
        # Логгирование и запись в файл
        print("Q computes! on step " + str(self.current_time))
        file = "d:\\Python\\MaxwellIntegralEq\\logs\\plate_12_12\\Q\\plate_12_12_step_" + str(self.current_time) +".dat"
        self.write_step_Q_file(new_Q, file)

        # Шаг во времени
        self.current_time += 1
        print(" ")


