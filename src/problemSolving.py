import numpy as np
import numba
from typing import List, Callable
from scipy.constants import speed_of_light
from divergence_approx import div_vec_approx, gradient_vec

"""Adams-Bashforth 2-step method coeffs"""
adams_bashforth2_c0: float = 3. / 2.
adams_bashforth2_c1: float = -1. / 2.

"""Adams-Bashforth 3-step method coeffs"""
adams_bashforth3_c0: float = 23. / 12.
adams_bashforth3_c1: float = -4. / 3.
adams_bashforth3_c2: float = 5. / 12.

"""Adams-Bashforth 4-step method coeffs"""
adams_bashforth4_c0: float = 55. / 24.
adams_bashforth4_c1: float = -59. / 24.
adams_bashforth4_c2: float = 37. / 24.
adams_bashforth4_c3: float = -3. / 8.

"""Adams-Bashforth 5-step method coeffs"""
adams_bashforth5_c0: float = 1901. / 720.
adams_bashforth5_c1: float = -1387. / 360.
adams_bashforth5_c2: float = 109. / 30.
adams_bashforth5_c3: float = -637. / 360.
adams_bashforth5_c4: float = 251. / 720.


@numba.jit(nopython=True, parallel=True, nogil=True)
def f_function(params: List[float]) -> float:
    """
    :param params: List of float custom function parameters
    :return: float scalar
    :examples:
    """
    if params[0] == 0. or params[0] == params[1]:
        return 0.
    if np.abs(1. / params[0] * (params[1] - params[0])) >= 20.:
        return 0.
    if params[1] > params[0] > 0.:
        return np.exp(-(1. / (params[0] * (params[1] - params[0]))))
    else:
        return 0.


@numba.jit(nopython=True, parallel=True, nogil=True)
def proportion_linear(time_right: float,
                      time_between: float,
                      time_step: float) -> tuple:
    """
    :param time_right: right border in time
    :param time_between: time between borders
    :param time_step: step in time between borders
    :return: _Iterable_(alpha, beta)

        Typical usage example:

        time_between = 1.98
        time_step = time_2 - time_1
        alpha, beta = proportion_linear(time_2, time_between, time_step)
        assert alpha + beta == 1
        assert alpha > beta
    """
    beta = (time_between - time_right) / time_step
    return (1 - beta), beta


@numba.jit(nopython=True, parallel=True, nogil=True)
def component_next_euler(vec_integral: np.array = None,
                         vec_integral_1: np.array = None,
                         vec_integral_2: np.array = None,
                         vec_integral_3: np.array = None,
                         vec_integral_4: np.array = None,
                         vec_cur: np.array = None,
                         vec_cur_1: np.array = None,
                         vec_cur_2: np.array = None,
                         vec_cur_3: np.array = None,
                         delta: float = 0.1,
                         **kwargs) -> np.array:
    return vec_integral * 2 * delta + vec_cur_1


@numba.jit(nopython=True, parallel=True, nogil=True)
def component_next_euler2(vec_integral: np.array = None,
                          vec_integral_1: np.array = None,
                          vec_integral_2: np.array = None,
                          vec_integral_3: np.array = None,
                          vec_integral_4: np.array = None,
                          vec_cur: np.array = None,
                          vec_cur_1: np.array = None,
                          vec_cur_2: np.array = None,
                          vec_cur_3: np.array = None,
                          delta: float = 0.1,
                          **kwargs) -> np.array:
    return 8 * vec_cur - 8 * vec_cur_2 + vec_cur_3 - 12 * delta * vec_integral_1


@numba.jit(nopython=True, parallel=True, nogil=True)
def component_next_adams(vec_integral: np.array = None,
                         vec_integral_1: np.array = None,
                         vec_integral_2: np.array = None,
                         vec_integral_3: np.array = None,
                         vec_integral_4: np.array = None,
                         vec_cur: np.array = None,
                         vec_cur_1: np.array = None,
                         vec_cur_2: np.array = None,
                         vec_cur_3: np.array = None,
                         delta: float = 0.1,
                         **kwargs) -> np.array:
    return vec_cur + delta * vec_integral


@numba.jit(nopython=True, parallel=True, nogil=True)
def component_next_adams2(vec_integral: np.array = None,
                          vec_integral_1: np.array = None,
                          vec_integral_2: np.array = None,
                          vec_integral_3: np.array = None,
                          vec_integral_4: np.array = None,
                          vec_cur: np.array = None,
                          vec_cur_1: np.array = None,
                          vec_cur_2: np.array = None,
                          vec_cur_3: np.array = None,
                          delta: float = 0.1,
                          **kwargs) -> np.array:
    return vec_cur + delta * (adams_bashforth2_c0 * vec_integral + 
                              adams_bashforth2_c1 * vec_integral_1)


@numba.jit(nopython=True, parallel=True, nogil=True)
def component_next_adams3(vec_integral: np.array = None,
                          vec_integral_1: np.array = None,
                          vec_integral_2: np.array = None,
                          vec_integral_3: np.array = None,
                          vec_integral_4: np.array = None,
                          vec_cur: np.array = None,
                          vec_cur_1: np.array = None,
                          vec_cur_2: np.array = None,
                          vec_cur_3: np.array = None,
                          delta: float = 0.1,
                          **kwargs):
    return vec_cur + delta * (adams_bashforth3_c0 * vec_integral + 
                              adams_bashforth3_c1 * vec_integral_1 + 
                              adams_bashforth3_c2 * vec_integral_2)


@numba.jit(nopython=True, parallel=True, nogil=True)
def component_next_adams4(vec_integral: np.array = None,
                          vec_integral_1: np.array = None,
                          vec_integral_2: np.array = None,
                          vec_integral_3: np.array = None,
                          vec_integral_4: np.array = None,
                          vec_cur: np.array = None,
                          vec_cur_1: np.array = None,
                          vec_cur_2: np.array = None,
                          vec_cur_3: np.array = None,
                          delta: float = 0.1,
                          **kwargs):
    return vec_cur + delta * (adams_bashforth4_c0 * vec_integral +
                              adams_bashforth4_c1 * vec_integral_1 +
                              adams_bashforth4_c2 * vec_integral_2 +
                              adams_bashforth4_c3 * vec_integral_3)


@numba.jit(nopython=True, parallel=True, nogil=True)
def component_next_adams5(vec_integral: np.array = None,
                          vec_integral_1: np.array = None,
                          vec_integral_2: np.array = None,
                          vec_integral_3: np.array = None,
                          vec_integral_4: np.array = None,
                          vec_cur: np.array = None,
                          vec_cur_1: np.array = None,
                          vec_cur_2: np.array = None,
                          vec_cur_3: np.array = None,
                          delta: float = 0.1,
                          **kwargs):
    return vec_cur + delta * (adams_bashforth5_c0 * vec_integral +
                              adams_bashforth5_c1 * vec_integral_1 +
                              adams_bashforth5_c2 * vec_integral_2 +
                              adams_bashforth5_c3 * vec_integral_3 +
                              adams_bashforth5_c4 * vec_integral_4)





class ProblemSolving:
    # ------------------------------------------------------------------------------------------
    number_of_frames: int  # Количество разбиений объекта
    number_of_steps: int  # Количество временных шагов эксперимента
    time_step: float  # Временной шаг для процесса во времени
    max_time: float  # Максимальное время расчёта
    timeline: np.array  # Массив временных отметок в виде массива numpy
    timeline_index: np.array  # Массив индексов временных отметок в виде массива numpy
    _time_current_: int  # Обозначение текущего времени отсчёта
    # ------------------------------------------------------------------------------------------
    free_function: Callable  # Функция правой части
    free_function_params: List[float]  # Параметры функции правой части
    orientation: np.array  # Массив типа numpy - нормированный вектор направления распространения волны
    component_next_func: Callable   # Функция для поиска следующей компоненты по определению первой производной
    # ------------------------------------------------------------------------------------------
    Q: np.array  # Массив данных поверхностных зарядов на объекте на каждый временной шаг
    P: np.array  # Массив данных временных производных пов. зарядов на объекте на каждый шаг
    G: np.array  # Массив данных векторов касательных токов на объекте на каждый шаг
    D: np.array  # Массив данных временных произв. векторов кас. токов на объекте на каждый шаг
    # ------------------------------------------------------------------------------------------
    frames: np.array  # Массив данных разбиений на решаемом объекте
    collocations: np.array  # Массив данных коллокаций на решаемом объекте
    norms: np.array  # Массив данных нормалей к разбиениям на объекте
    neighbours: np.array  # Массив индексов соседей для каждой ячейки разбиения
    tau: np.array  # Параметры запаздывания во времени по объекту
    coefficients_G1: np.array  # Массив данных коэффициентов для первого интегрального ядра G1: NxNx3
    coefficients_G2: np.array  # Массив данных коэффициентов для первого интегрального ядра G2: NxNx3
    coefficients_G3: np.array  # Массив данных коэффициентов для первого интегрального ядра G3: NxNx1

    def __init__(self,
                 time_step: float,
                 max_time: float,
                 number_of_frames: int,
                 free_function: Callable = f_function,
                 free_function_params: List[float] = None,
                 component_next_func: Callable = component_next_euler2,
                 orientation: np.array = np.array([0., 1., 0.]),
                 frames: np.array = None,
                 collocations: np.array = None,
                 norms: np.array = None,
                 neighbours: np.array = None,
                 collocation_distances: np.array = None,
                 coefficients_G1: np.array = None,
                 coefficients_G2: np.array = None,
                 coefficients_G3: np.array = None):
        """
        :param time_step: - step in time for non-stationary experiment
        :param max_time: - maximum time of observation
        :param number_of_frames: - number of frames in object
        """
        # Блок метаинформации
        self.time_step = time_step
        self.max_time = max_time
        self.timeline = np.arange(0, max_time, time_step)
        self.timeline_index = np.arange(0, self.timeline.shape[0], 1)
        self.number_of_frames = number_of_frames
        self.number_of_steps = len(self.timeline_index) - 1
        self._time_current_ = 1                                     # t
        # Блок входных данных задачи
        self.free_function = free_function
        self.free_function_params = free_function_params
        if orientation is None:
            self.orientation = np.array([0., 1., 0.])
        else:
            self.orientation = np.array(orientation) / np.sqrt(np.array(orientation) @ np.array(orientation))
        self.component_next_func = component_next_func
        # Блок выходных данных
        self.Q = np.zeros((self.number_of_frames, 2, 1))            # np.array of (N, t, 1) elements
        self.P = np.zeros((self.number_of_frames, 2, 1))            # np.array of (N, t, 1) elements
        self.G = np.zeros((self.number_of_frames, 2, 3))            # np.array of (N, t, 3) elements
        self.D = np.zeros((self.number_of_frames, 1, 3))            # np.array of (N, t-1, 3) elements
        # Блок обязательно считанных ранее данных фигур, касательно объекта, на котором решаем
        self.frames = frames
        self.collocations = collocations
        self.norms = norms
        self.neighbours = neighbours
        self.tau = collocation_distances / speed_of_light  # Вычисление параметров запаздывания по объекту
        self.coefficients_G1 = coefficients_G1
        self.coefficients_G2 = coefficients_G2
        self.coefficients_G3 = coefficients_G3

    def time_index_between(self,
                           time: float) -> tuple:
        """
        :param time: - scalar or np.array
        :return: tuple of np.arrays
        """
        return np.array(np.floor(time / self.time_step), dtype="int"), \
               np.array(np.ceil(time / self.time_step), dtype="int")

    def time_index(self,
                   time: float) -> np.array:
        """
        :param time: - scalar or np.array
        :return: np.array of indexes
        """
        return np.array(np.round(time / self.time_step), dtype="int")

    def get_Q_element(self,
                      time: float = 0.0,
                      frame: int = 0) -> float:
        """
        :param time: - scalar
        :param frame: - scalar
        :return: float scalar
        """
        if frame < 0 or frame >= self.number_of_frames:
            return 0.0
        if time < 0:
            return 0.0
        elif time > self.timeline[self._time_current_]:
            return 0.0
        else:
            time_ind = self.time_index(time)
            return self.Q[frame, time_ind, 0]

    def get_Q_vec(self,
                  time: float = 0.0) -> np.array:
        """
        :param time: - scalar
        :return: np.array of (N, ) elements
        """
        if time < 0:
            return np.zeros(self.number_of_frames)
        elif time > self.timeline[self._time_current_]:
            return np.zeros(self.number_of_frames)
        else:
            time_ind = self.time_index(time)
            return self.Q[:, time_ind, 0]

    def get_Q_element_ind(self,
                          index: int = 0,
                          frame: int = 0) -> float:
        """
        :param index: - scalar int
        :param frame: - scalar int
        :return: float scalar
        """
        if frame < 0 or frame >= self.number_of_frames:
            return 0.0
        if index < 0:
            return 0.0
        elif index > self._time_current_:
            return 0.0
        else:
            return self.Q[frame, index, 0]

    def get_Q_vec_ind(self,
                      index: int = 0) -> np.array:
        """
        :param index: - scalar int
        :return: np.array of (N, ) elements
        """
        if index < 0:
            return np.zeros(self.number_of_frames)
        elif index > self._time_current_:
            return np.zeros(self.number_of_frames)
        else:
            return self.Q[:, index, 0]

    def get_P_element(self,
                      time: float = 0.0,
                      frame: int = 0) -> float:
        """
        :param time: - scalar
        :param frame: - scalar
        :return: float scalar
        """
        if frame < 0 or frame >= self.number_of_frames:
            return 0.0
        if time < 0:
            return 0.0
        elif time > self.timeline[self._time_current_]:
            return 0.0
        else:
            time_ind = self.time_index(time)
            return self.P[frame, time_ind, 0]

    def get_P_vec(self,
                  time: float = 0.0) -> np.array:
        """
        :param time: - scalar
        :return: np.array of (N, ) elements
        """
        if time < 0:
            return np.zeros(self.number_of_frames)
        elif time > self.timeline[self._time_current_]:
            return np.zeros(self.number_of_frames)
        else:
            time_ind = self.time_index(time)
            return self.P[:, time_ind, 0]

    def get_P_element_ind(self,
                          index: int = 0,
                          frame: int = 0) -> float:
        """
        :param index: - scalar int
        :param frame: - scalar int
        :return: float scalar
        """
        if frame < 0 or frame >= self.number_of_frames:
            return 0.0
        if index < 0:
            return 0.0
        elif index > self._time_current_:
            return 0.0
        else:
            return self.P[frame, index, 0]

    def get_P_vec_ind(self,
                      index: int = 0) -> np.array:
        """
        :param index: - scalar int
        :return: np.array of (N, ) elements
        """
        if index < 0:
            return np.zeros(self.number_of_frames)
        elif index > self._time_current_:
            return np.zeros(self.number_of_frames)
        else:
            return self.P[:, index, 0]

    def get_G_element(self,
                      time: float = 0.0,
                      frame: int = 0) -> np.array:
        """
        :param time: - scalar
        :param frame: - scalar
        :return: np.array of 3 numbers in (i, j, k)
        """
        if frame < 0 or frame >= self.number_of_frames:
            return np.zeros(3)
        if time < 0:
            return np.zeros(3)
        elif time > self.timeline[self._time_current_]:
            return np.zeros(3)
        else:
            time_ind = self.time_index(time)
            return self.G[frame, time_ind]

    def get_G_vec(self,
                  time: float = 0.0) -> np.array:
        """
        :param time: - scalar
        :return: np.array of (N, 3) elements
        """
        if time < 0:
            return np.zeros((self.number_of_frames, 3))
        elif time > self.timeline[self._time_current_]:
            return np.zeros((self.number_of_frames, 3))
        else:
            time_ind = self.time_index(time)
            return self.G[:, time_ind]

    def get_G_element_ind(self,
                          index: int = 0,
                          frame: int = 0) -> np.array:
        """
        :param index: - scalar int
        :param frame: - scalar int
        :return: np.array of 3 numbers in (i, j, k)
        """
        if frame < 0 or frame >= self.number_of_frames:
            return np.zeros(3)
        if index < 0:
            return np.zeros(3)
        elif index > self._time_current_:
            return np.zeros(3)
        else:
            return self.G[frame, index]

    def get_G_vec_ind(self,
                      index: int = 0) -> np.array:
        """
        :param index: - scalar int
        :return: np.array of (N, 3) elements
        """
        if index < 0:
            return np.zeros((self.number_of_frames, 3))
        elif index > self._time_current_:
            return np.zeros((self.number_of_frames, 3))
        else:
            return self.G[:, index]

    def get_D_element(self,
                      time: float = 0.0,
                      frame: int = 0) -> np.array:
        """
        :param time: - scalar
        :param frame: - scalar
        :return: np.array of 3 numbers in (i, j, k)
        """
        if frame < 0 or frame >= self.number_of_frames:
            return np.zeros(3)
        if time < 0:
            return np.zeros(3)
        elif time > self.timeline[self._time_current_]:
            return np.zeros(3)
        else:
            time_ind = self.time_index(time)
            return self.D[frame, time_ind]

    def get_D_vec(self,
                  time: float = 0.0) -> np.array:
        """
        :param time: - scalar
        :return: np.array of (N, 3) elements
        """
        if time < 0:
            return np.zeros((self.number_of_frames, 3))
        elif time > self.timeline[self._time_current_]:
            return np.zeros((self.number_of_frames, 3))
        else:
            time_ind = self.time_index(time)
            return self.D[:, time_ind]

    def get_D_element_ind(self,
                          index: int = 0,
                          frame: int = 0) -> np.array:
        """
        :param index: - scalar int
        :param frame: - scalar int
        :return: np.array of 3 numbers in (i, j, k)
        """
        if frame < 0 or frame >= self.number_of_frames:
            return np.zeros(3)
        if index < 0:
            return np.zeros(3)
        elif index > self._time_current_ - 1:
            return np.zeros(3)
        else:
            return self.D[frame, index]

    def get_D_vec_ind(self,
                      index: int = 0) -> np.array:
        """
        :param index: - scalar int
        :return: np.array of 3 numbers in (i, j, k)
        """
        if index < 0:
            return np.zeros((self.number_of_frames, 3))
        elif index > self._time_current_ - 1:
            return np.zeros((self.number_of_frames, 3))
        else:
            return self.D[:, index]

    def time_current_get(self, mode: int = 0):
        """
        :param mode: integer key for interpret what to return
        :return: if mode == 0 returns integer index, else returns real time count
        """
        if mode == 0:
            return self._time_current_
        else:
            return self.timeline[self._time_current_]

    def time_current_inc(self):
        if self._time_current_ != self.max_time:
            self._time_current_ += 1

    def set_State(self,
                  new_Q: np.array = None,
                  new_P: np.array = None,
                  new_G: np.array = None,
                  new_D: np.array = None):
        if new_Q is None:
            self.Q = np.concatenate((self.Q, np.zeros((self.number_of_frames, 1))), axis=1)
        else:
            self.Q = np.concatenate((self.Q, new_Q.reshape((self.number_of_frames, 1))), axis=1)
        if new_P is None:
            self.P = np.concatenate((self.P, np.zeros((self.number_of_frames, 1))), axis=1)
        else:
            self.P = np.concatenate((self.P, new_P.reshape((self.number_of_frames, 1))), axis=1)
        if new_G is None:
            self.G = np.concatenate((self.G, np.zeros((self.number_of_frames, 3))), axis=1)
        else:
            self.G = np.concatenate((self.G, new_G.reshape((self.number_of_frames, 3))), axis=1)
        if new_D is None:
            self.D = np.concatenate((self.D, np.zeros((self.number_of_frames, 3))), axis=1)
        else:
            self.D = np.concatenate((self.D, new_D.reshape((self.number_of_frames, 3))), axis=1)

        self.time_current_inc()
