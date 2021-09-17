import numpy as np
import numba
from typing import List, Callable
from scipy.constants import speed_of_light
from divergence_approx import div_vec_approx, gradient_vec


@numba.jit(nopython=True, parallel=True, nogil=True)
def f_function(params: List[float]) -> float:
    """
    @param params: List of float custom function parameters
    @return float scalar
    @examples:
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
    @param time_right: - right border in time
    @param time_between: - time between borders
    @param time_step: - step in time between borders
    @return _Iterable_(alpha, beta)
    @examples:
        time_1 = 0.55
        time_2 = 2.17
        time_between = 1.98
        time_step = time_2 - time_1
        alpha, beta = proportion_linear(time_2, time_between, time_step)
        assert alpha + beta == 1
        assert alpha > beta
    """
    beta = (time_between - time_right) / time_step
    return (1 - beta), beta


class ProblemSolving:
    # ------------------------------------------------------------------------------------------
    number_of_frames: int               # Количество разбиений объекта
    number_of_steps: int                # Количество временных шагов эксперимента
    time_step: float                    # Временной шаг для процесса во времени
    max_time: float                     # Максимальное время расчёта
    timeline: np.array                  # Массив временных отметок в виде массива numpy
    timeline_index: np.array            # Массив индексов временных отметок в виде массива numpy
    # ------------------------------------------------------------------------------------------
    free_function: Callable             # Функция правой части
    free_function_params: List[float]   # Параметры функции правой части
    orientation: np.array               # Массив типа numpy - нормированный вектор направления распространения волны
    # ------------------------------------------------------------------------------------------
    Q: np.array                         # Массив данных поверхностных зарядов на объекте на каждый временной шаг
    P: np.array                         # Массив данных временных производных пов. зарядов на объекте на каждый шаг
    G: np.array                         # Массив данных векторов касательных токов на объекте на каждый шаг
    D: np.array                         # Массив данных временных произв. векторов кас. токов на объекте на каждый шаг
    # ------------------------------------------------------------------------------------------
    frames: np.array                    # Массив данных разбиений на решаемом объекте
    collocations: np.array              # Массив данных коллокаций на решаемом объекте
    norms: np.array                     # Массив данных нормалей к разбиениям на объекте
    neighbours: np.array                # Массив индексов соседей для каждой ячейки разбиения
    tau: np.array                       # Параметры запаздывания во времени по объекту
    coefficients_G1: np.array           # Массив данных коэффициентов для первого интегрального ядра G1: NxNx3
    coefficients_G2: np.array           # Массив данных коэффициентов для первого интегрального ядра G2: NxNx3
    coefficients_G3: np.array           # Массив данных коэффициентов для первого интегрального ядра G3: NxNx1

    def __init__(self,
                 time_step: float,
                 max_time: float,
                 number_of_frames: int,
                 free_function: Callable = None,
                 free_function_params: List[float] = None,
                 orientation: np.array = None,
                 frames: np.array = None,
                 collocations: np.array = None,
                 norms: np.array = None,
                 neighbours: np.array = None,
                 collocation_distances: np.array = None,
                 coefficients_G1: np.array = None,
                 coefficients_G2: np.array = None,
                 coefficients_G3: np.array = None):
        """
        @param time_step: - step in time for non-stationary experiment
        @param max_time: - maximum time of observation
        @param number_of_frames: - number of frames in object
        """
        # Блок метаинформации
        self.time_step = time_step
        self.max_time = max_time
        self.timeline = np.arange(0, max_time, time_step)
        self.timeline_index = np.arange(0, self.timeline.shape[0], 1)
        self.number_of_frames = number_of_frames
        self.number_of_steps = len(self.timeline_index) - 1
        # Блок входных данных задачи
        self.free_function = free_function
        self.free_function_params = free_function_params
        if orientation is None:
            self.orientation = np.array([0., 1., 0.])
        else:
            self.orientation = np.array(orientation) / np.sqrt(np.array(orientation) @ np.array(orientation))
        # Блок выходных данных
        self.Q = np.zeros((self.number_of_frames, 2, 1))
        self.P = np.zeros((self.number_of_frames, 2, 1))
        self.G = np.zeros((self.number_of_frames, 2, 3))
        self.D = np.zeros((self.number_of_frames, 1, 3))
        # Блок обязательно считанных ранее данных фигур, касательно объекта, на котором решаем
        self.frames = frames
        self.collocations = collocations
        self.norms = norms
        self.neighbours = neighbours
        self.tau = collocation_distances / speed_of_light       # Вычисление параметров запаздывания по объекту
        self.coefficients_G1 = coefficients_G1
        self.coefficients_G2 = coefficients_G2
        self.coefficients_G3 = coefficients_G3

    def time_index_between(self,
                           time: float) -> tuple:
        """
        @param time - scalar or np.array
        @return tuple of np.arrays
        """
        return np.array(np.floor(time / self.time_step), dtype="int"), \
               np.array(np.ceil(time / self.time_step), dtype="int")

    def time_index(self,
                   time: float) -> np.array:
        """
        @param time - scalar or np.array
        @return np.array of indexes
        """
        return np.array(np.round(time / self.time_step), dtype="int")

    def get_Q_element(self,
                      time: float,
                      frame: int) -> float:
        """
        @param time - scalar
        @param frame - scalar
        @return float scalar
        """
        if time < 0:
            return 0.0
        elif time > self.max_time:
            return 0.0
        else:
            time_ind = self.time_index(time)
            return self.Q[frame, time_ind, 0]

    def get_P_element(self,
                      time: float,
                      frame: int) -> float:
        """
        @param time - scalar
        @param frame - scalar
        @return float scalar
        """
        if time < 0:
            return 0.0
        elif time > self.max_time:
            return 0.0
        else:
            time_ind = self.time_index(time)
            return self.P[frame, time_ind, 0]

    def get_G_element(self,
                      time: float,
                      frame: int) -> np.array:
        """
        @param time - scalar
        @param frame - scalar
        @return np.array of 3 numbers in (i, j, k)
        """
        if time < 0:
            return np.zeros(3)
        elif time > self.max_time:
            return np.zeros(3)
        else:
            time_ind = self.time_index(time)
            return self.G[frame, time_ind]

    def get_D_element(self,
                      time: float,
                      frame: int) -> np.array:
        """
        @param time - scalar
        @param frame - scalar
        @return np.array of 3 numbers in (i, j, k)
        """
        if time < 0:
            return np.zeros(3)
        elif time > self.max_time:
            return np.zeros(3)
        else:
            time_ind = self.time_index(time)
            return self.D[frame, time_ind]
