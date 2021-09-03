import numpy as np


class Grid:
    def __init__(self, filename):
        self.coord_grid = []            # Точки сетки
        self.N_x = 0                    # Число разбиений по X
        self.N_y = 0                    # Число разбиений по Y
        self.N_z = 0                    # Число разбиений по Z
        self.step_x = 0                 # Шаг по X
        self.step_y = 0                 # Шаг по Y
        self.step_z = 0                 # Шаг по X
        self.max_x = 0                  # Максимум значения координат по X
        self.max_y = 0                  # Максимум значения координат по Y
        self.max_z = 0                  # Максимум значения координат по Z
        self.min_x = 0                  # Минимум значений координат по X
        self.min_y = 0                  # Минимум значений координат по Y
        self.min_z = 0                  # Минимум значений координат по Z
        self.filename = filename        # Имя файла из которого считали
        self._read_file_()              # Локальная функция конструктора, вызывать не надо

    def _read_file_(self):
        file = open(self.filename, "r")
        long, lat = map(int, file.readline().split())
        coords_grid = []
        for x_grid in range(long):
            for y_grid in range(lat):
                coords_grid.append(list(map(float, file.readline().split())))
        self.coord_grid = np.array(coords_grid)
        file.close()

        self.max_x = np.max(self.coord_grid[:, 0])
        self.max_y = np.max(self.coord_grid[:, 1])
        self.max_z = np.max(self.coord_grid[:, 2])

        self.min_x = np.min(self.coord_grid[:, 0])
        self.min_y = np.min(self.coord_grid[:, 1])
        self.min_z = np.min(self.coord_grid[:, 2])

        if (self.min_x == self.max_x):
            self.N_x = 0
            self.N_y = long
            self.N_z = lat
            self.step_y = (self.max_y - self.min_y) / (self.N_y - 1)
            self.step_z = (self.max_z - self.min_z) / (self.N_z - 1)
            self.step_x = 0
        elif (self.min_y == self.max_y):
            self.N_x = long
            self.N_y = 0
            self.N_z = lat
            self.step_z = (self.max_z - self.min_z) / (self.N_z - 1)
            self.step_x = (self.max_x - self.min_x) / (self.N_x - 1)
            self.step_y = 0
        elif (self.min_z == self.max_z):
            self.N_x = long
            self.N_y = lat
            self.N_z = 0
            self.step_y = (self.max_y - self.min_y) / (self.N_y - 1)
            self.step_x = (self.max_x - self.min_x) / (self.N_x - 1)
            self.step_z = 0
