import os
import json
import numpy as np
import customalgebra as ca

class Config:
    # Конструктор класса читать конфиг
    def __init__(self, config_path):
        self.config_path = config_path                      # Путь к конфигу
        with open(self.config_path, "r") as json_file:      # Читать конфиг как json
            self.config_dict = json.load(json_file)         # Положить json в словарь

        # Шаг по времени
        self.time_step = float(self.config_dict['options']['time_step'])

        # Ориентация внешнего поля
        self.E0 = np.array(list(map(float, self.config_dict['options']['E0'])))
        self.E0 = self.E0 / ca.L2(self.E0)                            # Нормируем

        # Ориентация направления изменения эл. поля
        self.k0 = np.array(list(map(float, self.config_dict['options']['k0'])))
        self.k0 = self.k0 / ca.L2(self.k0)                            # Нормируем

        self.D = float(self.config_dict['options']['D'])                     # Масштаб импульса
        self.d = float(self.config_dict['options']['d'])                     # Смещение импульса во времени
        self.speed_of_light = float(self.config_dict['options']['speed_of_light'])         # Скорость света
        self.max_time = float(self.config_dict['options']['max_time'])       # Максимальное время рассмотрения

        # Пути к папкам и файлам
        self.path_coeffs = self.config_dict['paths']['path_coeffs']         # Путь к папке ядер объектов
        self.path_figures = self.config_dict['paths']['path_figures']       # Путь к папке разбиений объектов
        self.path_grids = self.config_dict['paths']['path_grids']           # Путь к папке сеток
        self.path_logs = self.config_dict['paths']['path_logs']             # Путь к папке логов
        self.figure_file = self.config_dict['paths']['figure_file']         # Путь к выбранному объекту
        self.figure_logs = self.config_dict['paths']['figure_logs']         # Путь к логам выбранного объекта
        self.figure_coeffs = self.config_dict['paths']['figure_coeffs']     # Путь к ядрам выбранного объекта