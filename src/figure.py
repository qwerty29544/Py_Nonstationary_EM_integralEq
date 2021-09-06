import numpy as np


def frame_center(frames):
    manual_shape = 2                # Размерность массива рамки как минимум Ndim = 2
    len_shape = len(frames.shape)   # Размерность входного массива рамок
    if len_shape < manual_shape:    # Проверка размерности входного массива
        return np.zeros((frames.shape[0], 3))
    center = np.mean(frames, len_shape - manual_shape)  # Вычисление массива центров рамок
    return center


class Figure:
    def __init__(self, filename, mode="quadr"):
        self.filename = filename
        self.number_of_objects = 0
        self.number_of_modules = []
        self.number_of_frames = []
        self.total_frames_in_objects = []
        self.bounded = []
        self.frames = []
        self.begend = []
        self.mode = mode
        self.total_frames_number = 0
        self.total_modules_number = 0

        self._read_file_(self.filename)
        self._total_stats_count_()
        self._frames_management_()

        self.TQPR = []
        self.squares = []
        self.norms = []
        self.collocations = []
        self.local_basis = []
        self.max_diameter = []
        self.min_R = []

        # Вызов функции подстчёта данных величин
        self._TQPR_norms_squares_colloc_calc_()
        self._max_diameter_()
        self._min_R_colloc_()

        # Можно считать и не хранить это Г*****
        self.collocation_distances = []
        self._colloc_dist_compute_()

        self.neighbours = []

    def _read_file_(self, filename):
        f = open(filename, "r")
        self.number_of_objects = int(f.readline())
        for obj in range(self.number_of_objects):
            self.number_of_modules.append(int(f.readline()))
            frames = []
            bound = []
            points_object = []
            for module in range(self.number_of_modules[obj]):
                bound.append(int(f.readline()))
                frames.append(int(f.readline()))
                f.readline()
                points_module = []
                for frame in range(frames[module]):
                    frame_local_list = list(map(float, f.readline().split()))
                    frame_points = []
                    if self.mode == "tri":
                        for point in range(3):
                            frame_points.append(frame_local_list[(3 * point):(3 * (point + 1))])
                    else:
                        for point in range(4):
                            frame_points.append(frame_local_list[(3 * point):(3 * (point + 1))])
                    points_module.append(frame_points)
                points_object.append(points_module)
            self.number_of_frames.append(frames)
            self.bounded.append(bound)
            self.frames.append(points_object)
        f.close()

    def _total_stats_count_(self):
        for obj in range(self.number_of_objects):
            self.total_modules_number += self.number_of_modules[obj]
            module_frames = 0
            for module in range(self.number_of_modules[obj]):
                module_frames += self.number_of_frames[obj][module]
                self.total_frames_number += self.number_of_frames[obj][module]
            self.total_frames_in_objects.append(module_frames)

    def _frames_management_(self):
        frames_obj = []
        for obj in range(self.number_of_objects):
            frames_modules = []
            for module in range(self.number_of_modules[obj]):
                for frame in range(self.number_of_frames[obj][module]):
                    framed = []
                    if self.mode == "tri":
                        for point in range(3):
                            framed.append(self.frames[obj][module][frame][point])
                    else:
                        for point in range(4):
                            framed.append(self.frames[obj][module][frame][point])
                    frames_modules.append(framed)
            frames_obj.append(np.array(frames_modules))
        self.frames = frames_obj

    def _TQPR_norms_squares_colloc_calc_(self):
        collocations_res = []
        TQPR_res = []
        norms_res = []
        squares_res = []
        local_basis_res = []
        for obj in range(self.number_of_objects):
            TQPR_obj = []
            norms_obj = []
            squares_obj = []
            collocations_obj = []
            local_basis_obj = []
            for frame in range(self.total_frames_in_objects[obj]):
                TQPR_frame = []
                local_basis_frame = []
                TQ = ((self.frames[obj][frame][2] + self.frames[obj][frame][1]) / 2. -
                      (self.frames[obj][frame][3] + self.frames[obj][frame][0]) / 2.)
                PR = ((self.frames[obj][frame][3] + self.frames[obj][frame][2]) / 2. -
                      (self.frames[obj][frame][1] + self.frames[obj][frame][0]) / 2.)
                norm = np.cross(TQ, PR)
                square = np.sqrt(norm @ norm)
                norm = norm / square
                colloc = (self.frames[obj][frame][0] +
                          self.frames[obj][frame][1] +
                          self.frames[obj][frame][2] +
                          self.frames[obj][frame][3]) / 4.

                tau1 = TQ.copy()
                tau3 = norm.copy()
                tau2 = np.cross(TQ, norm)
                square_tau2 = np.sqrt(tau2 @ tau2)
                tau2 = tau2 / square_tau2

                local_basis_frame.append(tau1)
                local_basis_frame.append(tau2)
                local_basis_frame.append(tau3)
                local_basis_obj.append(local_basis_frame)

                # TQPR
                TQPR_frame.append(list(TQ))
                TQPR_frame.append(list(PR))
                TQPR_obj.append(TQPR_frame)
                # Площадь
                squares_obj.append(square)
                # Нормы
                norms_obj.append(list(norm))
                collocations_obj.append(colloc)

            # Записываем для каждого объекта
            TQPR_res.append(np.array(TQPR_obj))
            norms_res.append(np.array(norms_obj))
            squares_res.append(np.array(squares_obj))
            collocations_res.append(np.array(collocations_obj))
            local_basis_res.append(np.array(local_basis_obj))

        # Присваиваем полям класса
        self.TQPR = TQPR_res
        self.norms = norms_res
        self.squares = squares_res
        self.collocations = collocations_res
        self.local_basis = local_basis_res

    def _max_diameter_(self):
        max_d = []
        for obj in range(self.number_of_objects):
            max_diameter = 0
            for frame in range(self.total_frames_in_objects[obj]):
                d1 = self.frames[obj][frame][0] - self.frames[obj][frame][1]
                d2 = self.frames[obj][frame][1] - self.frames[obj][frame][2]
                d3 = self.frames[obj][frame][2] - self.frames[obj][frame][3]
                d4 = self.frames[obj][frame][3] - self.frames[obj][frame][0]
                d5 = self.frames[obj][frame][0] - self.frames[obj][frame][2]
                d6 = self.frames[obj][frame][3] - self.frames[obj][frame][1]
                max_d_frame = max(np.sqrt(d1 @ d1),
                                  np.sqrt(d2 @ d2),
                                  np.sqrt(d3 @ d3),
                                  np.sqrt(d4 @ d4),
                                  np.sqrt(d5 @ d5),
                                  np.sqrt(d6 @ d6))
                if max_diameter < max_d_frame:
                    max_diameter = max_d_frame
            max_d.append(max_diameter)
        self.max_diameter = max_d

    def _min_R_colloc_(self):
        min_R_glob = []
        for obj in range(self.number_of_objects):
            d = self.collocations[obj][0] - self.collocations[obj][1]
            min_R_obj = np.sqrt(d @ d)
            for frame in range(self.total_frames_in_objects[obj]):
                for associative in range(frame + 1, self.total_frames_in_objects[obj]):
                    d = self.collocations[obj][frame] - self.collocations[obj][associative]
                    d = np.sqrt(d @ d)
                    if min_R_obj > d:
                        min_R_obj = d
            min_R_glob.append(min_R_obj)
        self.min_R = min_R_glob

    # Возможна матричная реализация и чисто расчётная без хранения
    def _colloc_dist_compute_(self):
        coll_dist_obj = []
        for obj in range(self.number_of_objects):
            coll_in_obj = []
            for i in range(self.total_frames_in_objects[obj]):
                range_between = []
                for j in range(self.total_frames_in_objects[obj]):
                    vec = self.collocations[obj][i] - self.collocations[obj][j]
                    range_between.append(np.sqrt(vec @ vec))
                coll_in_obj.append(range_between)
            coll_dist_obj.append(np.array(coll_in_obj))
        self.collocation_distances = coll_dist_obj

    # Нахождение соседей по разбиениям
    def _neighbours_find_(self):
        brothers = []
        for obj in range(self.number_of_objects):
            neighbours_in_obj = np.zeros((self.total_frames_in_objects[obj], 4)) - 1
            for frame in range(self.total_frames_in_objects[obj]):
                fl = np.zeros(4)
                for j in range(self.total_frames_in_objects[obj]):
                    if ((self.frames[obj][frame][0] == self.frames[obj][j][3]) and
                            (self.frames[obj][frame][1] == self.frames[obj][j][2])):
                        neighbours_in_obj[frame][0] = j
                        fl[0] = 1
                    if ((self.frames[obj][frame][1] == self.frames[obj][j][0]) and
                            (self.frames[obj][frame][2] == self.frames[obj][j][3])):
                        neighbours_in_obj[frame][1] = j
                        fl[1] = 1
                    if ((self.frames[obj][frame][2] == self.frames[obj][j][1]) and
                            (self.frames[obj][frame][3] == self.frames[obj][j][0])):
                        neighbours_in_obj[frame][2] = j
                        fl[2] = 1
                    if ((self.frames[obj][frame][3] == self.frames[obj][j][2]) and
                            (self.frames[obj][frame][0] == self.frames[obj][j][1])):
                        neighbours_in_obj[frame][3] = j
                        fl[3] = 1
                    if (np.prod(fl) == 1):
                        break
            brothers.append(neighbours_in_obj)
        self.neighbours = brothers


    def print_details(self):
        print("|| -----------------------------------------------------||\n")
        print("Number of objects in figure: " + str(self.number_of_objects))
        print("Total number of modules in fiqure is " + str(self.total_modules_number))
        print("Total number of frames in figure is " + str(self.total_frames_number))
        print("\n|| -----------------------------------------------------||")
        for obj in range(self.number_of_objects):
            print("\n\tNumber of modules in object #" + str(obj + 1) + ": " + str(self.number_of_modules[obj]))
            for module in range(self.number_of_modules[obj]):
                print("\t\tIn module #" + str(module + 1) + " we have " + str(self.number_of_frames[obj][module]) + " frames")

    def get_Curant(self):
        return min(self.min_R) / (3 * 1e8)