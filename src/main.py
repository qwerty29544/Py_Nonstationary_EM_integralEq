import numpy as np
import figure
from integral import integrate_func
import G_functions
import compute_coeffs
import grid
import charge
import os
from envir_paths import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Считывание данных
    # figure_sphere = figure.Figure("d:\\Python\\Py_Nonstationary_EM_integralEq\\figures\\sphere.dat")
    # G3_coeff = compute_coeffs.compute_G3_coefficients(frame = figure_sphere.frames[0],
    #                                                   collocation=figure_sphere.collocations[0],
    #                                                   number_of_frames=figure_sphere.total_frames_in_objects[0],
    #                                                   integration_method=integrate_func,
    #                                                   max_diameter=figure_sphere.max_diameter[0],
    #                                                   n_vertex=4,
    #                                                   num_slices=4)
    # print("one axis")
    # print(np.sum(G3_coeff, axis=1) * (-1) * (9e16))
    # print("zero axis")
    # print(np.sum(G3_coeff, axis=0) * (-1) * (9e16))
    # create_base_proj_structure()
    config = Config("C:\\Users\\MariaRemark\\PycharmProjects\\Py_Nonstationary_EM_integralEq\\config.json")
    fig = figure.Figure(config.figure_file)
    # Загрузка коэффициентов
    G1_load = compute_coeffs.coeffs_load(os.path.join(config.figure_coeffs, "G1_plate_20_20.txt"))
    G2_load = compute_coeffs.coeffs_load(os.path.join(config.figure_coeffs, "G2_plate_20_20.txt"))
    G3_load = compute_coeffs.coeffs_load(os.path.join(config.figure_coeffs, "G3_plate_20_20.txt"))
    G3_load = G3_load.reshape((G3_load.shape[0], G3_load.shape[1]))

    if config.time_step > fig.get_Curant():
        assert False

    print()
    print("steps " + str(config.max_time / config.time_step))
    print()

    fig.print_details()

    # Инициализация класса просчёта схемы
    Charge_1 = charge.Charge(config_path=config.config_path,
                             max_diameter=fig.max_diameter[0],
                             G1=G1_load,
                             G2=G2_load,
                             G3=G3_load,
                             tau=fig.collocation_distances[0],
                             norms=fig.norms[0],
                             frames=fig.frames[0],
                             squares=fig.squares[0],
                             collocations=fig.collocations[0],
                             neighbors=fig.neighbours[0],
                             total_number_of_frames=fig.total_frames_in_objects[0])

    # Шаги во времени
    for i in range(0, int(np.ceil(config.max_time/config.time_step))):
        Charge_1.step_in_time()




