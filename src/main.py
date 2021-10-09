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
    create_base_proj_structure()
    figure_plate_24_24 = figure.Figure(os.path.join(FIGURES_PATH, "plate_24_24.dat"))
    # # Загрузка коэффициентов
    G1_load = compute_coeffs.coeffs_load(os.path.join(COEFFS_PATH, "plate_24_24", "G1_plate_24_24.txt"))
    G2_load = compute_coeffs.coeffs_load(os.path.join(COEFFS_PATH, "plate_24_24", "G2_plate_24_24.txt"))
    G3_load = compute_coeffs.coeffs_load(os.path.join(COEFFS_PATH, "plate_24_24", "G3_plate_24_24.txt"))
    G3_load = G3_load.reshape((G3_load.shape[0], G3_load.shape[1]))
    # Шаг по времени
    curant_ts = figure_plate_24_24.get_Curant() / 6
    # # Максимальное время
    max_time = 3 / (3*1e8)
    
    print()
    print("steps " + str(max_time / curant_ts))
    print()
    # #
    figure_plate_24_24.print_details()
    #
    # # Инициализация класса просчёта схемы
    Charge_1 = charge.Charge(time_step=curant_ts,
                             max_time=max_time,
                             max_diameter=figure_plate_24_24.max_diameter[0],
                             G1=G1_load,
                             G2=G2_load,
                             G3=G3_load,
                             tau=figure_plate_24_24.collocation_distances[0],
                             norms=figure_plate_24_24.norms[0],
                             frames=figure_plate_24_24.frames[0],
                             squares=figure_plate_24_24.squares[0],
                             collocations=figure_plate_24_24.collocations[0],
                             total_number_of_frames=figure_plate_24_24.total_frames_in_objects[0],
                             div_approx=3)
    #
    # # Шаги во времени
    for i in range(0, int(np.ceil(max_time/curant_ts))):
        Charge_1.step_in_time()



