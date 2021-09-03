import numpy as np
import figure
from integral import integrate_func
import G_functions
import compute_coeffs
import grid
import charge

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Считывание данных
    figure_plate_40_40 = figure.Figure("d:\\Python\\MaxwellIntegralEq\\figures\\plate_12_12.dat")

    # Загрузка коэффициентов
    G1_load = compute_coeffs.coeffs_load("d:\\Python\\MaxwellIntegralEq\\coeffs\\plate_12_12\\G1_plate_12_12.txt")
    G2_load = compute_coeffs.coeffs_load("d:\\Python\\MaxwellIntegralEq\\coeffs\\plate_12_12\\G2_plate_12_12.txt")
    G3_load = compute_coeffs.coeffs_load("d:\\Python\\MaxwellIntegralEq\\coeffs\\plate_12_12\\G3_plate_12_12.txt")
    G3_load = G3_load.reshape((G3_load.shape[0], G3_load.shape[1]))
    # Шаг по времени
    curant_ts = figure_plate_40_40.get_Curant() / 6
    # Максимальное время
    max_time = 3 / (3*1e8)

    print()
    print("steps " + str(max_time / curant_ts))
    print()
    #
    figure_plate_40_40.print_details()

    # Инициализация класса просчёта схемы
    Charge_1 = charge.Charge(time_step=curant_ts,
                             max_time=max_time,
                             max_diameter=figure_plate_40_40.max_diameter[0],
                             G1=G1_load,
                             G2=G2_load,
                             G3=G3_load,
                             tau=figure_plate_40_40.collocation_distances[0],
                             norms=figure_plate_40_40.norms[0],
                             frames=figure_plate_40_40.frames[0],
                             squares=figure_plate_40_40.squares[0],
                             collocations=figure_plate_40_40.collocations[0],
                             total_number_of_frames=figure_plate_40_40.total_frames_in_objects[0],
                             div_approx=3)

    # Шаги во времени
    for i in range(0, int(np.ceil(max_time/curant_ts))):
        Charge_1.step_in_time()



