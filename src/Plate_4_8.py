from integral import integrate_func
from G_functions import G1_func, G2_func
from figure import Figure
import compute_coeffs
import os
from envir_paths import *

create_base_proj_structure()

N = [16, 24, 32, 40, 48, 56, 64, 72, 80, 100]
list_of_figures = []
for n in N:
    figure = "_".join(["plate", n, n])
    list_of_figures.append(figure)

for fig in list_of_figures:
    plate = Figure(os.path.join(FIGURES_PATH, fig + ".dat"))
    coeffs_G1 = compute_coeffs.compute_coeffs(frame=plate.frames[0],
                                              collocation=plate.collocations[0],
                                              number_of_frames=plate.total_frames_in_objects[0],
                                              integration_method=integrate_func,
                                              integral_function=G1_func,
                                              max_diameter=plate.max_diameter[0],
                                              ndim=3,
                                              n_vertex=4,
                                              num_slices=4,
                                              num_slices_diag=3)
    print("G1 of " + fig)
    compute_coeffs.coeffs_save(coeffs=coeffs_G1,
                               filename=os.path.join(COEFFS_PATH, fig, "G1_" + fig + ".txt"))
    coeffs_G2 = compute_coeffs.compute_coeffs(frame=plate.frames[0],
                                              collocation=plate.collocations[0],
                                              number_of_frames=plate.total_frames_in_objects[0],
                                              integration_method=integrate_func,
                                              integral_function=G2_func,
                                              max_diameter=plate.max_diameter[0],
                                              ndim=3,
                                              n_vertex=4,
                                              num_slices=4,
                                              num_slices_diag=3)
    compute_coeffs.coeffs_save(coeffs=coeffs_G2,
                               filename=os.path.join(COEFFS_PATH, fig, "G2_" + fig + ".txt"))
    print("G2 of " + fig)
    coeffs_G3 = compute_coeffs.compute_G3_coefficients(frame=plate.frames[0],
                                                       collocation=plate.collocations[0],
                                                       number_of_frames=plate.total_frames_in_objects[0],
                                                       integration_method=integrate_func,
                                                       max_diameter=plate.max_diameter[0],
                                                       n_vertex=4,
                                                       num_slices=4)
    compute_coeffs.coeffs_save(coeffs=coeffs_G3,
                               filename=os.path.join(COEFFS_PATH, fig, "G3_" + fig + ".txt"))
    print("G3 of " + fig)