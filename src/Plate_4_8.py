from integral import integrate_func
from G_functions import G1_func, G2_func, G3_func, integr_G3
from figure import Figure
import compute_coeffs


plate_12_12 = Figure("d:\\Python\\Py_Nonstationary_EM_integralEq\\figures\\plate_12_12.dat")


coeffs_G3 = compute_coeffs.compute_G3_coefficients(frame=plate_12_12.frames[0],
                                                   collocation=plate_12_12.collocations[0],
                                                   number_of_frames=plate_12_12.total_frames_in_objects[0],
                                                   integration_method=integrate_func,
                                                   max_diameter=plate_12_12.max_diameter[0],
                                                   n_vertex=4,
                                                   num_slices=4)
print("G3")
compute_coeffs.coeffs_save(coeffs_G3, "d:\\Python\\Py_Nonstationary_EM_integralEq\\coeffs\\plate_12_12\\G3_plate_12_12.txt")


coeffs_G1 = compute_coeffs.compute_coeffs(frame=plate_12_12.frames[0],
                                          collocation=plate_12_12.collocations[0],
                                          number_of_frames=plate_12_12.total_frames_in_objects[0],
                                          integration_method=integrate_func,
                                          integral_function=G1_func,
                                          max_diameter=plate_12_12.max_diameter[0],
                                          ndim=3,
                                          n_vertex=4,
                                          num_slices=4,
                                          num_slices_diag=3)
print("G1")
compute_coeffs.coeffs_save(coeffs_G1, "d:\\Python\\Py_Nonstationary_EM_integralEq\\coeffs\\plate_12_12\\G1_plate_12_12.txt")

coeffs_G2 = compute_coeffs.compute_coeffs(frame=plate_12_12.frames[0],
                                          collocation=plate_12_12.collocations[0],
                                          number_of_frames=plate_12_12.total_frames_in_objects[0],
                                          integration_method=integrate_func,
                                          integral_function=G2_func,
                                          max_diameter=plate_12_12.max_diameter[0],
                                          ndim=3,
                                          n_vertex=4,
                                          num_slices=4,
                                          num_slices_diag=3)
print("G2")
compute_coeffs.coeffs_save(coeffs_G2, "d:\\Python\\Py_Nonstationary_EM_integralEq\\coeffs\\plate_12_12\\G2_plate_12_12.txt")

