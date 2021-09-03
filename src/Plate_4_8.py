from compute_wrap import compute_wrap
from integral import integrate_func
from G_functions import G1_func, G2_func, G3_func
from figure import Figure


plate_12_12 = Figure("d:\\Python\\MaxwellIntegralEq\\figures\\plate_12_12.dat")
compute_wrap(frames=plate_12_12.frames[0],
             collocations=plate_12_12.collocations[0],
             total_frames_in_object=plate_12_12.total_frames_in_objects[0],
             max_d=plate_12_12.max_diameter[0],
             num_slice=10,
             num_slice_diag=50,
             ndim=3,
             n_vert=4,
             integration_method=integrate_func,
             function_to_integrate=G1_func,
             save_to_file="d:\\Python\\MaxwellIntegralEq\\coeffs\\plate_12_12\\G1_plate_12_12.txt")

compute_wrap(frames=plate_12_12.frames[0],
             collocations=plate_12_12.collocations[0],
             total_frames_in_object=plate_12_12.total_frames_in_objects[0],
             max_d=plate_12_12.max_diameter[0],
             num_slice=10,
             num_slice_diag=20,
             ndim=3,
             n_vert=4,
             integration_method=integrate_func,
             function_to_integrate=G2_func,
             save_to_file="d:\\Python\\MaxwellIntegralEq\\coeffs\\plate_12_12\\G2_plate_12_12.txt")

compute_wrap(frames=plate_12_12.frames[0],
             collocations=plate_12_12.collocations[0],
             total_frames_in_object=plate_12_12.total_frames_in_objects[0],
             max_d=plate_12_12.max_diameter[0],
             num_slice=10,
             num_slice_diag=70,
             ndim=1,
             n_vert=4,
             integration_method=integrate_func,
             function_to_integrate=G3_func,
             save_to_file="d:\\Python\\MaxwellIntegralEq\\coeffs\\plate_12_12\\G3_plate_12_12.txt")