import compute_coeffs


def compute_wrap(frames,
                 collocations,
                 total_frames_in_object,
                 max_d,
                 n_vert,
                 num_slice,
                 num_slice_diag,
                 save_to_file,
                 ndim,
                 function_to_integrate,
                 integration_method):

    coefficients = compute_coeffs.compute_coeffs(frame=frames,
                                                 collocation=collocations,
                                                 number_of_frames=total_frames_in_object,
                                                 integration_method=integration_method,
                                                 integral_function=function_to_integrate,
                                                 max_diameter=max_d,
                                                 n_vertex=n_vert,
                                                 num_slices=num_slice,
                                                 num_slices_diag=num_slice_diag,
                                                 ndim=ndim)

    print("coefficients computes!")
    compute_coeffs.coeffs_save(coefficients, filename=save_to_file)
    print("saved to " + save_to_file)
    return coefficients
