import src.integral as integral
import src.G_functions as G_functions
import numpy as np

def test_integrate_func_G1():
    collocation_test = np.array([0., 0., 0.])
    frame_test = np.array([[1.5, 0.5, -1],
                           [2.5, 0.5, -1],
                           [2.5, 1.5, -1],
                           [1.5, 1.5, -1]])
    n_vertex_test = 4
    num_slices_test = 4
    integral_function_test = G_functions.G1_func
    integration_function_test = integral.integrate_func
    target_test = 1 / (3 * 1e8) * np.array([-2 / np.sqrt(6), -1 / np.sqrt(6), 1 / np.sqrt(6)])
    rb = np.sqrt(2) / num_slices_test
    znak_tochnosti = 9
    assert np.prod(np.round(integration_function_test(collocation_test, frame_test,
                                n_vertex_test, num_slices_test,
                                integral_function_test, rb), znak_tochnosti) == np.round(target_test, znak_tochnosti))



def test_integrate_func_G2():
    collocation_test = np.array([0., 0., 0.])
    frame_test = np.array([[1.5, 0.5, -1],
                           [2.5, 0.5, -1],
                           [2.5, 1.5, -1],
                           [1.5, 1.5, -1]])
    n_vertex_test = 4
    num_slices_test = 16
    integral_function_test = G_functions.G2_func
    integration_function_test = integral.integrate_func
    target_test = np.array([-2. / (np.sqrt(6.**3)), -1. / (np.sqrt(6.**3)), 1. / (np.sqrt(6.**3))])
    rb = np.sqrt(2) / num_slices_test
    znak_tochnosti = 2
    assert np.prod(np.round(integration_function_test(collocation_test, frame_test,
                                n_vertex_test, num_slices_test,
                                integral_function_test, rb), znak_tochnosti) == np.round(target_test, znak_tochnosti))


def test_integrate_func_G3():
    collocation_test = np.array([2., 0., 0.])
    frame_test = np.array([[1.5, 0.5, 0],
                           [2.5, 0.5, 0],
                           [2.5, 1.5, 0],
                           [1.5, 1.5, 0]])
    n_vertex_test = 4
    num_slices_test = 8
    integral_function_test = G_functions.G3_func
    integration_function_test = integral.integrate_func
    target_test = 1/(3*1e8 ** 2)
    rb = np.sqrt(2) / num_slices_test
    znak_tochnosti = 16
    assert np.round(integration_function_test(collocation_test, frame_test,
                                n_vertex_test, num_slices_test,
                                integral_function_test, rb), znak_tochnosti) == np.round(target_test, znak_tochnosti)



def test2_integrate_func_G2():
    collocation_test = np.array([0., 0., 0.])
    frame_test = np.array([[24, 13, 0],
                           [25, 13, 0],
                           [25, 14, 0],
                           [24, 14, 0]])
    n_vertex_test = 4
    num_slices_test = 1
    integral_function_test = G_functions.G2_func
    integration_function_test = integral.integrate_func
    target_test = np.array([-24.5 / (np.sqrt(24.5 ** 2 + 13.5 ** 2) ** 3), -13.5 / (np.sqrt(24.5 ** 2 + 13.5 ** 2)**3), 0.])
    rb = np.sqrt(2) / num_slices_test
    znak_tochnosti = 15
    assert np.prod(np.round(integration_function_test(collocation_test, frame_test,
                                n_vertex_test, num_slices_test,
                                integral_function_test, rb), znak_tochnosti) == np.round(target_test, znak_tochnosti))



# Аналитический тест
def test3_integrate_func_G1():
    collocation_test = np.array([5., 5., 0.])
    frame_test = np.array([[2, 2, 0],
                           [4, 2, 0],
                           [4, 4, 0],
                           [2, 4, 0]])
    n_vertex_test = 4
    num_slices_test = 10
    integral_function_test = G_functions.G1_func
    integration_function_test = integral.integrate_func
    const = 2.7706315209427352994804450053544987990045633876999694264686621245
    target_test = np.array([const, const, 0.])
    rb = np.sqrt(8) / num_slices_test
    znak_tochnosti = 4
    assert np.prod(np.round((3 * 1e8) * integration_function_test(collocation_test, frame_test,
                                n_vertex_test, num_slices_test,
                                integral_function_test, rb), znak_tochnosti) == np.round(target_test, znak_tochnosti))
