import src.problemSolving as ps
import numpy as np


def test_f_function():
    assert (ps.f_function((1 / 22, 1)) == 0)
    assert (ps.f_function((1 / 21, 1)) == 0)
    assert (ps.f_function((1 / 20, 1)) != 0)
    assert (ps.f_function((0, 1)) == 0)
    assert (ps.f_function((1, 1)) == 0)


def test_linear_prop():
    assert sum(ps.proportion_linear(1.9, 1.7, 0.3)) == 1
    assert sum(ps.proportion_linear(1.9, 1.6, 0.35)) == 1
    assert sum(ps.proportion_linear(0.1, 0, 0.2)) == 1


def test_time_index_between():
    test_object = ps.ProblemSolving(0.5, 4, 16)
    assert test_object.time_index_between(2.9)[0] == np.array(5)
    assert test_object.time_index_between(2.9)[1] == np.array(6)
    assert test_object.time_index_between(2.9999999)[0] == np.array(5)
    assert test_object.time_index_between(2.9999999)[1] == np.array(6)
    assert test_object.time_index_between(-0.9)[0] == np.array(-2)
    assert test_object.time_index_between(-0.9)[1] == np.array(-1)
    assert test_object.time_index_between(0)[1] == np.array(0)


def test_time_index():
    test_object = ps.ProblemSolving(0.5, 4, 16)
    assert test_object.time_index(3) == 6
    assert test_object.time_index(-1) == -2
    assert test_object.time_index(0) == 0


def test_get_Q_element():
    test_object = ps.ProblemSolving(0.5, 4, 16)
    assert test_object.get_Q_element(-1.4, 5) == 0.0
    assert test_object.get_Q_element(16, 10) == 0.0
    assert test_object.get_Q_element(0.5, 4) == 0