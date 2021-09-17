import src.charge as charge
import numpy as np
import os


def test_f_function():
    assert (charge.f_function(1 / 22, 1) == 0)
    assert (charge.f_function(1 / 21, 1) == 0)
    assert (charge.f_function(1 / 20, 1) != 0)
    assert (charge.f_function(0, 1) == 0)
    assert (charge.f_function(1, 1) == 0)


def test_index():
    array = np.arange(9)
    assert array[-1] == 8
    assert array[0] == 0

test_index()