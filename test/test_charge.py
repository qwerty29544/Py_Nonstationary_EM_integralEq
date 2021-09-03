import src.charge as charge

def test_f_function():
    assert (charge.f_function(1 / 22, 1) == 0)
    assert (charge.f_function(1 / 21, 1) == 0)
    assert (charge.f_function(1 / 20, 1) != 0)
    assert (charge.f_function(0, 1) == 0)
    assert (charge.f_function(1, 1) == 0)
