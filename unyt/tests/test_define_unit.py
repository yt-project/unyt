from unyt.unit_object import define_unit
from unyt.array import unyt_quantity


def test_define_unit():
    define_unit("mph", (1.0, "mile/hr"))
    a = unyt_quantity(2.0, "mph")
    b = unyt_quantity(1.0, "mile")
    c = unyt_quantity(1.0, "hr")
    assert a == 2.0*b/c
    d = unyt_quantity(1000.0, "cm**3")
    define_unit("L", d, prefixable=True)
    e = unyt_quantity(1.0, "mL")
    f = unyt_quantity(1.0, "cm**3")
    assert e == f
