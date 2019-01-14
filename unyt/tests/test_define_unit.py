import pytest

from unyt.unit_object import define_unit
from unyt.array import unyt_quantity


def test_define_unit():
    define_unit("mph", (1.0, "mile/hr"))
    a = unyt_quantity(2.0, "mph")
    b = unyt_quantity(1.0, "mile")
    c = unyt_quantity(1.0, "hr")
    assert a == 2.0 * b / c
    d = unyt_quantity(1000.0, "cm**3")
    define_unit("Baz", d, prefixable=True)
    e = unyt_quantity(1.0, "mBaz")
    f = unyt_quantity(1.0, "cm**3")
    assert e == f

    define_unit("Foo", (1.0, "V/sqrt(s)"))
    g = unyt_quantity(1.0, "Foo")
    volt = unyt_quantity(1.0, "V")
    second = unyt_quantity(1.0, "s")
    assert g == volt / second ** (0.5)


def test_define_unit_error():
    from unyt import define_unit

    with pytest.raises(RuntimeError):
        define_unit("foobar", "baz")
    with pytest.raises(RuntimeError):
        define_unit("foobar", 12)
    with pytest.raises(RuntimeError):
        define_unit("C", (1.0, "A*s"))
