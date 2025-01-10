import pytest

from unyt.array import unyt_quantity
from unyt.unit_object import define_unit
from unyt.unit_registry import UnitRegistry


def test_define_unit():
    define_unit("kph", (1.0, "km/hr"))
    a = unyt_quantity(2.0, "kph")
    b = unyt_quantity(1.0, "km")
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

    ## allow_override test
    define_unit("Foo", (1.0, "V/s"), allow_override=True)
    h = unyt_quantity(1.0, "Foo")
    assert h != g
    assert h == volt / second

    # Test custom registry
    reg = UnitRegistry()
    define_unit("Foo", (1, "m"), registry=reg)
    define_unit("Baz", (1, "Foo**2"), registry=reg)
    i = unyt_quantity(1, "Baz", registry=reg)
    j = unyt_quantity(1, "m**2", registry=reg)
    assert i == j
    print("done!")

def test_define_unit_error():
    from unyt import define_unit

    with pytest.raises(RuntimeError):
        define_unit("foobar", "baz")
    with pytest.raises(RuntimeError):
        define_unit("foobar", 12)
    with pytest.raises(RuntimeError):
        define_unit("C", (1.0, "A*s"))

if __name__ == "__main__":
    test_define_unit()
    test_define_unit_error()
