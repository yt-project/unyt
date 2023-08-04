"""
Test unyt.testing module that contains utilities for writing tests.

"""
import pytest

from unyt import accepts, meter, returns, second
from unyt.array import unyt_array, unyt_quantity
from unyt.dimensions import length, time
from unyt.testing import assert_allclose_units


def test_equality():
    a1 = unyt_array([1.0, 2.0, 3.0], "cm")
    a2 = unyt_array([1.0, 2.0, 3.0], "cm")
    assert_allclose_units(a1, a2)


def test_unequal_error():
    a1 = unyt_array([1.0, 2.0, 3.0], "cm")
    a2 = unyt_array([4.0, 5.0, 6.0], "cm")
    with pytest.raises(AssertionError):
        assert_allclose_units(a1, a2)


def test_conversion_error():
    a1 = unyt_array([1.0, 2.0, 3.0], "cm")
    a2 = unyt_array([1.0, 2.0, 3.0], "kg")
    with pytest.raises(AssertionError):
        assert_allclose_units(a1, a2)


def test_runtime_error():
    a1 = unyt_array([1.0, 2.0, 3.0], "cm")
    a2 = unyt_array([1.0, 2.0, 3.0], "cm")
    with pytest.raises(RuntimeError):
        assert_allclose_units(a1, a2, rtol=unyt_quantity(1e-7, "cm"))


def test_atol_conversion_error():
    a1 = unyt_array([1.0, 2.0, 3.0], "cm")
    a2 = unyt_array([1.0, 2.0, 3.0], "cm")
    with pytest.raises(AssertionError):
        assert_allclose_units(a1, a2, atol=unyt_quantity(0.0, "kg"))


def test_accepts():
    @accepts(a=time, v=length / time)
    def foo(a, v):
        return a * v

    _ = foo(a=2 * second, v=3 * meter / second)

    with pytest.raises(TypeError):
        _ = foo(a=2 * meter, v=3 * meter / second)

    with pytest.raises(TypeError):
        _ = foo(a=2 * second, v=3 * meter)


def test_accepts_partial():
    @accepts(a=time)
    def bar(a, v):
        return a * v

    _ = bar(a=2 * second, v=3 * meter / second)
    _ = bar(a=2 * second, v=3 * meter)

    with pytest.raises(TypeError):
        _ = bar(a=2 * meter, v=3 * meter / second)

    @accepts(v=length / time)
    def baz(a, v):
        return a * v

    _ = baz(a=2 * second, v=3 * meter / second)
    _ = baz(a=2 * meter, v=3 * meter / second)

    with pytest.raises(TypeError):
        _ = baz(a=2 * second, v=3 * meter)


def test_returns():
    @returns(length)
    def foo(a, v):
        return a * v

    _ = foo(a=2 * second, v=3 * meter / second)

    with pytest.raises(TypeError):
        _ = foo(a=2 * meter, v=3 * meter / second)

    with pytest.raises(TypeError):
        _ = foo(a=2 * second, v=3 * meter)


def test_returns_multiple():
    @returns(time, length / time)
    def bar(a, v):
        return a, v

    _ = bar(a=2 * second, v=3 * meter / second)

    with pytest.raises(TypeError):
        _ = bar(a=2 * meter, v=3 * meter / second)

    with pytest.raises(TypeError):
        _ = bar(a=2 * second, v=3 * meter)
