"""
Test unyt.testing module that contains utilities for writing tests.

"""
import pytest
from numpy.testing import assert_allclose

from unyt.array import unyt_array, unyt_quantity
from unyt.testing import assert_allclose_units


def test_equality():
    a1 = unyt_array([1.0, 2.0, 3.0], "cm")
    a2 = unyt_array([1.0, 2.0, 3.0], "cm")
    assert_allclose(a1, a2)


def test_unequal_error():
    a1 = unyt_array([1.0, 2.0, 3.0], "cm")
    a2 = unyt_array([4.0, 5.0, 6.0], "cm")
    with pytest.raises(AssertionError):
        assert_allclose(a1, a2)


@pytest.mark.parametrize("second_unit", ["km", "kg"])
def test_conversion_error(second_unit):
    a1 = unyt_array([1.0, 2.0, 3.0], "cm")
    a2 = unyt_array([1.0, 2.0, 3.0], second_unit)
    with pytest.raises(AssertionError):
        assert_allclose(a1, a2)


def test_runtime_error():
    a1 = unyt_array([1.0, 2.0, 3.0], "cm")
    a2 = unyt_array([1.0, 2.0, 3.0], "cm")
    with pytest.raises(RuntimeError):
        assert_allclose(a1, a2, rtol=unyt_quantity(1e-7, "cm"))


def test_atol_conversion_error():
    a1 = unyt_array([1.0, 2.0, 3.0], "cm")
    a2 = unyt_array([1.0, 2.0, 3.0], "cm")
    with pytest.raises(AssertionError):
        assert_allclose(a1, a2, atol=unyt_quantity(0.0, "kg"))


def test_depr_assert_allclose_units():
    a1 = unyt_array([1.0, 2.0, 3.0], "cm")
    a2 = unyt_array([1.0, 2.0, 3.0], "cm")
    with pytest.warns(DeprecationWarning):
        assert_allclose_units(a1, a2)
