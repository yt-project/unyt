"""
Test unyt.testing module that contains utilities for writing tests.

"""

# ----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import re

import pytest

from unyt.array import unyt_array, unyt_quantity
from unyt.testing import assert_allclose_units, assert_equal_units


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


def test_equal_missing_units():
    a = unyt_array([1, 2, 3], "K")
    b = unyt_array([1, 2, 3], "cm")
    with pytest.raises(
        AssertionError,
        match=re.escape("First argument is missing a `units` attribute."),
    ):
        assert_equal_units(a.d, b)
    with pytest.raises(
        AssertionError,
        match=re.escape("Second argument is missing a `units` attribute."),
    ):
        assert_equal_units(a, b.d)


def test_equal_incompatible_units():
    a = unyt_array([1, 2, 3], "K")
    b = unyt_array([1, 2, 3], "cm")
    with pytest.raises(
        AssertionError, match=re.escape("Incompatible units. Got 'K', expected 'cm'.")
    ):
        assert_equal_units(a, b)


def test_equal_wrong_shape():
    a = unyt_array([1, 2, 3], "cm")
    b = unyt_array([1, 2, 3, 4], "cm")
    with pytest.raises(AssertionError, match="^(\nArrays are not equal)"):
        # we don't check for the complete error message here because there's
        # a bug in numpy where units are not included in the error message
        # last checked with numpy 1.21.0
        """
        AssertionError:
        Arrays are not equal
        (shapes (3,), (4,) mismatch)
        x: unyt_array([1, 2, 3])
        y: unyt_array([1, 2, 3, 4])
        """
        assert_equal_units(a, b)


def test_equal_units_mismatch():
    a = unyt_array([1, 2, 3], "cm")
    b = unyt_array([1, 2, 3], "km")
    with pytest.raises(
        AssertionError, match=re.escape("Mismatching units. Got 'cm', expected 'km'.")
    ):
        assert_equal_units(a, b)
