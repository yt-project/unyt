# tests for NEP 18
from numpy import array_repr

from unyt import cm


def test_array_repr():
    arr = [1, 2, 3] * cm
    assert array_repr(arr) == "unyt_array([1, 2, 3] cm)"
