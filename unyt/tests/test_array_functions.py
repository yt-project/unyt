# tests for NEP 18
import numpy as np

from unyt import cm


def test_array_repr():
    arr = [1, 2, 3] * cm
    assert np.array_repr(arr) == "unyt_array([1, 2, 3] cm)"


def test_linalg_inv():
    arr = np.random.random_sample((3, 3)) * cm
    iarr = np.linalg.inv(arr)
    assert 1 * iarr.units == 1 / cm
