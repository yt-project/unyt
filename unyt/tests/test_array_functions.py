# tests for NEP 18
import numpy as np

from unyt import cm, s, g


def test_array_repr():
    arr = [1, 2, 3] * cm
    assert np.array_repr(arr) == "unyt_array([1, 2, 3] cm)"


def test_linalg_inv():
    arr = np.random.random_sample((3, 3)) * cm
    iarr = np.linalg.inv(arr)
    assert 1 * iarr.units == 1 / cm


def test_histogram():
    arr = np.random.normal(size=1000) * cm
    counts, bins = np.histogram(arr, bins=10, range=(arr.min(), arr.max()))
    assert type(counts) is np.ndarray
    assert bins.units == arr.units


def test_histogram2d():
    x = np.random.normal(size=100) * cm
    y = np.random.normal(loc=10, size=100) * s
    counts, xbins, ybins = np.histogram2d(x, y)
    assert counts.ndim == 2
    assert 1 * xbins.units == 1 * x.units
    assert 1 * ybins.units == 1 * y.units


def test_histogramdd():
    x = np.random.normal(size=100) * cm
    y = np.random.normal(size=100) * s
    z = np.random.normal(size=100) * g
    counts, (xbins, ybins, zbins) = np.histogramdd((x, y, z))
    assert counts.ndim == 3
    assert 1 * xbins.units == 1 * x.units
    assert 1 * ybins.units == 1 * y.units
    assert 1 * zbins.units == 1 * z.units
