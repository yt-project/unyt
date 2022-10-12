# tests for NEP 18
import re

import numpy as np
import pytest

from unyt import cm, g, km, s
from unyt.array import unyt_array
from unyt.exceptions import UnitInconsistencyError


def test_array_repr():
    arr = [1, 2, 3] * cm
    assert np.array_repr(arr) == "unyt_array([1, 2, 3], units='cm')"


def test_dot_vectors():
    a = [1, 2, 3] * cm
    b = [1, 2, 3] * s
    res = np.dot(a, b)
    assert res.units == cm * s
    assert res.d == 14


# NOTE: explicitly setting the dtype of out arrays as `dtype=np.int_` is the
# cross-platform way to guarantee that their dtype matches that of np.arange(x)
# (on POSIX system it's int64, while on windows it's int32)
@pytest.mark.parametrize(
    "out",
    [
        None,
        np.empty((3, 3), dtype=np.int_),
        np.empty((3, 3), dtype=np.int_, order="C") * cm * s,
        np.empty((3, 3), dtype=np.int_, order="C") * km * s,
    ],
    ids=[
        "None",
        "pure ndarray",
        "same units",
        "convertible units",
    ],
)
def test_dot_matrices(out):
    a = np.arange(9) * cm
    a.shape = (3, 3)
    b = np.arange(9) * s
    b.shape = (3, 3)

    res = np.dot(a, b, out=out)

    if out is not None:
        np.testing.assert_array_equal(res, out)
        assert res is out

    if isinstance(out, unyt_array):
        # check that the result can be converted to predictible units
        res.in_units("cm * s")
        assert out.units == res.units


def test_invalid_dot_matrices():
    a = np.arange(9) * cm
    a.shape = (3, 3)
    b = np.arange(9) * s
    b.shape = (3, 3)

    out = np.empty((3, 3), dtype=np.int_, order="C") * s**2
    with pytest.raises(
        TypeError,
        match=re.escape(
            "output array is not acceptable "
            "(units 's**2' cannot be converted to 'cm*s')"
        ),
    ):
        np.dot(a, b, out=out)


def test_vdot():
    a = np.arange(9) * cm
    b = np.arange(9) * s
    res = np.vdot(a, b)
    assert res.units == cm * s


def test_inner():
    a = np.array([1, 2, 3]) * cm
    b = np.array([0, 1, 0]) * s
    res = np.inner(a, b)
    assert res.d == 2
    assert res.units == cm * s


def test_outer():
    a = np.array([1, 2, 3]) * cm
    b = np.array([0, 1, 0]) * s
    res = np.outer(a, b)
    expected = np.array([[0, 1, 0], [0, 2, 0], [0, 3, 0]])
    np.testing.assert_array_equal(res.ndview, expected)
    assert res.units == cm * s


def test_kron():
    a = np.eye(2) * cm
    b = np.ones((2, 2)) * s
    res = np.kron(a, b)
    assert res.units == cm * s


def test_matmul():
    a = np.array([[1, 0], [0, 1]]) * cm
    b = np.array([[4, 1], [2, 2]]) * s
    expected = np.array([[4, 1], [2, 2]]) * cm * s
    out = np.empty_like(expected)
    res = np.matmul(a, b, out=out)
    np.testing.assert_array_equal(res, expected)
    assert res.units == expected.units

    with pytest.xfail(
        reason=(
            "At of numpy 1.21.2, np.matmul seem to "
            "escape the __unyt_array__ protocol"
        )
    ):
        assert res is out


def test_linalg_inv():
    arr = np.random.random_sample((3, 3)) * cm
    iarr = np.linalg.inv(arr)
    assert 1 * iarr.units == 1 / cm


def test_linalg_tensorinv():
    a = np.eye(4 * 6) * cm
    a.shape = (4, 6, 8, 3)
    ia = np.linalg.tensorinv(a)
    assert 1 * ia.units == 1 / cm


def test_linalg_pinv():
    a = np.random.randn(9, 6) * cm
    B = np.linalg.pinv(a)
    assert 1 * B.units == 1 / cm
    np.testing.assert_allclose(a, np.dot(a, np.dot(B, a)))
    np.testing.assert_allclose(B, np.dot(B, np.dot(a, B)))


@pytest.mark.xfail(
    reason=(
        "as of numpy 1.21.2, the __array_function__ protocol doesn't let "
        "us overload np.linalg.pinv when the first argument isn't a unyt_array"
    )
)
def test_matrix_stack_linalg_pinv():
    stack = [np.eye(4) * g for _ in range(3)]
    B = np.linalg.pinv(stack)
    assert 1 * B.units == 1 / g


@pytest.mark.xfail(
    reason=(
        "as of numpy 1.21.2, the __array_function__ protocol doesn't let "
        "us overload np.linalg.pinv when the first argument isn't a unyt_array"
    )
)
def test_invalid_matrix_stack_linalg_pinv():
    stack = [np.eye(4) * g, np.eye(4) * s]
    with pytest.raises(
        TypeError,
        match=re.escape(
            "numpy.linalg.pinv cannot operate on a stack "
            "of matrices with different units."
        ),
    ):
        np.linalg.pinv(stack)


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


def test_concatenate():
    x1 = np.random.normal(size=100) * cm
    x2 = np.random.normal(size=100) * cm
    res = np.concatenate((x1, x2))
    assert 1 * res.units == 1 * cm
    assert res.shape == (200,)


def test_concatenate_different_units():
    x1 = np.random.normal(size=100) * cm
    x2 = np.random.normal(size=100) * s
    with pytest.raises(
        UnitInconsistencyError,
        match=(
            r"Expected all unyt_array arguments to have identical units\. "
            r"Received mixed units \(cm, s\)"
        ),
    ):
        np.concatenate((x1, x2))


def test_cross():
    x1 = np.random.random_sample((2, 2)) * cm
    x2 = np.eye(2) * s
    res = np.cross(x1, x2)
    assert res.units == cm * s


def test_intersect1d():
    x1 = [1, 2, 3, 4, 5, 6, 7, 8] * cm
    x2 = [0, 2, 4, 6, 8] * cm
    res = np.intersect1d(x1, x2)
    assert 1 * res.units == 1 * cm
    np.testing.assert_array_equal(res, [2, 4, 6, 8])


def test_intersect1d_return_indices():
    x1 = [1, 2, 3, 4, 5, 6, 7, 8] * cm
    x2 = [0, 2, 4, 6, 8] * cm
    ures = np.intersect1d(x1, x2, return_indices=True)
    rres = np.intersect1d(x1.d, x2.d, return_indices=True)
    np.testing.assert_array_equal(ures, rres)


def test_union1d():
    x1 = [-1, 0, 1] * cm
    x2 = [-2, -1, -3] * cm
    res = np.union1d(x1, x2)
    assert 1 * res.units == 1 * cm
    np.testing.assert_array_equal(res, [-3, -2, -1, 0, 1])


def test_linalg_norm():
    x = [1, 1, 1] * s
    res = np.linalg.norm(x)
    assert 1 * res.units == 1 * s
    assert res == pytest.approx(np.sqrt(3))


def test_vstack():
    x1 = [0, 1, 2] * cm
    x2 = [3, 4, 5] * cm
    res = np.vstack((x1, x2))
    assert 1 * res.units == 1 * cm
    np.testing.assert_array_equal(res, [[0, 1, 2], [3, 4, 5]])


def test_hstack():
    x1 = np.array([[0, 1, 2], [3, 4, 5]]) * cm
    x2 = np.array([[6, 7, 8], [9, 10, 11]]) * cm
    res = np.hstack((x1, x2))
    assert 1 * res.units == 1 * cm
    np.testing.assert_array_equal(res, [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])


@pytest.mark.parametrize(
    "axis, expected", [(0, [[0, 1, 2], [3, 4, 5]]), (1, [[0, 3], [1, 4], [2, 5]])]
)
def test_stack(axis, expected):
    x1 = [0, 1, 2] * cm
    x2 = [3, 4, 5] * cm
    res = np.stack((x1, x2), axis=axis)
    assert 1 * res.units == 1 * cm
    np.testing.assert_array_equal(res, expected)
