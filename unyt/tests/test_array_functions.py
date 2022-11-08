# tests for NumPy __array_function__ support
import re

import numpy as np
import pytest

from unyt import cm, g, km, s
from unyt._array_functions import _HANDLED_FUNCTIONS as HANDLED_FUNCTIONS
from unyt.array import unyt_array, unyt_quantity
from unyt.exceptions import UnitInconsistencyError

# this is a subset of NOT_HANDLED_FUNCTIONS for which there's nothing to do
# because they don't apply to (real) numeric types
# or they work as expected out of the box
# This is not necessarilly complete !
NOOP_FUNCTIONS = {
    np.all,  # expects booleans
    np.alltrue,  # expects booleans
    np.amax,  # works out of the box (tested)
    np.amin,  # works out of the box (tested)
    np.angle,  # expects complex numbers
    np.any,  # works out of the box (tested)
    np.append,  # we get it for free with np.concatenate (tested)
    np.argmax,  # returns pure numbers
    np.argmin,  # returns pure numbers
    np.argpartition,  # return pure numbers
    np.argsort,  # returns pure numbers
    np.argwhere,  # returns pure numbers
    np.array_repr,  # hooks into __repr__
    np.array_str,  # hooks into __str__
    np.atleast_1d,  # works out of the box (tested)
    np.atleast_2d,  # works out of the box (tested)
    np.atleast_3d,  # works out of the box (tested)
    np.average,  # works out of the box (tested)
    np.can_cast,  # works out of the box (tested)
    np.iscomplex,  # works out of the box (tested)
    np.iscomplexobj,  # works out of the box (tested)
    np.isreal,  # works out of the box (tested)
    np.isrealobj,  # works out of the box (tested)
    np.nan_to_num,  # works out of the box (tested)
    np.nanargmax,  # return pure numbers
    np.nanargmin,  # return pure numbers
    np.nanmax,  # works out of the box (tested)
    np.nanmean,  # works out of the box (tested)
    np.nanmedian,  # works out of the box (tested)
    np.nanmin,  # works out of the box (tested)
    np.trim_zeros,  # works out of the box (tested)
    np.max,  # works out of the box (tested)
    np.mean,  # works out of the box (tested)
    np.median,  # works out of the box (tested)
    np.min,  # works out of the box (tested)
    np.ndim,  # return pure numbers
    np.shape,  # returns pure numbers
    np.size,  # returns pure numbers
    np.sort,  # works out of the box (tested)
    np.sum,  # works out of the box (tested)
    np.repeat,  # works out of the box (tested)
    np.tile,  # works out of the box (tested)
}

# this set represents all functions that need inspection, tests, or both
# it is always possible that some of its elements belong in NOOP_FUNCTIONS
TODO_FUNCTIONS = {
    np.allclose,
    np.apply_along_axis,
    np.apply_over_axes,
    np.array_equal,
    np.array_equiv,
    np.array_split,
    np.bincount,
    np.broadcast_arrays,
    np.broadcast_to,
    np.busday_count,
    np.busday_offset,
    np.choose,
    np.clip,
    np.column_stack,
    np.common_type,
    np.compress,
    np.convolve,
    np.copy,
    np.copyto,
    np.corrcoef,
    np.correlate,
    np.count_nonzero,
    np.cov,
    np.cumprod,
    np.cumproduct,
    np.cumsum,
    np.datetime_as_string,
    np.delete,
    np.diag,
    np.diag_indices_from,
    np.diagflat,
    np.diagonal,
    np.diff,
    np.digitize,
    np.dsplit,
    np.dstack,
    np.ediff1d,
    np.einsum,
    np.einsum_path,
    np.empty_like,
    np.expand_dims,
    np.extract,
    np.fill_diagonal,
    np.fix,
    np.flatnonzero,
    np.flip,
    np.fliplr,
    np.flipud,
    np.full_like,
    np.geomspace,
    np.gradient,
    np.histogram_bin_edges,
    np.hsplit,
    np.i0,
    np.imag,
    np.in1d,
    np.insert,
    np.interp,
    np.is_busday,
    np.isclose,
    np.isin,
    np.isneginf,
    np.isposinf,
    np.ix_,
    np.lexsort,
    np.linalg.cholesky,
    np.linalg.cond,
    np.linalg.det,
    np.linalg.eig,
    np.linalg.eigh,
    np.linalg.eigvals,
    np.linalg.eigvalsh,
    np.linalg.lstsq,
    np.linalg.matrix_power,
    np.linalg.matrix_rank,
    np.linalg.multi_dot,
    np.linalg.qr,
    np.linalg.slogdet,
    np.linalg.solve,
    np.linalg.svd,
    np.linalg.tensorsolve,
    np.linspace,
    np.logspace,
    np.may_share_memory,
    np.meshgrid,
    np.min_scalar_type,
    np.moveaxis,
    np.msort,
    np.nancumprod,
    np.nancumsum,
    np.nanpercentile,
    np.nanprod,
    np.nanquantile,
    np.nanstd,
    np.nansum,
    np.nanvar,
    np.nonzero,
    np.ones_like,
    np.packbits,
    np.pad,
    np.partition,
    np.percentile,
    np.piecewise,
    np.place,
    np.poly,
    np.polyadd,
    np.polyder,
    np.polydiv,
    np.polyfit,
    np.polyint,
    np.polymul,
    np.polysub,
    np.polyval,
    np.prod,
    np.product,
    np.ptp,  # note: should return delta_K for temperatures !
    np.put,
    np.put_along_axis,
    np.putmask,
    np.quantile,
    np.ravel,
    np.ravel_multi_index,
    np.real,
    np.real_if_close,
    np.reshape,
    np.resize,
    np.result_type,
    np.roll,
    np.rollaxis,
    np.roots,
    np.rot90,
    np.round,
    np.round_,
    np.save,
    np.savetxt,
    np.savez,
    np.savez_compressed,
    np.searchsorted,
    np.select,
    np.setdiff1d,
    np.setxor1d,
    np.shares_memory,
    np.sinc,
    np.sometrue,
    np.split,
    np.squeeze,
    np.std,
    np.swapaxes,
    np.take,
    np.take_along_axis,
    np.tensordot,
    np.trace,
    np.transpose,
    np.tril,
    np.tril_indices_from,
    np.triu,
    np.triu_indices_from,
    np.unique,
    np.unpackbits,
    np.unravel_index,
    np.unwrap,
    np.vander,
    np.var,
    np.vsplit,
    np.where,
    np.zeros_like,
}

removed_functions = {
    "alen",  # deprecated in numpy 1.18, removed in 1.22
    "asscalar",  # deprecated in numpy 1.18, removed in 1.22
    "fv",  # deprecated in numpy 1.18, removed in 1.20
    "ipmt",  # deprecated in numpy 1.18, removed in 1.20
    "irr",  # deprecated in numpy 1.18, removed in 1.20
    "mirr",  # deprecated in numpy 1.18, removed in 1.20
    "nper",  # deprecated in numpy 1.18, removed in 1.20
    "npv",  # deprecated in numpy 1.18, removed in 1.20
    "pmt",  # deprecated in numpy 1.18, removed in 1.20
    "ppmt",  # deprecated in numpy 1.18, removed in 1.20
    "pv",  # deprecated in numpy 1.18, removed in 1.20
    "rank",  # deprecated in numpy 1.10, removed in 1.18
    "rate",  # deprecated in numpy 1.18, removed in 1.20
}

NOT_HANDLED_FUNCTIONS = NOOP_FUNCTIONS | TODO_FUNCTIONS

for func in removed_functions:
    if hasattr(np, func):
        NOT_HANDLED_FUNCTIONS.add(getattr(np, func))


def get_wrapped_functions(*modules):
    """get functions that support __array_function__ in modules

    This was adapted from astropy's tests
    """
    wrapped_functions = {}
    for mod in modules:
        for name, f in mod.__dict__.items():
            if callable(f) and hasattr(f, "__wrapped__"):
                if f is np.printoptions or f.__name__.startswith("_"):
                    continue
                wrapped_functions[mod.__name__ + "." + name] = f
    return dict(sorted(wrapped_functions.items()))


def test_wrapping_completeness():
    """Ensure we wrap all numpy functions that support __array_function__"""
    handled_numpy_functions = set(HANDLED_FUNCTIONS.keys())
    # ensure no functions appear in both NOT_HANDLED_FUNCTIONS and HANDLED_FUNCTIONS
    assert NOT_HANDLED_FUNCTIONS.isdisjoint(handled_numpy_functions)
    # get list of functions that support wrapping by introspection on numpy module
    wrappable_functions = get_wrapped_functions(np, np.fft, np.linalg)
    for function in HANDLED_FUNCTIONS:
        # ensure we only have wrappers for functions that support wrapping
        assert function in wrappable_functions.values()
    all_funcs = NOT_HANDLED_FUNCTIONS.union(handled_numpy_functions)
    # ensure all functions in numpy that support wrapping either have wrappers
    # or are explicitly whitelisted
    for function in wrappable_functions.values():
        assert function in all_funcs


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
        assert np.shares_memory(res, out)
        assert isinstance(res, unyt_array)
        assert isinstance(out, np.ndarray)

    if isinstance(out, unyt_array):
        # check that the result can be converted to predictible units
        res.in_units("cm * s")
        assert out.units == res.units


def test_dot_mixed_ndarray_unyt_array():
    a = np.ones((3, 3))
    b = np.ones((3, 3)) * cm

    res = np.dot(a, b)

    assert isinstance(res, unyt_array)
    assert res.units == cm

    out = np.zeros((3, 3))
    res = np.dot(a, b, out=out)

    assert isinstance(res, unyt_array)
    assert type(out) is np.ndarray
    assert np.shares_memory(out, res)
    np.testing.assert_array_equal(out, res)

    out = np.zeros((3, 3)) * km
    res = np.dot(a, b, out=out)

    assert isinstance(res, unyt_array)
    assert isinstance(out, unyt_array)
    assert res.units == out.units == cm
    assert np.shares_memory(res, out)

    # check this works with an ndarray as the first operand
    out = np.zeros((3, 3)) * km
    res = np.dot(b, a, out=out)

    assert isinstance(res, unyt_array)
    assert isinstance(out, unyt_array)
    assert res.units == out.units == cm
    assert np.shares_memory(res, out)


def test_invalid_dot_matrices():
    a = np.arange(9) * cm
    a.shape = (3, 3)
    b = np.arange(9) * s
    b.shape = (3, 3)

    out = np.empty((3, 3), dtype=np.int_, order="C") * s**2
    res = np.dot(a, b, out=out)

    np.testing.assert_array_equal(res, out)
    assert out.units == res.units == cm * s


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


# see https://github.com/numpy/numpy/issues/22444
@pytest.mark.xfail(
    reason=(
        "as of numpy 1.21.2, the __array_function__ protocol doesn't let "
        "us overload np.linalg.pinv for stacks of matrices"
    )
)
def test_matrix_stack_linalg_pinv():
    stack = [np.eye(4) * g for _ in range(3)]
    B = np.linalg.pinv(stack)
    assert 1 * B.units == 1 / g


# see https://github.com/numpy/numpy/issues/22444
@pytest.mark.xfail(
    reason=(
        "as of numpy 1.21.2, the __array_function__ protocol doesn't let "
        "us overload np.linalg.pinv for stacks of matrices"
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
    assert xbins.units == x.units
    assert ybins.units == y.units


def test_histogramdd():
    x = np.random.normal(size=100) * cm
    y = np.random.normal(size=100) * s
    z = np.random.normal(size=100) * g
    counts, (xbins, ybins, zbins) = np.histogramdd((x, y, z))
    assert counts.ndim == 3
    assert xbins.units == x.units
    assert ybins.units == y.units
    assert zbins.units == z.units


def test_concatenate():
    x1 = np.random.normal(size=100) * cm
    x2 = np.random.normal(size=100) * cm
    res = np.concatenate((x1, x2))
    assert res.units == cm
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
    assert res.units == cm
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
    assert res.units == cm
    np.testing.assert_array_equal(res, [-3, -2, -1, 0, 1])


def test_linalg_norm():
    x = [1, 1, 1] * s
    res = np.linalg.norm(x)
    assert res.units == s
    assert res == pytest.approx(np.sqrt(3))


def test_vstack():
    x1 = [0, 1, 2] * cm
    x2 = [3, 4, 5] * cm
    res = np.vstack((x1, x2))
    assert res.units == cm
    np.testing.assert_array_equal(res, [[0, 1, 2], [3, 4, 5]])


def test_hstack():
    x1 = np.array([[0, 1, 2], [3, 4, 5]]) * cm
    x2 = np.array([[6, 7, 8], [9, 10, 11]]) * cm
    res = np.hstack((x1, x2))
    assert res.units == cm
    np.testing.assert_array_equal(res, [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])


@pytest.mark.parametrize(
    "axis, expected", [(0, [[0, 1, 2], [3, 4, 5]]), (1, [[0, 3], [1, 4], [2, 5]])]
)
def test_stack(axis, expected):
    x1 = [0, 1, 2] * cm
    x2 = [3, 4, 5] * cm
    res = np.stack((x1, x2), axis=axis)
    assert res.units == cm
    np.testing.assert_array_equal(res, expected)


def test_amax():
    x1 = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]] * cm
    res = np.amax(x1)
    assert type(res) is unyt_quantity
    res = np.amax(x1, axis=1)
    assert type(res) is unyt_array


def test_amin():
    x1 = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]] * cm
    res = np.amin(x1)
    assert type(res) is unyt_quantity
    res = np.amin(x1, axis=1)
    assert type(res) is unyt_array


def test_around():
    x1 = [[1, 2, 3], [1, 2, 3], [1, 2, 3.0]] * g
    res = np.around(x1, 2)
    assert type(res) is unyt_array
    assert res.units == g


def test_atleast_nd():
    x0 = 1.0 * cm

    x1 = np.atleast_1d(x0)
    assert type(x1) is unyt_array
    assert x1.ndim == 1
    assert x1.units == cm

    x2 = np.atleast_2d(x0)
    assert type(x2) is unyt_array
    assert x2.ndim == 2
    assert x2.units == cm

    x3 = np.atleast_3d(x0)
    assert type(x3) is unyt_array
    assert x3.ndim == 3
    assert x3.units == cm


def test_average():
    x1 = [0.0, 1.0, 2.0] * cm
    res = np.average(x1)
    assert type(res) is unyt_quantity
    assert res == 1 * cm


def test_trim_zeros():
    x1 = [0, 1, 2, 3, 0] * cm
    res = np.trim_zeros(x1)
    assert type(res) is unyt_array


def test_any():
    assert not np.any([0, 0, 0] * cm)
    assert np.any([1, 0, 0] * cm)


def test_append():
    a = [0, 1, 2, 3] * cm
    b = np.append(a, [4, 5, 6] * cm)
    assert type(b) is unyt_array
    assert b.units == cm


def test_append_inconsistent_units():
    a = [0, 1, 2, 3] * cm
    with pytest.raises(
        UnitInconsistencyError,
        match=re.escape(
            r"Expected all unyt_array arguments to have identical units. "
            r"Received mixed units (cm, dimensionless)"
        ),
    ):
        np.append(a, [4, 5, 6])


def test_asfarray():
    x1 = np.eye(3, dtype="int64") * cm

    x2 = np.asfarray(x1)
    assert type(x2) is unyt_array
    assert x2.units == cm
    assert x2.dtype == "float64"


def test_block():
    x1 = 1 * np.ones((3, 3)) * cm
    x2 = 2 * np.ones((3, 1)) * cm
    res = np.block([[x1, x2]])
    assert type(res) is unyt_array
    assert res.units == cm


def test_block_units_inconsistency():
    # check that unit inconsistency is correctly detected
    # for nested lists
    x1 = 1 * np.ones((3, 3)) * cm
    x2 = [3 * cm, 3 * cm, 3 * km]
    with pytest.raises(UnitInconsistencyError):
        np.block([[x1, x2]])


def test_can_cast():
    a = [0, 1, 2] * cm
    assert np.can_cast(a, "float64")
    assert np.can_cast(a, "int64")
    assert not np.can_cast(a, "float16")


def test_isreal_like():
    a = [1, 2, 3] * cm
    assert np.all(np.isreal(a))
    assert np.isrealobj(a)
    assert not np.any(np.iscomplex(a))
    assert not np.iscomplexobj(a)

    b = [1j, 2j, 3j] * cm
    assert not np.any(np.isreal(b))
    assert not np.isrealobj(b)
    assert np.all(np.iscomplex(b))
    assert np.iscomplexobj(b)


@pytest.mark.parametrize(
    "func",
    [
        np.fft.fft,
        np.fft.hfft,
        np.fft.rfft,
        np.fft.ifft,
        np.fft.ihfft,
        np.fft.irfft,
    ],
)
def test_fft_1D(func):
    x1 = [0, 1, 2] * cm
    res = func(x1)
    assert type(res) is unyt_array
    assert res.units == (1 / cm).units


@pytest.mark.parametrize(
    "func",
    [
        np.fft.fft2,
        np.fft.fftn,
        np.fft.rfft2,
        np.fft.rfftn,
        np.fft.ifft2,
        np.fft.ifftn,
        np.fft.irfft2,
        np.fft.irfftn,
    ],
)
def test_fft_ND(func):
    x1 = [[0, 1, 2], [0, 1, 2], [0, 1, 2]] * cm
    res = func(x1)
    assert type(res) is unyt_array
    assert res.units == (1 / cm).units


@pytest.mark.parametrize("func", [np.fft.fftshift, np.fft.ifftshift])
def test_fft_shift(func):
    x1 = [[0, 1, 2], [0, 1, 2], [0, 1, 2]] * cm
    res = func(x1)
    assert type(res) is unyt_array
    assert res.units == cm


def test_trapz_no_x():
    y = [0, 1, 2, 3] * cm
    res = np.trapz(y)
    assert type(res) is unyt_quantity
    assert res.units == cm


def test_trapz_with_raw_x():
    y = [0, 1, 2, 3] * cm
    x = [0, 1, 2, 3]
    res = np.trapz(y, x)
    assert type(res) is unyt_quantity
    assert res.units == cm


def test_trapz_with_unit_x():
    y = [0, 1, 2, 3] * cm
    x = [0, 1, 2, 3] * s
    res = np.trapz(y, x)
    assert type(res) is unyt_quantity
    assert res.units == cm * s


def test_trapz_with_raw_dx():
    y = [0, 1, 2, 3] * cm
    dx = 2.0
    res = np.trapz(y, dx=dx)
    assert type(res) is unyt_quantity
    assert res.units == cm


def test_trapz_with_unit_dx():
    y = [0, 1, 2, 3] * cm
    dx = 2.0 * s
    res = np.trapz(y, dx=dx)
    assert type(res) is unyt_quantity
    assert res.units == cm * s


@pytest.mark.parametrize(
    "op",
    ["min", "max", "mean", "median", "sum", "nanmin", "nanmax", "nanmean", "nanmedian"],
)
def test_scalar_reduction(op):
    x = [0, 1, 2] * cm
    res = getattr(np, op)(x)
    assert type(res) is unyt_quantity
    assert res.units == cm


@pytest.mark.parametrize("op", ["sort", "sort_complex"])
def test_sort(op):
    x = [2, 0, 1] * cm
    res = getattr(np, op)(x)
    assert type(res) is unyt_array
    assert res.units == cm


def test_repeat():
    x = [2, 0, 1] * cm
    res = np.repeat(x, 2)
    assert type(res) is unyt_array
    assert res.units == cm


def test_tile():
    x = [2, 0, 1] * cm
    res = np.tile(x, (2, 3))
    assert type(res) is unyt_array
    assert res.units == cm
