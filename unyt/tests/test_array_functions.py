# tests for NumPy __array_function__ support
import re
from importlib.metadata import version

import numpy as np
import pytest
from numpy.testing import assert_allclose
from packaging.version import Version

from unyt import A, K, cm, degC, delta_degC, g, km, rad, s
from unyt._array_functions import (
    _HANDLED_FUNCTIONS as HANDLED_FUNCTIONS,
    _UNSUPPORTED_FUNCTIONS as UNSUPPORTED_FUNCTIONS,
)
from unyt.array import unyt_array, unyt_quantity
from unyt.exceptions import (
    UnitConversionError,
    UnitInconsistencyError,
    UnytError,
)
from unyt.testing import assert_array_equal_units
from unyt.unit_object import Unit
from unyt.unit_registry import UnitRegistry

NUMPY_VERSION = Version(version("numpy"))

# this is a subset of NOT_HANDLED_FUNCTIONS for which there's nothing to do
# because they don't apply to (real) numeric types
# or they work as expected out of the box
# This is not necessarilly complete !
NOOP_FUNCTIONS = {
    np.all,  # expects booleans
    np.amax,  # works out of the box (tested)
    np.amin,  # works out of the box (tested)
    np.angle,  # expects complex numbers
    np.any,  # works out of the box (tested)
    np.append,  # we get it for free with np.concatenate (tested)
    np.apply_along_axis,  # works out of the box (tested)
    np.argmax,  # returns pure numbers
    np.argmin,  # returns pure numbers
    np.argpartition,  # returns pure numbers
    np.argsort,  # returns pure numbers
    np.argwhere,  # returns pure numbers
    np.array_str,  # hooks into __str__
    np.atleast_1d,  # works out of the box (tested)
    np.atleast_2d,  # works out of the box (tested)
    np.atleast_3d,  # works out of the box (tested)
    np.average,  # works out of the box (tested)
    np.can_cast,  # works out of the box (tested)
    np.common_type,  # works out of the box (tested)
    np.result_type,  # works out of the box (tested)
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
    np.shares_memory,  # works out of the box (tested)
    np.nonzero,  # works out of the box (tested)
    np.count_nonzero,  # returns pure numbers
    np.flatnonzero,  # works out of the box (tested)
    np.isneginf,  # works out of the box (tested)
    np.isposinf,  # works out of the box (tested)
    np.empty_like,  # works out of the box (tested)
    np.full_like,  # works out of the box (tested)
    np.ones_like,  # works out of the box (tested)
    np.zeros_like,  # works out of the box (tested)
    np.copy,  # works out of the box (tested)
    np.meshgrid,  # works out of the box (tested)
    np.transpose,  # works out of the box (tested)
    np.reshape,  # works out of the box (tested)
    np.resize,  # works out of the box (tested)
    np.roll,  # works out of the box (tested)
    np.rollaxis,  # works out of the box (tested)
    np.rot90,  # works out of the box (tested)
    np.expand_dims,  # works out of the box (tested)
    np.squeeze,  # works out of the box (tested)
    np.flip,  # works out of the box (tested)
    np.fliplr,  # works out of the box (tested)
    np.flipud,  # works out of the box (tested)
    np.delete,  # works out of the box (tested)
    np.partition,  # works out of the box (tested)
    np.broadcast_to,  # works out of the box (tested)
    np.broadcast_arrays,  # works out of the box (tested)
    np.split,  # works out of the box (tested)
    np.array_split,  # works out of the box (tested)
    np.dsplit,  # works out of the box (tested)
    np.hsplit,  # works out of the box (tested)
    np.vsplit,  # works out of the box (tested)
    np.swapaxes,  # works out of the box (tested)
    np.moveaxis,  # works out of the box (tested)
    np.nansum,  # works out of the box (tested)
    np.std,  # works out of the box (tested)
    np.nanstd,  # works out of the box (tested)
    np.nanvar,  # works out of the box (tested)
    np.nanprod,  # works out of the box (tested)
    np.diag,  # works out of the box (tested)
    np.diag_indices_from,  # returns pure numbers
    np.diagflat,  # works out of the box (tested)
    np.diagonal,  # works out of the box (tested)
    np.ravel,  # returns pure numbers
    np.ravel_multi_index,  # returns pure numbers
    np.unravel_index,  # returns pure numbers
    np.fix,  # works out of the box (tested)
    np.round,  # is implemented via np.around
    np.may_share_memory,  # returns pure numbers (booleans)
    np.linalg.matrix_power,  # works out of the box (tested)
    np.linalg.cholesky,  # works out of the box (tested)
    np.linalg.multi_dot,  # works out of the box (tested)
    np.linalg.matrix_rank,  # returns pure numbers
    np.linalg.qr,  # works out of the box (tested)
    np.linalg.slogdet,  # undefined units
    np.linalg.cond,  # works out of the box (tested)
    np.gradient,  # works out of the box (tested)
    np.cumsum,  # works out of the box (tested)
    np.nancumsum,  # works out of the box (tested)
    np.nancumprod,  # we get it for free with np.cumprod (tested)
    np.bincount,  # works out of the box (tested)
    np.unique,  # works out of the box (tested)
    np.take,  # works out of the box (tested)
    np.min_scalar_type,  # returns dtypes
    np.extract,  # works out of the box (tested)
    np.setxor1d,  # we get it for free with previously implemented functions (tested)
    np.lexsort,  # returns pure numbers
    np.digitize,  # returns pure numbers
    np.tril_indices_from,  # returns pure numbers
    np.triu_indices_from,  # returns pure numbers
    np.imag,  # works out of the box (tested)
    np.real,  # works out of the box (tested)
    np.real_if_close,  # works out of the box (tested)
    np.einsum_path,  # returns pure numbers
    np.cov,  # returns pure numbers
    np.corrcoef,  # returns pure numbers
    np.compress,  # works out of the box (tested)
    np.take_along_axis,  # works out of the box (tested)
}


if NUMPY_VERSION >= Version("2.0.0dev0"):
    # the followin all work out of the box (tested)
    NOOP_FUNCTIONS |= {
        np.linalg.cross,
        np.linalg.diagonal,
        np.linalg.matmul,
        np.linalg.matrix_norm,
        np.linalg.matrix_transpose,
        np.linalg.svdvals,
        np.linalg.tensordot,
        np.linalg.trace,
        np.linalg.vecdot,
        np.linalg.vector_norm,
        np.astype,
        np.matrix_transpose,
        np.unique_all,
        np.unique_counts,
        np.unique_inverse,
        np.unique_values,
        np.vecdot,
    }

# Functions for which behaviour is intentionally left to default
IGNORED_FUNCTIONS = {
    np.i0,
    # IO functions (no way to add units)
    np.save,
    np.savez,
    np.savez_compressed,
}


DEPRECATED_FUNCTIONS = {
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
    "msort",  # deprecated in numpy 1.24
    # numpy 1.25 deprecations
    "product",
    "cumproduct",
    "round_",  # removed in 2.0
    "sometrue",
    "alltrue",
}

NOT_HANDLED_FUNCTIONS = NOOP_FUNCTIONS | UNSUPPORTED_FUNCTIONS | IGNORED_FUNCTIONS

for func in DEPRECATED_FUNCTIONS:
    if hasattr(np, func):
        NOT_HANDLED_FUNCTIONS.add(getattr(np, func))


def get_decorators(func):
    # adapted from
    # https://stackoverflow.com/questions/3232024/introspection-to-get-decorator-names-on-a-method
    import ast
    import inspect

    target = func
    decorators = {}

    def visit_FunctionDef(node):
        decorators[node.name] = []
        for n in node.decorator_list:
            name = ""
            if isinstance(n, ast.Call):
                name = n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
            else:
                name = n.attr if isinstance(n, ast.Attribute) else n.id

            decorators[node.name].append(name)

    node_iter = ast.NodeVisitor()
    node_iter.visit_FunctionDef = visit_FunctionDef
    try:
        node_iter.visit(ast.parse(inspect.getsource(target)))
        return decorators[func.__name__]
    except TypeError:
        # may be raised if inspecting a C compiled function
        # in which case, we return with empty hands
        return []


def get_wrapped_functions(*modules):
    """get functions that support __array_function__ in modules

    This was adapted from astropy's tests
    """
    wrapped_functions = {}
    for mod in modules:
        for name, f in mod.__dict__.items():
            if callable(f) and hasattr(f, "__wrapped__"):
                if (
                    f is np.printoptions
                    or f.__name__.startswith("_")
                    or "deprecate" in get_decorators(f)
                ):
                    continue
                wrapped_functions[mod.__name__ + "." + name] = f
    return dict(sorted(wrapped_functions.items()))


def test_wrapping_completeness():
    """Ensure we wrap all numpy functions that support __array_function__"""
    handled_numpy_functions = set(HANDLED_FUNCTIONS.keys())
    # ensure no functions appear in both NOT_HANDLED_FUNCTIONS and HANDLED_FUNCTIONS
    assert NOT_HANDLED_FUNCTIONS.isdisjoint(
        handled_numpy_functions
    ), NOT_HANDLED_FUNCTIONS.intersection(handled_numpy_functions)
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


@pytest.mark.parametrize(
    "arrays",
    [
        [np.array([1]), [2] * Unit()],
        [np.array([1]), [2] * Unit(registry=UnitRegistry())],
        [[1], [2] * Unit()],
    ],
)
def test_unit_validation(arrays):
    # see https://github.com/yt-project/unyt/issues/462
    # numpy.concatenate isn't essential to this test
    # what we're really testing is the unit consistency validation
    # underneath, but we do so using public API
    res = np.concatenate(arrays)
    assert res.units.is_dimensionless


def test_unit_validation_dimensionless_factors():
    # see https://github.com/yt-project/unyt/issues/477
    # numpy.concatenate isn't essential to this test
    # what we're really testing is the unit consistency validation
    # underneath, but we do so using public API
    res = np.concatenate([[1] * cm, unyt_array([1], "cm*dimensionless")])
    assert res.units is cm


def test_array_repr():
    arr = [1, 2, 3] * cm
    assert re.fullmatch(r"unyt_array\(\[1, 2, 3\], (units=)?'cm'\)", np.array_repr(arr))


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
    rng = np.random.default_rng()
    arr = rng.random((3, 3)) * cm
    iarr = np.linalg.inv(arr)
    assert 1 * iarr.units == 1 / cm


def test_linalg_tensorinv():
    a = np.eye(4 * 6) * cm
    a.shape = (4, 6, 8, 3)
    ia = np.linalg.tensorinv(a)
    assert 1 * ia.units == 1 / cm


def test_linalg_pinv():
    rng = np.random.default_rng()
    a = rng.standard_normal(size=(9, 6)) * cm
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


@pytest.mark.skipif(
    NUMPY_VERSION < Version("2.0.0dev0"), reason="linalg.diagonal is new in numpy 2.0"
)
def test_linalg_diagonal():
    a = np.eye(3) * cm
    b = np.linalg.diagonal(a)
    assert type(b) is unyt_array
    assert b.units == a.units


@pytest.mark.skipif(
    NUMPY_VERSION < Version("2.0.0dev0"), reason="linalg.trace is new in numpy 2.0"
)
def test_linalg_trace():
    a = np.eye(3) * cm
    b = np.linalg.trace(a)
    assert type(b) is unyt_quantity
    assert b.units == a.units


@pytest.mark.skipif(
    NUMPY_VERSION < Version("2.0.0dev0"), reason="linalg.outer is new in numpy 2.0"
)
def test_linalg_outer():
    a = np.arange(10) * cm
    assert_array_equal_units(np.linalg.outer(a, a), np.outer(a, a))


@pytest.mark.skipif(
    NUMPY_VERSION < Version("2.0.0dev0"), reason="linalg.cross is new in numpy 2.0"
)
def test_linalg_cross():
    a = np.arange(3) * cm
    assert_array_equal_units(np.linalg.cross(a, a), np.cross(a, a))


@pytest.mark.skipif(
    NUMPY_VERSION < Version("2.0.0dev0"), reason="linalg.matmul is new in numpy 2.0"
)
def test_linalg_matmul():
    a = np.eye(3) * cm
    assert_array_equal_units(np.linalg.matmul(a, a), np.matmul(a, a))


@pytest.mark.skipif(
    NUMPY_VERSION < Version("2.0.0dev0"),
    reason="linalg.matrix_norm is new in numpy 2.0",
)
def test_linalg_matrix_norm():
    a = np.eye(3) * cm
    assert_array_equal_units(np.linalg.matrix_norm(a), np.linalg.norm(a))


@pytest.mark.skipif(
    NUMPY_VERSION < Version("2.0.0dev0"), reason="matrix_transpose is new in numpy 2.0"
)
@pytest.mark.parametrize("namespace", [None, "linalg"])
def test_matrix_transpose(namespace):
    if namespace is None:
        func = np.matrix_transpose
    else:
        func = getattr(np, namespace).matrix_transpose
    a = np.arange(0, 9).reshape(3, 3)
    assert_array_equal_units(func(a), np.transpose(a))


@pytest.mark.skipif(
    NUMPY_VERSION < Version("2.0.0dev0"), reason="vecdot is new in numpy 2.0"
)
@pytest.mark.parametrize("namespace", [None, "linalg"])
def test_vecdot(namespace):
    if namespace is None:
        func = np.vecdot
    else:
        func = getattr(np, namespace).vecdot
    a = np.arange(0, 9)
    assert_array_equal_units(func(a, a), np.vdot(a, a))


@pytest.mark.skipif(
    NUMPY_VERSION < Version("2.0.0dev0"),
    reason="linalg.vector_norm is new in numpy 2.0",
)
def test_linalg_vector_norm():
    a = np.arange(0, 9)
    assert_array_equal_units(np.linalg.vector_norm(a), np.linalg.norm(a))


def test_linalg_svd():
    rng = np.random.default_rng()
    a = (rng.standard_normal(size=(9, 6)) + 1j * rng.standard_normal(size=(9, 6))) * cm
    u, s, vh = np.linalg.svd(a)
    assert type(u) is np.ndarray
    assert type(vh) is np.ndarray
    assert type(s) is unyt_array
    assert s.units == cm

    s = np.linalg.svd(a, compute_uv=False)
    assert type(s) is unyt_array
    assert s.units == cm


@pytest.mark.skipif(
    NUMPY_VERSION < Version("2.0.0dev0"), reason="linalg.svdvals is new in numpy 2.0"
)
def test_linalg_svdvals():
    q = np.arange(9).reshape(3, 3) * cm

    _, ref, _ = np.linalg.svd(q)
    res = np.linalg.svdvals(q)
    assert type(res) is unyt_array
    assert_allclose(res, ref, rtol=5e-16)


@pytest.mark.skipif(
    NUMPY_VERSION < Version("2.0.0dev0"), reason="linalg.tensordot is new in numpy 2.0"
)
def test_linalg_tensordot():
    q = np.arange(9).reshape(3, 3) * cm
    ref = np.tensordot(q, q)
    res = np.linalg.tensordot(q, q)
    assert_array_equal_units(res, ref)


def test_histogram():
    rng = np.random.default_rng()
    arr = rng.normal(size=1000) * cm
    counts, bins = np.histogram(arr, bins=10, range=(arr.min(), arr.max()))
    assert type(counts) is np.ndarray
    assert bins.units == arr.units


def test_histogram_implicit_units():
    # see https://github.com/yt-project/unyt/issues/465
    rng = np.random.default_rng()
    arr = rng.normal(size=1000) * cm
    counts, bins = np.histogram(arr, bins=10, range=(arr.min().value, arr.max().value))
    assert type(counts) is np.ndarray
    assert bins.units == arr.units


def test_histogram2d():
    rng = np.random.default_rng()
    x = rng.normal(size=100) * cm
    y = rng.normal(loc=10, size=100) * s
    counts, xbins, ybins = np.histogram2d(x, y)
    assert counts.ndim == 2
    assert xbins.units == x.units
    assert ybins.units == y.units


def test_histogramdd():
    rng = np.random.default_rng()
    x = rng.normal(size=100) * cm
    y = rng.normal(size=100) * s
    z = rng.normal(size=100) * g
    counts, (xbins, ybins, zbins) = np.histogramdd((x, y, z))
    assert counts.ndim == 3
    assert xbins.units == x.units
    assert ybins.units == y.units
    assert zbins.units == z.units


def test_histogram_bin_edges():
    rng = np.random.default_rng()
    arr = rng.normal(size=1000) * cm
    bins = np.histogram_bin_edges(arr)
    assert type(bins) is unyt_array
    assert bins.units == arr.units


def test_concatenate():
    rng = np.random.default_rng()
    x1 = rng.normal(size=100) * cm
    x2 = rng.normal(size=100) * cm
    res = np.concatenate((x1, x2))
    assert res.units == cm
    assert res.shape == (200,)


def test_concatenate_different_units():
    rng = np.random.default_rng()
    x1 = rng.normal(size=100) * cm
    x2 = rng.normal(size=100) * s
    with pytest.raises(
        UnitInconsistencyError,
        match=(
            r"Expected all unyt_array arguments to have identical units\. "
            r"Received mixed units \(cm, s\)"
        ),
    ):
        np.concatenate((x1, x2))


def test_cross():
    x1 = [1, 2, 3] * cm
    x2 = [4, 5, 6] * s
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


@pytest.mark.parametrize("func", [np.vstack, np.hstack, np.dstack, np.column_stack])
def test_xstack(func):
    x1 = [0, 1, 2] * cm
    x2 = [3, 4, 5] * cm
    res = func((x1, x2))
    assert type(res) is unyt_array
    assert res.units == cm


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

    x = [1, 2, 3] * cm
    assert np.any(x >= 3)
    assert np.any(x >= 3 * cm)
    assert not np.any(x >= 3 * km)


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


@pytest.mark.skipif(
    NUMPY_VERSION >= Version("2.0.0dev0"), reason="np.asfarray is removed in numpy 2.0"
)
def test_asfarray():
    x1 = np.eye(3, dtype="int64") * cm

    x2 = np.asfarray(x1)  # noqa: NPY201
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


if NUMPY_VERSION >= Version("2.0.0dev0"):
    _trapezoid_func = np.trapezoid
else:
    _trapezoid_func = np.trapz


def test_trapezoid_no_x():
    y = [0, 1, 2, 3] * cm
    res = _trapezoid_func(y)
    assert type(res) is unyt_quantity
    assert res.units == cm


def test_trapezoid_with_raw_x():
    y = [0, 1, 2, 3] * cm
    x = [0, 1, 2, 3]
    res = _trapezoid_func(y, x)
    assert type(res) is unyt_quantity
    assert res.units == cm


def test_trapezoid_with_unit_x():
    y = [0, 1, 2, 3] * cm
    x = [0, 1, 2, 3] * s
    res = _trapezoid_func(y, x)
    assert type(res) is unyt_quantity
    assert res.units == cm * s


def test_trapezoid_with_raw_dx():
    y = [0, 1, 2, 3] * cm
    dx = 2.0
    res = _trapezoid_func(y, dx=dx)
    assert type(res) is unyt_quantity
    assert res.units == cm


def test_trapezoid_with_unit_dx():
    y = [0, 1, 2, 3] * cm
    dx = 2.0 * s
    res = _trapezoid_func(y, dx=dx)
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


def test_shares_memory():
    x = [1, 2, 3] * cm
    assert np.shares_memory(x, x.view(np.ndarray))


def test_nonzero():
    x = [1, 2, 0] * cm
    res = np.nonzero(x)
    assert len(res) == 1
    np.testing.assert_array_equal(res[0], [0, 1])

    res2 = np.flatnonzero(x)
    np.testing.assert_array_equal(res[0], res2)


def test_isinf():
    x = [1, float("inf"), float("-inf")] * cm
    res = np.isneginf(x)
    np.testing.assert_array_equal(res, [False, False, True])
    res = np.isposinf(x)
    np.testing.assert_array_equal(res, [False, True, False])


def test_allclose():
    x = [1, 2, 3] * cm
    y = [1, 2, 3] * km
    assert not np.allclose(x, y)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ([1, 2, 3] * cm, [1, 2, 3] * km, [False] * 3),
        ([1, 2, 3] * cm, [1, 2, 3], [True] * 3),
        ([1, 2, 3] * K, [-272.15, -271.15, -270.15] * degC, [True] * 3),
    ],
)
def test_isclose(a, b, expected):
    res = np.isclose(a, b)
    np.testing.assert_array_equal(res, expected)


def test_isclose_error():
    x = [1, 2, 3] * cm
    y = [1, 2, 3] * g
    with pytest.raises(UnitConversionError):
        np.isclose(x, y)


@pytest.mark.parametrize(
    "func",
    [
        np.linspace,
        np.logspace,
        np.geomspace,
    ],
)
def test_xspace(func):
    res = func(1 * cm, 11 * cm, 10)
    assert type(res) is unyt_array
    assert res.units == cm


def test_full_like():
    x = [1, 2, 3] * cm
    res = np.full_like(x, 6 * cm)
    assert type(res) is unyt_array
    assert res.units == cm


@pytest.mark.parametrize(
    "func",
    [
        np.empty_like,
        np.zeros_like,
        np.ones_like,
    ],
)
def test_x_like(func):
    x = unyt_array([1, 2, 3], cm, dtype="float32")
    res = func(x)
    assert type(res) is unyt_array
    assert res.units == x.units
    assert res.shape == x.shape
    assert res.dtype == x.dtype


def test_copy():
    x = [1, 2, 3] * cm
    y = np.copy(x)
    # by default, subok=False, so we shouldn't
    # expect a unyt_array without switching this arg
    assert type(y) is np.ndarray


def test_copy_subok():
    x = [1, 2, 3] * cm
    y = np.copy(x, subok=True)
    assert type(y) is unyt_array
    assert y.units == cm


def test_copyto():
    x = [1, 2, 3] * cm
    y = np.empty_like(x)
    np.copyto(y, x)
    assert type(y) is unyt_array
    assert y.units == cm
    np.testing.assert_array_equal(x, y)


def test_copyto_edge_cases():
    x = [1, 2, 3] * cm
    y = [1, 2, 3] * g
    # copying to an array with a different unit is supported
    # to be in line with how we treat the 'out' param in most
    # numpy operations
    np.copyto(y, x)
    assert type(y) is unyt_array
    assert y.units == cm

    y = np.empty_like(x.view(np.ndarray))
    np.copyto(y, x)
    assert type(y) is np.ndarray


@pytest.mark.skipif(
    NUMPY_VERSION < Version("2.0.0dev0"), reason="astype is new in numpy 2.0"
)
def test_astype():
    x = np.array([1, 2, 3], dtype="int64") * cm
    res = np.astype(x, "int32")
    assert type(res) is unyt_array
    assert res.units == cm


def test_meshgrid():
    x = [1, 2, 3] * cm
    y = [1, 2, 3] * s
    x2d, y2d = np.meshgrid(x, y)
    assert type(x2d) is unyt_array
    assert type(y2d) is unyt_array
    assert x2d.units == cm
    assert y2d.units == s


@pytest.mark.parametrize(
    "func, args, kwargs",
    [
        (np.transpose, (), {}),
        (np.reshape, ((9, 2),), {}),
        (np.resize, ((3, 6),), {}),
        (np.expand_dims, (0,), {}),
        (np.squeeze, (), {}),
        (np.swapaxes, (0, 1), {}),
        (np.moveaxis, (0, 2), {}),
        (np.rot90, (), {}),
        (np.roll, (3,), {}),
        (np.rollaxis, (2,), {}),
        (np.flip, (), {}),
        (np.fliplr, (), {}),
        (np.flipud, (), {}),
        (np.broadcast_to, ((1, 1, 2, 3, 3),), {"subok": True}),
        (np.delete, (0, 1), {}),
        (np.partition, (2,), {}),
    ],
)
def test_reshaper(func, args, kwargs):
    x = [
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            [
                [10, 11, 12],
                [13, 14, 15],
                [16, 17, 18],
            ],
        ]
    ] * cm
    y = func(x, *args, **kwargs)
    assert type(y) is unyt_array
    assert y.units == cm


def test_broadcast_arrays():
    x = [1, 2, 3] * cm
    y = [
        4,
    ] * g
    res = np.broadcast_arrays(x, y, subok=True)
    assert all(type(_) is unyt_array for _ in res)


@pytest.mark.parametrize(
    "func, args",
    [
        (np.split, (3, 2)),
        (np.dsplit, (3,)),
        (np.hsplit, (2,)),
        (np.vsplit, (1,)),
        (np.array_split, (3,)),
    ],
)
def test_xsplit(func, args):
    x = [
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            [
                [10, 11, 12],
                [13, 14, 15],
                [16, 17, 18],
            ],
        ]
    ] * cm
    y = func(x, *args)
    assert all(type(_) is unyt_array for _ in y)
    assert all(_.units == cm for _ in y)


@pytest.mark.parametrize(
    "func, expected_units",
    [
        (np.prod, cm**9),
        (np.var, cm**2),
        (np.std, cm),
        (np.nanprod, cm**9),
        (np.nansum, cm),
        (np.nanvar, cm**2),
        (np.nanstd, cm),
        (np.trace, cm),
    ],
)
def test_scalar_reducer(func, expected_units):
    x = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ] * cm
    y = func(x)
    assert type(y) is unyt_quantity
    assert y.units == expected_units


@pytest.mark.parametrize(
    "func",
    [
        np.percentile,
        np.quantile,
        np.nanpercentile,
        np.nanquantile,
    ],
)
def test_percentile(func):
    x = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ] * cm
    y = func(x, 1)
    assert type(y) is unyt_quantity
    assert y.units == cm


@pytest.mark.parametrize(
    "func",
    [
        np.diag,
        np.diagflat,
        np.diagonal,
    ],
)
def test_diagx(func):
    x = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ] * cm
    y = func(x)
    assert type(y) is unyt_array
    assert y.units == cm


def test_fix():
    y = np.fix(1.2 * cm)
    assert y == 1.0 * cm


def test_linalg_matrix_power():
    x = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ] * cm
    y = np.linalg.matrix_power(x, 2)
    assert type(y) is unyt_array
    assert y.units == cm**2


def test_linalg_det():
    x = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ] * cm
    y = np.linalg.det(x)
    assert type(y) is unyt_quantity
    assert y.units == cm ** (len(x))


def test_linalg_cholesky():
    x = np.eye(3) * cm
    y = np.linalg.cholesky(x)
    assert type(y) is unyt_array
    assert y.units == cm


def test_linalg_lstsq():
    a = np.eye(3) * cm
    b = np.ones(3).T * g
    # setting rcond explicitly to avoid a FutureWarning
    # see https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
    x, residuals, rank, s = np.linalg.lstsq(a, b, rcond=-1)

    assert type(x) is unyt_array
    assert x.units == g / cm
    assert type(residuals) is unyt_array
    assert residuals.units == g / cm
    assert type(s) is unyt_array
    assert s.units == cm


def test_linalg_multi_dot():
    a = np.eye(3) * cm
    b = np.eye(3) * g
    c = np.eye(3) * s
    res = np.linalg.multi_dot([a, b, c])
    assert type(res) is unyt_array
    assert res.units == cm * g * s


def test_linalg_qr():
    x = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ] * cm
    q, r = np.linalg.qr(x)
    assert type(q) is unyt_array
    assert q.units == cm
    assert type(r) is unyt_array
    assert r.units == cm


@pytest.mark.parametrize("func", [np.linalg.solve, np.linalg.tensorsolve])
def test_linalg_solve(func):
    a = np.eye(3) * cm
    b = np.ones(3).T * g

    x = func(a, b)
    assert type(x) is unyt_array
    assert x.units == g / cm


def is_any_dimless(x) -> bool:
    return (not hasattr(x, "units")) or x.units.is_dimensionles


def test_linalg_cond():
    a = np.eye(3) * cm
    res = np.linalg.cond(a)
    assert is_any_dimless(res)


@pytest.mark.parametrize("func", [np.linalg.eig, np.linalg.eigh])
def test_eig(func):
    a = np.eye(3) * cm
    w, v = func(a)
    assert type(w) is unyt_array
    assert w.units == cm
    assert is_any_dimless(v)


@pytest.mark.parametrize("func", [np.linalg.eigvals, np.linalg.eigvalsh])
def test_eigvals(func):
    a = np.eye(3) * cm
    w = func(a)
    assert type(w) is unyt_array
    assert w.units == cm


def test_savetxt(tmp_path):
    a = [1, 2, 3] * cm
    with pytest.raises(
        UserWarning,
        match=re.escape(
            "numpy.savetxt does not preserve units, "
            "and will only save the raw numerical data from the unyt_array object.\n"
            "If this is the intended behaviour, call `numpy.savetxt(file, arr.d)` "
            "to silence this warning.\n"
            "If you want to preserve units, use `unyt.savetxt` "
            "(and `unyt.loadtxt`) instead."
        ),
    ):
        np.savetxt(tmp_path / "savefile.npy", a)

    # check that doing what the warning says doesn't trigger any other warning
    np.savetxt(tmp_path / "savefile.npy", a.d)


def test_apply_along_axis():
    a = np.eye(3) * cm
    ret = np.apply_along_axis(lambda x: x * cm, 0, a)
    assert type(ret) is unyt_array
    assert ret.units == cm**2


def test_apply_over_axes():
    a = np.eye(3) * cm
    ret = np.apply_over_axes(lambda x, axis: x * cm, a, (0, 1))
    assert type(ret) is unyt_array
    assert ret.units == cm**3


def test_array_equal():
    a = [1, 2, 3] * cm
    b = [1, 2, 3] * cm
    c = [1, 2, 3] * km
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_array_equiv():
    a = [1, 2, 3] * cm
    b = [1, 2, 3] * cm
    c = [1, 2, 3] * km
    d = [[1, 2, 3]] * cm
    assert np.array_equiv(a, b)
    assert np.array_equiv(a, d)
    assert not np.array_equiv(a, c)


def test_common_type():
    a = np.array([1, 2, 3], dtype="float32") * cm
    b = np.array([1, 2, 3], dtype="float64") * cm
    dtype = np.common_type(a, b)
    assert dtype == np.dtype("float64")


def test_result_type():
    scalar = 3 * cm
    array = np.arange(7, dtype="i1")
    if NUMPY_VERSION >= Version("2.0.0dev0"):
        # promotion rules vary under NEP 50. The default behaviour is different
        # in numpy 2.0 VS numpy 1.x
        # see https://github.com/numpy/numpy/pull/23912
        # see https://numpy.org/neps/nep-0050-scalar-promotion.html
        expected_dtype = scalar.dtype
    else:
        expected_dtype = array.dtype

    assert np.result_type(scalar, array) == expected_dtype


@pytest.mark.parametrize(
    "func",
    [
        np.diff,
        np.ediff1d,
        np.gradient,
        np.ptp,
    ],
)
@pytest.mark.parametrize("input_units, output_units", [(cm, cm), (K, delta_degC)])
def test_deltas(func, input_units, output_units):
    x = np.arange(0, 4) * input_units
    res = func(x)
    assert isinstance(res, unyt_array)
    assert res.units == output_units


@pytest.mark.parametrize(
    "func",
    [
        np.cumsum,
        np.nancumsum,
    ],
)
def test_cumsum(func):
    a = [1, 2, 3] * cm
    res = func(a)
    assert type(res) is unyt_array
    assert res.units == cm


@pytest.mark.parametrize(
    "func",
    [
        np.cumprod,
        np.nancumprod,
    ],
)
def test_cumprod(func):
    a = [1, 2, 3] * cm
    with pytest.raises(
        UnytError,
        match=re.escape(
            r"numpy.cumprod (and other cumulative product function) cannot be used "
            r"with a unyt_array as all return elements should (but cannot) "
            r"have different units."
        ),
    ):
        func(a)


def test_bincount():
    a = [1, 2, 3] * cm
    res = np.bincount(a)
    assert type(res) is np.ndarray


def test_unique():
    a = [1, 2, 3] * cm
    res = np.unique(a)
    assert type(res) is unyt_array
    assert res.units == cm


@pytest.mark.skipif(
    NUMPY_VERSION < Version("2.0.0dev0"), reason="unique_all is new in numpy 2.0"
)
def test_unique_all():
    q = np.arange(9).reshape(3, 3) * cm
    values, indices, inverse_indices, counts = np.unique(
        q,
        return_index=True,
        return_inverse=True,
        return_counts=True,
        equal_nan=False,
    )
    res = np.unique_all(q)
    assert len(res) == 4
    assert_array_equal_units(res.values, values)
    assert_array_equal_units(res.indices, indices)
    assert_array_equal_units(res.inverse_indices, inverse_indices)
    assert_array_equal_units(res.counts, counts)


@pytest.mark.skipif(
    NUMPY_VERSION < Version("2.0.0dev0"), reason="unique_counts is new in numpy 2.0"
)
def test_unique_counts():
    q = np.arange(9).reshape(3, 3) * cm
    values, counts = np.unique(
        q,
        return_counts=True,
        equal_nan=False,
    )
    res = np.unique_counts(q)
    assert len(res) == 2
    assert_array_equal_units(res.values, values)
    assert_array_equal_units(res.counts, counts)


@pytest.mark.skipif(
    NUMPY_VERSION < Version("2.0.0dev0"), reason="unique_inverse is new in numpy 2.0"
)
def test_unique_inverse():
    q = np.arange(9).reshape(3, 3) * cm
    values, inverse_indices = np.unique(
        q,
        return_inverse=True,
        equal_nan=False,
    )
    res = np.unique_inverse(q)
    assert len(res) == 2
    assert_array_equal_units(res.values, values)
    assert_array_equal_units(res.inverse_indices, inverse_indices)


@pytest.mark.skipif(
    NUMPY_VERSION < Version("2.0.0dev0"), reason="unique_values is new in numpy 2.0"
)
def test_unique_values():
    q = np.arange(9).reshape(3, 3) * cm
    values = np.unique(q, equal_nan=False)
    res = np.unique_values(q)
    assert_array_equal_units(res, values)


def test_take():
    a = [1, 2, 3] * cm
    res = np.take(a, [0, 1])
    assert type(res) is unyt_array
    assert res.units == cm


def test_pad():
    a = [1, 2, 3] * cm
    res = np.pad(a, [0, 1])
    assert type(res) is unyt_array
    assert res.units == cm


def test_sinc():
    a = [1, 2, 3] * cm
    res = np.sinc(a)
    # we *want* this one to ignore units
    assert type(res) is np.ndarray


def test_choose_mixed_units():
    choices = [[1, 2, 3] * cm, [4, 5, 6] * km]
    with pytest.raises(UnitInconsistencyError):
        np.choose([1, 0, 1], choices=choices)


def test_choose():
    choices = [[1, 2, 3] * cm, [4, 5, 6] * cm]
    res = np.choose([1, 0, 1], choices=choices)
    assert type(res) is unyt_array
    assert res.units == cm


def test_extract():
    a = [1, 2, 3] * cm
    res = np.extract(a > 1 * cm, a)
    assert type(res) is unyt_array
    assert res.units == cm


def test_fill_diagonal_mixed_units():
    a = np.zeros(9).reshape((3, 3)) * cm
    with pytest.raises(UnitInconsistencyError):
        np.fill_diagonal(a, 1 * km)


@pytest.mark.parametrize("val", [1 * cm, 1])
def test_fill_diagonal(val):
    a = np.zeros(9).reshape((3, 3)) * cm
    np.fill_diagonal(a, val)
    assert type(a) is unyt_array
    assert a.units == cm


def test_insert_mixed_units():
    a = [1, 2, 3] * cm
    with pytest.raises(UnitInconsistencyError):
        np.insert(a, 1, 42 * km)


@pytest.mark.parametrize("val", [42, 42 * cm])
def test_insert(val):
    a = [1, 2, 3] * cm
    res = np.insert(a, 1, val)
    assert type(res) is unyt_array
    assert res.units == cm


def test_isin_mixed_units():
    a = [1, 2, 3] * cm
    with pytest.raises(UnitInconsistencyError):
        np.isin(1, a)


def test_isin():
    a = [1, 2, 3] * cm
    assert np.isin(1 * cm, a)


@pytest.mark.filterwarnings("ignore:`in1d` is deprecated. Use `np.isin` instead.")
def test_in1d_mixed_units():
    a = [1, 2, 3] * cm
    with pytest.raises(UnitInconsistencyError):
        np.in1d([1, 2], a)


@pytest.mark.filterwarnings("ignore:`in1d` is deprecated. Use `np.isin` instead.")
def test_in1d():
    a = [1, 2, 3] * cm
    b = [1, 2] * cm
    assert np.all(np.in1d(b, a))


def test_place_mixed_units():
    arr = np.arange(6).reshape(2, 3) * cm
    with pytest.raises(UnitInconsistencyError):
        np.place(arr, arr > 2, [44, 55])


def test_place():
    arr = np.arange(6).reshape(2, 3) * cm
    np.place(arr, arr > 2, [44, 55] * cm)
    assert type(arr) is unyt_array
    assert arr.units == cm


def test_put_mixed_units():
    arr = np.arange(6).reshape(2, 3) * cm
    with pytest.raises(UnitInconsistencyError):
        np.put(arr, [1, 2], [44, 55])


def test_put():
    arr = np.arange(6).reshape(2, 3) * cm
    np.put(arr, [1, 2], [44, 55] * cm)
    assert type(arr) is unyt_array
    assert arr.units == cm


def test_put_along_axis_mixed_units():
    arr = np.arange(6).reshape(2, 3) * cm
    with pytest.raises(UnitInconsistencyError):
        np.put_along_axis(arr, np.array([[1, 2], [0, 1]]), [44, 55], 1)


def test_put_along_axis():
    arr = np.arange(6).reshape(2, 3) * cm
    np.put_along_axis(arr, np.array([[1, 2], [0, 1]]), [44, 55] * cm, 1)
    assert type(arr) is unyt_array
    assert arr.units == cm


def test_putmask_mixed_units():
    arr = np.arange(6, dtype=np.int_).reshape(2, 3) * cm
    with pytest.raises(UnitInconsistencyError):
        np.putmask(arr, arr > 2 * cm, np.zeros_like(arr.d))


def test_putmask():
    arr = np.arange(6, dtype=np.int_).reshape(2, 3) * cm
    np.putmask(arr, arr > 2 * cm, np.zeros_like(arr))

    assert type(arr) is unyt_array
    assert arr.units == cm


def test_searchsorted_mixed_units():
    with pytest.raises(UnitInconsistencyError):
        np.searchsorted([1, 2, 3, 4, 5] * cm, 3 * km)


@pytest.mark.parametrize("val", [3 * cm, 3])
def test_searchsorted(val):
    res = np.searchsorted([1, 2, 3, 4, 5] * cm, val)
    assert res == 2


@pytest.mark.parametrize(
    "choicelist_gen, default",
    [
        ([lambda x: x**3, lambda x: x**2], 34 * cm),  # invalid choicelist
        ([lambda x: x, lambda x: x + 3 * x.units], 34),  # invalid default
    ],
)
def test_select_mixed_units(choicelist_gen, default):
    a = [1, 2, 3, 4, 5, 6] * cm
    with pytest.raises(UnitInconsistencyError):
        np.select(
            [a > 3 * cm, a < 3 * cm], [f(a) for f in choicelist_gen], default=default
        )


def test_select():
    a = [1, 2, 3, 4, 5, 6] * cm
    res = np.select([a > 3 * cm, a < 3 * cm], [a, a + 3 * cm], default=34 * cm)
    assert_array_equal_units(res, [4, 5, 34, 4, 5, 6] * cm)


def test_setdiff1d_mixed_units():
    a = [1, 2, 3] * cm
    b = [0, 1, 2]
    with pytest.raises(UnitInconsistencyError):
        np.setdiff1d(a, b)


def test_setdiff1d():
    a = [1, 2, 3] * cm
    b = [0, 1, 2] * cm
    res = np.setdiff1d(a, b)
    assert_array_equal_units(res, [3] * cm)


def test_setxor1d_mixed_units():
    a = [1, 2, 3] * cm
    b = [0, 1, 2]
    with pytest.raises(UnitInconsistencyError):
        np.setxor1d(a, b)


def test_setxor1d():
    a = [1, 2, 3] * cm
    b = [0, 1, 2] * cm
    res = np.setxor1d(a, b)
    assert_array_equal_units(res, [0, 3] * cm)


def test_clip_mixed_units():
    a = [1, 2, 3, 4, 5, 6] * cm
    with pytest.raises(UnitInconsistencyError):
        np.clip(a, 3 * cm, 4)


@pytest.mark.parametrize("vmin,vmax", [(3 * cm, 4 * cm), (3, 4)])
def test_clip(vmin, vmax):
    a = [1, 2, 3, 4, 5, 6] * cm
    res = np.clip(a, vmin, vmax)
    assert_array_equal_units(res, [3, 3, 3, 4, 4, 4] * cm)


def test_where_mixed_units():
    x = [-1, 2, -3] * cm
    y = [0, 0, 0]
    with pytest.raises(UnitInconsistencyError):
        np.where(x > y, x, y)


def test_where_single_arg():
    x = [0, 2, -1, 0, 1] * cm
    res = np.where(x)
    assert isinstance(res, tuple)
    assert len(res) == 1
    assert type(res[0]) is np.ndarray
    np.testing.assert_array_equal(res[0], [1, 2, 4])


def test_where_xy():
    x = [-1, 2, -3] * cm
    y = [0, 0, 0] * cm
    res = np.where(x > y, x, y)
    assert type(res) is unyt_array
    assert res.units == cm


@pytest.mark.parametrize(
    "func",
    [
        np.imag,
        np.real,
        np.real_if_close,
    ],
)
def test_complex_reductions(func):
    a = [1 + 1j for _ in range(3)] * A
    res = func(a)
    assert type(res) is unyt_array
    assert res.units == A


@pytest.mark.parametrize(
    "func",
    [np.tril, np.triu],
)
def test_triangles(func):
    a = np.eye(4) * cm
    res = func(a)
    assert type(res) is unyt_array
    assert res.units == cm


def test_einsum():
    a = np.eye(4) * cm

    # extract diagonal
    res = np.einsum("ii->i", a)
    assert type(res) is unyt_array
    assert res.units == cm

    # sum diagonal elements, the result should be a scalar
    res = np.einsum("ii", a)
    assert type(res) is unyt_quantity
    assert res.units == cm


def test_convolve():
    a = [1, 2, 3] * cm
    v = [4, 5, 6] * s
    res = np.convolve(a, v)
    assert type(res) is unyt_array
    assert res.units == cm * s


def test_correlate():
    a = [1, 2, 3] * cm
    v = [4, 5, 6] * s
    res = np.correlate(a, v)
    assert type(res) is unyt_array
    assert res.units == cm * s


def test_tensordot():
    a = np.arange(60.0).reshape(3, 4, 5) * cm
    b = np.arange(24.0).reshape(4, 3, 2) * s
    res = np.tensordot(a, b, axes=([1, 0], [0, 1]))
    assert type(res) is unyt_array
    assert res.units == cm * s


def test_compress():
    a = [1, 2, 3] * cm
    res = np.compress(a > 1, a)
    assert type(res) is unyt_array
    assert res.units == cm

    np.compress(a > 1, a, out=res)
    assert type(res) is unyt_array
    assert res.units == cm

    np.compress(a > 1, a, out=res.view(np.ndarray))
    assert type(res) is unyt_array
    assert res.units == cm


def test_take_along_axis():
    a = np.array([[10, 30, 20], [60, 40, 50]]) * cm
    ai = np.argsort(a, axis=1)
    res = np.take_along_axis(a, ai, axis=1)
    assert type(res) is unyt_array
    assert res.units == cm


def test_unwrap():
    phase = np.linspace(0, np.pi, num=5) * rad
    phase[3:] += np.pi * rad
    res = np.unwrap(phase)
    assert type(res) is unyt_array
    assert res.units == rad


def test_interp():
    _x = np.array([1.1, 2.2, 3.3])
    _xp = np.array([1, 2, 3])
    _fp = np.array([4, 8, 12])

    # any of the three input array-like might be unitful
    # let's test all relevant combinations
    # return type should match fp's

    with pytest.raises(UnitInconsistencyError):
        np.interp(_x * cm, _xp, _fp)

    with pytest.raises(UnitInconsistencyError):
        res = np.interp(_x, _xp * cm, _fp)

    res = np.interp(_x * cm, _xp * cm, _fp)
    assert type(res) is np.ndarray

    res = np.interp(_x, _xp, _fp * K)
    assert type(res) is unyt_array
    assert res.units == K

    res = np.interp(_x * cm, _xp * cm, _fp * K)
    assert type(res) is unyt_array
    assert res.units == K
