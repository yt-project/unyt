import pickle
from collections import defaultdict

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from unyt import unyt_array, unyt_quantity
from unyt._on_demand_imports import _dask as dask

if not dask.__is_available__:
    pytest.skip("dask isn't installed", allow_module_level=True)

from unyt.dask_array import (
    _create_with_quantity,
    _use_unary_decorator,
    reduce_with_units,
    unyt_dask_array,
    unyt_from_dask,
)
from unyt.exceptions import UnitOperationError
from unyt.unit_symbols import cm, g, m


def test_unyt_dask_creation():
    x = dask.array.ones((10, 10), chunks=(2, 2))
    x_da = unyt_from_dask(x, m)
    assert type(x_da) is unyt_dask_array
    assert x_da.units == m
    assert type(x_da.compute()) is unyt_array

    x_da = _create_with_quantity(x, unyt_quantity(1, m))
    assert type(x_da) is unyt_dask_array
    assert x_da.units == m
    assert type(x_da.compute()) is unyt_array

    x_dask = x_da.to_dask()
    assert_array_equal(x.compute(), x_dask.compute())


def test_unyt_dask_slice():
    # tests __getitem__
    x = dask.array.ones((10, 10), chunks=(2, 2))
    x_da = unyt_from_dask(x, m)
    slc = x_da[:, 0]
    assert slc.units == m
    assert slc.compute().units == m
    assert type(slc.compute()) is unyt_array


def test_unyt_set():
    # tests __setitem__
    x = dask.array.ones((10, 10), chunks=(2, 2))
    x_da = unyt_from_dask(x, m)
    x_da[:, 0] = 3.0
    assert (x_da[:, 0].compute() == 3.0).sum() == 10


def test_unit_conversions():
    x = dask.array.ones((10, 10), chunks=(2, 2))
    x_da = unyt_from_dask(x, m)
    x_da = x_da.to(cm)
    assert type(x_da) is unyt_dask_array
    assert x_da.units == cm
    assert x_da.compute().units == cm
    assert x_da[0, 0].compute().value == 100

    x_da_2 = unyt_from_dask(x, m)
    result = x_da + x_da_2
    assert type(result) is unyt_dask_array
    assert result.units == m
    assert result.compute().units == m

    x_da_2 = unyt_from_dask(x, "g")
    with pytest.raises(UnitOperationError):
        x_da + x_da_2


def test_conversion_to_dask():
    x = dask.array.ones((10, 10), chunks=(2, 2))
    x_da = unyt_from_dask(x, m)
    x_again = x_da.to_dask()
    assert_array_equal(x.compute(), x_again.compute())
    assert type(x_again) is type(x)

    result = np.isfinite(x_da)  # should return plain dask array
    assert type(result) is dask.array.core.Array


def unary_test(the_func, unyt_dask_obj, unyt_array_in, *args, **kwargs):
    # the_func can be a numpy ufunc or dask.array numpy ufunc implementation

    result_delay = the_func(unyt_dask_obj, *args, **kwargs)
    correct_unyt = the_func(unyt_array_in, *args, **kwargs)

    assert result_delay.units == correct_unyt.units  # units should already match
    unary_result_test(result_delay, correct_unyt)


def unary_result_test(dask_unyt_delayed, correct_unyt):
    # computes a delayed dask_unyt_array and compares resulting units and values
    result = dask_unyt_delayed.compute()
    assert result.units == correct_unyt.units
    assert type(result) is type(correct_unyt)
    # value comparison:
    if type(result) == unyt_array:
        assert_array_equal(result, correct_unyt)
    else:
        assert result == correct_unyt


def test_unary():
    x = dask.array.ones((10, 10), chunks=(2, 2))
    x_da = unyt_from_dask(x, m)
    x_unyt = unyt_array(np.ones((10, 10)), m)

    for unary_op in [np.sqrt, dask.array.sqrt]:
        unary_test(unary_op, x_da, x_unyt)

    # some of the daskified ufunc attributes need to be called directly
    unary_result_test(x_da.sum(), x_unyt.sum())
    unary_result_test(x_da.min(), x_unyt.min())
    unary_result_test(x_da.max(), x_unyt.max())
    unary_result_test(x_da.mean(), x_unyt.mean())
    unary_result_test(x_da.std(), x_unyt.std())
    unary_result_test(x_da.cumsum(0), x_unyt.cumsum(0))
    unary_result_test(abs(x_da), abs(x_unyt))  # __abs__


@pytest.mark.parametrize(
    "logical_op", ["__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__"]
)
@pytest.mark.parametrize("other", [None, unyt_quantity(2, m)])
def test_logical(logical_op, other):
    def check_operator(arg1, arg2):
        # comparisons should return plain dask arrays
        func = getattr(arg1, logical_op)
        result = func(arg2)
        assert type(result) is dask.array.Array

    x = dask.array.ones((10, 10), chunks=(2, 2))
    x_da = unyt_from_dask(x, m)
    if other is None:
        other = 2 * x_da  # test another unyt-dask array
    check_operator(x_da, other)
    check_operator(other, x_da)


def test_binary():
    x = dask.array.ones((10, 10), chunks=(2, 2))
    x2 = dask.array.full((10, 10), 2, chunks=(2, 2))
    x_da = unyt_from_dask(x, m)
    x_da_2 = unyt_from_dask(x2, g)

    # multiplications
    result = x_da * x_da_2
    assert result.units == m * g
    result = x_da_2 * x_da
    assert result.units == m * g
    result = x_da_2 * 2
    assert result.units == g
    result = 2 * x_da_2
    assert result.units == g
    result = x_da_2 * unyt_quantity(2, "m")
    assert result.units == m * g
    result = unyt_quantity(2, "m") * x_da_2
    assert result.units == m * g

    # divisions
    result = x_da_2 / x_da
    assert result.units == g / m
    result = x_da / x_da_2
    assert result.units == m / g
    result = x_da_2 / 2
    assert result.units == g
    result = 2 / x_da_2  # __rtruediv__
    assert result.units == g**-1
    result = x_da_2 / unyt_quantity(2, "m")
    assert result.units == g / m
    result = unyt_quantity(2, "m") / x_da_2
    assert result.units == m / g

    result = x_da**2
    assert result.units == m * m


def test_addition():
    x = dask.array.ones((10, 10), chunks=(2, 2))
    x2 = dask.array.full((10, 10), 2, chunks=(2, 2))
    x_da = unyt_from_dask(x, m)

    # two unyt_dask_array objects, any order, any units with same dimension
    x_da_2 = unyt_from_dask(x2, m)
    result = x_da + x_da_2
    assert result.units == m
    result = x_da_2 + x_da
    assert result.units == m
    x_da_3 = unyt_from_dask(x2, "cm")
    result = x_da + x_da_3
    assert result.units == m
    result = x_da_3 + x_da
    assert result.units == m

    # one unyt_dask_array, one unyt_quantity, any order, any units with same dim
    result = x_da + unyt_quantity(1, "m")
    assert result.units == m
    result = unyt_quantity(1, "m") + x_da
    assert result.units == m
    result = unyt_quantity(100, "cm") + x_da  # test same dimensions
    assert result.units == m
    assert result.max().compute() == unyt_quantity(2, "m")
    result = x_da + unyt_quantity(100, "cm")  # test same dimensions
    assert result.units == m
    assert result.max().compute() == unyt_quantity(200, "cm")


def test_subtraction():
    x = dask.array.ones((10, 10), chunks=(2, 2))
    x2 = dask.array.full((10, 10), 2, chunks=(2, 2))
    x_da = unyt_from_dask(x, m)

    # two unyt_dask_array objects, any order, any units with same dimension
    x_da_2 = unyt_from_dask(x2, m)
    result = x_da - x_da_2
    assert result.units == m
    result = x_da_2 - x_da
    assert result.units == m
    x_da_3 = unyt_from_dask(x2, "cm")
    result = x_da - x_da_3
    assert result.units == m
    result = x_da_3 - x_da
    assert result.units == m

    # one unyt_dask_array, one unyt_quantity, any order, any units with same dim
    result = x_da - unyt_quantity(1, "m")
    assert result.units == m
    result = unyt_quantity(1, "m") - x_da
    assert result.units == m
    result = unyt_quantity(100, "cm") - x_da  # test same dimensions
    assert result.units == m
    assert result.max().compute() == unyt_quantity(0, "m")
    result = x_da - unyt_quantity(100, "cm")  # test same dimensions
    assert result.units == m
    assert result.max().compute() == unyt_quantity(0, "cm")


def test_unyt_type_result():
    # test that the return type of a compute is unyt_array or unyt_quantity when
    # appropriate

    x = dask.array.ones((10, 10), chunks=(2, 2))
    x_da = unyt_from_dask(x, m)
    x_unyt = unyt_array(np.ones((10, 10)), m)

    result = x_da.compute()
    assert type(result) == unyt_array
    assert_array_equal(result, x_unyt)

    result = x_da.min().compute()
    assert type(result) == unyt_quantity
    assert result == unyt_quantity(1, m)


_func_args = defaultdict(lambda: ())
_func_args["reshape"] = ((100, 1),)
_func_args["rechunk"] = ((5, 5),)
_func_args["cumsum"] = (0,)
_func_args["clip"] = (0, 2)
_func_args["swapaxes"] = (0, 1)
_func_args["repeat"] = (1, 0)
_func_args["astype"] = (int,)
_func_args["topk"] = (1,)


@pytest.mark.parametrize(
    ("da_func", "args"), [(f, _func_args[f]) for f in _use_unary_decorator]
)
def test_dask_passthroughs(da_func, args):
    # tests the array class functions that do not modify units
    x = dask.array.ones((10, 10), chunks=(2, 2))
    x_da = unyt_from_dask(x, m)
    assert getattr(x_da, da_func)(*args).units == m


def test_repr():
    x = dask.array.ones((10, 10), chunks=(2, 2))
    x_da = unyt_from_dask(x, m)
    assert "unyt_dask_array" in x_da.__repr__()
    table_str = x_da._repr_html_()
    assert "Units" in table_str


@pytest.mark.parametrize(
    ("dask_func_str", "actual", "axis", "check_nan"),
    [
        ("mean", unyt_quantity(1.0, "m"), None, True),
        ("min", unyt_quantity(1.0, "m"), None, True),
        ("max", unyt_quantity(1.0, "m"), None, True),
        ("std", unyt_quantity(0.0, "m"), None, True),
        (
            "median",
            unyt_quantity(1.0, "m"),
            1,
            True,
        ),  # median requires an axis for dask
        ("var", unyt_quantity(0.0, "m**2"), None, True),
        ("sum", unyt_quantity(100.0, "m"), None, True),
        (
            "cumsum",
            unyt_array(np.ones((10, 10)), "m").cumsum(axis=1),
            1,
            True,
        ),
        ("average", unyt_quantity(1.0, "m"), None, False),
        ("diagonal", unyt_array(np.ones((10,)), "m"), None, False),
        ("unique", unyt_array([1.0], "m"), None, False),
    ],
)
def test_dask_array_reductions(dask_func_str, actual, axis, check_nan):
    extra_kwargs = {}
    if axis is not None:
        extra_kwargs["axis"] = axis

    func_strs = [
        dask_func_str,
    ]
    if check_nan:
        func_strs.append("nan" + dask_func_str)

    for func_str in func_strs:
        dask_func = getattr(dask.array, func_str)
        x_da = unyt_from_dask(dask.array.ones((10, 10), chunks=(2, 2)), m)

        result = reduce_with_units(dask_func, x_da, **extra_kwargs).compute()
        assert_array_equal(result, actual)


def test_bad_dask_array_reductions():
    x_da = unyt_from_dask(dask.array.ones((10, 10), chunks=(2, 2)), m)

    with pytest.raises(
        ValueError, match="could not deduce np equivalent of dask reduction"
    ):
        reduce_with_units(lambda: None, x_da).compute()


def test_prod():
    x_da = unyt_from_dask(dask.array.ones((10, 10), chunks=(2, 2)), m)
    assert x_da.prod().units == m**100


def test_pickle():
    x_da = unyt_from_dask(dask.array.ones((10, 10), chunks=(2, 2)), m)
    x_da2 = pickle.loads(pickle.dumps(x_da))
    assert_array_equal(x_da, x_da2)
    assert x_da.units == x_da2.units


def test_np_array_funcs():
    # test some numpy array functions to catch __array_function__

    x_da = unyt_from_dask(dask.array.ones((10, 10), chunks=(2, 2)), m)
    actual = np.sum(x_da).compute()
    assert actual == unyt_quantity(100, m)

    plain_dask = dask.array.arange(100, chunks=(20,))
    x_da = unyt_from_dask(plain_dask, m)
    actual = np.mean(x_da).compute()
    expected = unyt_quantity(np.mean(plain_dask).compute(), m)
    assert actual == expected

    actual = np.std(x_da[x_da > 10.0]).compute()
    expected = unyt_quantity(np.std(plain_dask[plain_dask > 10]).compute(), m)
    assert actual == expected

    actual = np.max(np.power(x_da, 2)).compute()
    expected = unyt_quantity(np.max(np.power(plain_dask, 2)).compute(), m * m)
    assert actual == expected
