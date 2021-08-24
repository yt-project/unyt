from numpy.testing import assert_array_equal
from numpy import sqrt, ones, min, add, isfinite, sum
from unyt.dask_array import unyt_from_dask, unyt_dask_array, _create_with_quantity, reduce_with_units
from unyt._on_demand_imports import _dask as dask
from unyt import unyt_array, unyt_quantity
from unyt.unit_symbols import cm, m, g
from unyt.exceptions import UnitOperationError
import pytest


def dask_is_too_old(dask_version_str):

    if dask_version_str is None:
        return True

    major, minor, revis = [int(mmr) for mmr in dask_version_str.split(".")]
    # version 2021.04.1 is min for the __setitem__ test
    if major > 2021:
        return False
    if major < 2021:
        return True
    # major version == 2021 if we are here
    if (minor == 4 and revis >= 1) or minor > 4:  # == 2021
        return False
    return True


requires_dask_2021421 = pytest.mark.skipif(
    dask_is_too_old(dask.__version__), reason="test requires dask>=2021.04.1"
)


def test_dask_version_check():
    assert dask_is_too_old(None) is True
    assert dask_is_too_old("2.9.0") is True
    assert dask_is_too_old("2021.3.0") is True
    assert dask_is_too_old("2021.4.0") is True
    assert dask_is_too_old("2021.4.1") is False
    assert dask_is_too_old("2021.7.0") is False
    assert dask_is_too_old("2022.1.0") is False


def test_unyt_dask_creation():
    x = dask.array.ones((10, 10), chunks=(2, 2))
    x_da = unyt_from_dask(x, m)
    assert type(x_da) == unyt_dask_array
    assert x_da.units == m
    assert type(x_da.compute()) == unyt_array

    x_da = _create_with_quantity(x, unyt_quantity(1, m))
    assert type(x_da) == unyt_dask_array
    assert x_da.units == m
    assert type(x_da.compute()) == unyt_array

    x_dask = x_da.to_dask()
    assert_array_equal(x.compute(), x_dask.compute())


def test_unyt_dask_slice():

    # tests __getitem__
    x = dask.array.ones((10, 10), chunks=(2, 2))
    x_da = unyt_from_dask(x, m)
    slc = x_da[:, 0]
    assert slc.units == m
    assert slc.compute().units == m
    assert type(slc.compute()) == unyt_array


@requires_dask_2021421
def test_unyt_set():
    # tests __setitem__
    x = dask.array.ones((10, 10), chunks=(2, 2))
    x_da = unyt_from_dask(x, m)
    x_da[:, 0] = 3.0
    assert sum(x_da[:, 0].compute() == 3.0) == 10


def test_unit_conversions():
    x = dask.array.ones((10, 10), chunks=(2, 2))
    x_da = unyt_from_dask(x, m)
    x_da = x_da.to(cm)
    assert type(x_da) == unyt_dask_array
    assert x_da.units == cm
    assert x_da.compute().units == cm

    x_da_2 = unyt_from_dask(x, m)
    result = x_da + x_da_2
    assert type(result) == unyt_dask_array
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
    assert type(x_again) == type(x)

    result = isfinite(x_da)  # should return plain dask array
    assert type(result) == dask.array.core.Array


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
    assert type(result) == type(correct_unyt)
    # value comparison:
    if type(result) == unyt_array:
        assert_array_equal(result, correct_unyt)
    else:
        assert result == correct_unyt


def test_unary():

    x = dask.array.ones((10, 10), chunks=(2, 2))
    x_da = unyt_from_dask(x, m)
    x_unyt = unyt_array(ones((10, 10)), m)

    for unary_op in [sqrt, dask.array.sqrt]:
        unary_test(unary_op, x_da, x_unyt)

    # some of the daskified ufunc attributes need to be called directly
    unary_result_test(x_da.sum(), x_unyt.sum())
    unary_result_test(x_da.min(), x_unyt.min())
    unary_result_test(x_da.max(), x_unyt.max())
    unary_result_test(x_da.mean(), x_unyt.mean())
    unary_result_test(x_da.std(), x_unyt.std())
    unary_result_test(x_da.cumsum(0), x_unyt.cumsum(0))
    unary_result_test(abs(x_da), abs(x_unyt))  # __abs__


def test_logical():
    def logical_operators(x_da, other):
        # comparisons should return plain dask arrays
        result = x_da < other
        assert type(result) == dask.array.Array

        result = other > x_da
        assert type(result) == dask.array.Array

        result = x_da > other
        assert type(result) == dask.array.Array

        result = other < x_da
        assert type(result) == dask.array.Array

        result = x_da < other
        assert type(result) == dask.array.Array

        result = x_da <= other
        assert type(result) == dask.array.Array

        result = other >= x_da
        assert type(result) == dask.array.Array

        result = x_da >= other
        assert type(result) == dask.array.Array

        result = other <= x_da
        assert type(result) == dask.array.Array

        result = x_da <= other
        assert type(result) == dask.array.Array

        result = other == x_da
        assert type(result) == dask.array.Array

        result = x_da == other
        assert type(result) == dask.array.Array

        result = x_da != other
        assert type(result) == dask.array.Array

    x = dask.array.ones((10, 10), chunks=(2, 2))
    x_da = unyt_from_dask(x, m)

    logical_operators(x_da, 2)
    logical_operators(x_da, 2 * x_da)


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
    assert result.units == g ** -1
    result = x_da_2 / unyt_quantity(2, "m")
    assert result.units == g / m
    result = unyt_quantity(2, "m") / x_da_2
    assert result.units == m / g

    result = x_da ** 2
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
    x_unyt = unyt_array(ones((10, 10)), m)

    result = x_da.compute()
    assert type(result) == unyt_array
    assert_array_equal(result, x_unyt)

    result = x_da.min().compute()
    assert type(result) == unyt_quantity
    assert result == unyt_quantity(1, m)


def test_dask_passthroughs():

    # tests the simple dask functions that do not modify units
    x = dask.array.ones((10, 10), chunks=(2, 2))
    x_da = unyt_from_dask(x, m)
    assert x_da.reshape((100, 1)).units == m


def test_repr():
    x = dask.array.ones((10, 10), chunks=(2, 2))
    x_da = unyt_from_dask(x, m)
    assert "unyt_dask_array" in x_da.__repr__()
    assert "Units" in x_da._repr_html_()


def test_np_ufuncs():
    x = dask.array.ones((10, 10), chunks=(2, 2))
    x_da = unyt_from_dask(x, m)
    assert type(min(x_da).compute()) == unyt_quantity
    result = add(x_da, unyt_quantity(2, m)).compute()
    assert type(result) == unyt_array


@pytest.mark.parametrize(
    ("dask_func_str", "actual", "axis", "check_nan"),
    [
        ("mean", unyt_quantity(1., "m"), None, True),
        ("min", unyt_quantity(1., "m"), None, True),
        ("max", unyt_quantity(1., "m"), None, True),
        ("std", unyt_quantity(0., "m"), None, True),
        ("median", unyt_quantity(1., "m"), 1, True),  # median requires an axis for dask
        ("var", unyt_quantity(0., "m**2"), None, True),
        ("sum", unyt_quantity(100., "m"), None, True),
        ("cumsum", unyt_array(ones((10,10)), "m").cumsum(axis=1), 1, False), # nancumsum dask bug?
        ("average", unyt_quantity(1., "m"), None, False),
        ("diagonal", unyt_array(ones((10,)), "m"), None, False),
        ("unique", unyt_array([1.], "m"), None, False),

    ]
)
def test_dask_array_reductions(dask_func_str, actual, axis, check_nan):

    extra_kwargs = {}
    if axis:
        extra_kwargs["axis"] = axis

    func_strs = [dask_func_str, ]
    if check_nan:
        func_strs.append("nan" + dask_func_str)

    for func_str in func_strs:
        dask_func = getattr(dask.array, func_str)
        x_da = unyt_from_dask(dask.array.ones((10, 10), chunks=(2, 2)), m)

        result = reduce_with_units(dask_func, x_da, **extra_kwargs).compute()
        assert_array_equal(result, actual)