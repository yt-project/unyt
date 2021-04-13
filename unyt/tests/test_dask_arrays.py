from numpy.testing import  assert_array_equal
from numpy import sqrt, ones
from unyt.dask_array import unyt_from_dask, unyt_dask_array
import dask.array as dask_array
from unyt import unyt_array, unyt_quantity

from unyt.unit_symbols import cm, m, g, degree


def test_unyt_dask_creation():
    x = dask_array.ones((10, 10), chunks=(2, 2))
    x_da = unyt_from_dask(x, m)
    assert (type(x_da) == unyt_dask_array)
    assert (x_da.units == m)
    assert (type(x_da.compute()) == unyt_array)


def test_unit_conversions():
    x = dask_array.ones((10, 10), chunks=(2, 2))
    x_da = unyt_from_dask(x, m)
    x_da = x_da.to(cm)
    assert (type(x_da) == unyt_dask_array)
    assert (x_da.units == cm)
    assert (x_da.compute().units == cm)

    x_da_2 = unyt_from_dask(x, cm)
    result = x_da + x_da_2
    assert (type(result) == unyt_dask_array)
    assert (result.units == cm)
    assert (result.compute().units == cm)


def test_conversion_to_dask():
    x = dask_array.ones((10, 10), chunks=(2, 2))
    x_da = unyt_from_dask(x, m)
    x_again = x_da.to_dask()
    assert_array_equal(x.compute(), x_again.compute())
    assert (type(x_again) == type(x))


def unary_test(the_func, unyt_dask_obj, unyt_array_in, *args, **kwargs):
    # the_func can be a numpy ufunc or dask.array numpy ufunc implementation

    result_delay = the_func(unyt_dask_obj, *args, **kwargs)
    correct_unyt = the_func(unyt_array_in, *args, **kwargs)

    assert (result_delay.units == correct_unyt.units)  # units should already match
    unary_result_test(result_delay, correct_unyt)


def unary_result_test(dask_unyt_delayed, correct_unyt):
    # computes a delayed dask_unyt_array and compares resulting units and values
    result = dask_unyt_delayed.compute()
    assert (result.units == correct_unyt.units)
    assert (type(result) == type(correct_unyt))
    # value comparison:
    if type(result) == unyt_array:
        assert_array_equal(result, correct_unyt)
    else:
        assert(result == correct_unyt)


def test_unary():

    x = dask_array.ones((10, 10), chunks=(2, 2))
    x_da = unyt_from_dask(x, m)
    x_unyt = unyt_array(ones((10, 10)), m)

    for unary_op in [sqrt, dask_array.sqrt]:
        unary_test(unary_op, x_da, x_unyt)

    # some of the daskified ufunc attributes need to be called directly
    unary_result_test(x_da.sum(), x_unyt.sum())
    unary_result_test(x_da.min(), x_unyt.min())
    unary_result_test(x_da.max(), x_unyt.max())
    unary_result_test(x_da.mean(), x_unyt.mean())
    unary_result_test(x_da.std(), x_unyt.std())
    unary_result_test(x_da.cumsum(0), x_unyt.cumsum(0))

def test_binary():
    x = dask_array.ones((10, 10), chunks=(2, 2))
    x2 = dask_array.full((10, 10), 2, chunks=(2, 2))
    x_da = unyt_from_dask(x, m)
    x_da_2 = unyt_from_dask(x2, g)

    # multiplications
    result = x_da * x_da_2
    assert (result.units == m * g)
    result = x_da_2 * x_da
    assert (result.units == m * g)
    result = x_da_2 * 2
    assert (result.units == g)
    result = 2 * x_da_2
    assert (result.units == g)
    result = x_da_2 * unyt_quantity(2, 'm')
    assert (result.units == m*g)
    result = unyt_quantity(2, 'm') * x_da_2
    assert (result.units == m*g)

    # divisions
    result = x_da_2 / x_da
    assert (result.units == g / m)
    result = x_da / x_da_2
    assert (result.units == m / g)
    result = x_da_2 / 2
    assert (result.units == g)
    result = 2 / x_da_2  # __rtruediv__
    assert (result.units == g ** -1)
    result = x_da_2 / unyt_quantity(2, 'm')
    assert (result.units == g / m)
    result = unyt_quantity(2, 'm') / x_da_2
    assert (result.units == m / g)

    result = x_da ** 2
    assert (result.units == m*m)

    x_da_2 = unyt_from_dask(x2, m)
    result = x_da + x_da_2
    assert (result.units == m)
    result = x_da_2 + x_da
    assert (result.units == m)
    result = x_da + unyt_quantity(1, 'm')
    assert (result.units == m)
    result = unyt_quantity(1, 'm') + x_da
    assert (result.units == m)

    result = x_da - x_da_2
    assert (result.units == m)
    result = x_da_2 - x_da
    assert (result.units == m)
    result = x_da - unyt_quantity(1, 'm')
    assert (result.units == m)
    assert (result.min().compute() == unyt_quantity(0, 'm'))
    result = unyt_quantity(1, 'm') - x_da
    assert (result.units == m)


def test_unyt_type_result():
    # test that the return type of a compute is unyt_array or unyt_quantity when
    # appropriate

    x = dask_array.ones((10, 10), chunks=(2, 2))
    x_da = unyt_from_dask(x, m)
    x_unyt = unyt_array(ones((10, 10)), m)

    result = x_da.compute()
    assert (type(result) == unyt_array)
    assert_array_equal(result, x_unyt)

    result = x_da.min().compute()
    assert (type(result) == unyt_quantity)
    assert (result == unyt_quantity(1, m))


def test_dask_passthroughs():

    # tests the simple dask functions that do not modify units
    x = dask_array.ones((10, 10), chunks=(2, 2))
    x_da = unyt_from_dask(x, m)
    assert(x_da.reshape((100, 1)).units == m)




