"""Test Matplotlib ConversionInterface"""
import numpy as np
import pytest
from unyt._on_demand_imports import _matplotlib, NotAModule
from unyt import s, K
from unyt.exceptions import UnitConversionError


check_matplotlib = pytest.mark.skipif(
    isinstance(_matplotlib.pyplot, NotAModule), reason="matplotlib not installed"
)


@pytest.fixture
def ax(scope="module"):
    fig, ax = _matplotlib.pyplot.subplots()
    yield ax
    _matplotlib.pyplot.close()


@check_matplotlib
def test_label(ax):
    x = [0, 1, 2] * s
    y = [3, 4, 5] * K
    ax.plot(x, y)
    expected_xlabel = "$\\left(\\rm{s}\\right)$"
    assert ax.xaxis.get_label().get_text() == expected_xlabel
    expected_ylabel = "$\\left(\\rm{K}\\right)$"
    assert ax.yaxis.get_label().get_text() == expected_ylabel


@check_matplotlib
def test_convert_unit(ax):
    x = [0, 1, 2] * s
    y = [1000, 2000, 3000] * K
    ax.plot(x, y, yunits="Celcius")
    expected = y.to("Celcius")
    line = ax.lines[0]
    original_y_array = line.get_data()[1]
    converted_y_array = line.convert_yunits(original_y_array)
    results = converted_y_array == expected
    assert results.all()


@check_matplotlib
def test_convert_equivalency(ax):
    x = [0, 1, 2] * s
    y = [1000, 2000, 3000] * K
    ax.clear()
    ax.plot(x, y, yunits=("J", "thermal"))
    expected = y.to("J", "thermal")
    line = ax.lines[0]
    original_y_array = line.get_data()[1]
    converted_y_array = line.convert_yunits(original_y_array)
    results = converted_y_array == expected
    assert results.all()


@check_matplotlib
def test_dimensionless(ax):
    x = [0, 1, 2] * s
    y = [3, 4, 5] * K / K
    ax.plot(x, y)
    expected_ylabel = ""
    assert ax.yaxis.get_label().get_text() == expected_ylabel


@check_matplotlib
def test_conversionerror(ax):
    x = [0, 1, 2] * s
    y = [3, 4, 5] * K
    ax.plot(x, y)
    ax.xaxis.callbacks.exception_handler = None
    # Newer matplotlib versions catch our exception and raise a custom
    # ConversionError exception
    try:
        error_type = _matplotlib.units.ConversionError
    except AttributeError:
        error_type = UnitConversionError
    with pytest.raises(error_type):
        ax.xaxis.set_units("V")


@check_matplotlib
def test_ndarray_label(ax):
    x = [0, 1, 2] * s
    y = np.arange(3, 6)
    ax.plot(x, y)
    expected_xlabel = "$\\left(\\rm{s}\\right)$"
    assert ax.xaxis.get_label().get_text() == expected_xlabel
    expected_ylabel = ""
    assert ax.yaxis.get_label().get_text() == expected_ylabel


@check_matplotlib
def test_list_label(ax):
    x = [0, 1, 2] * s
    y = [3, 4, 5]
    ax.plot(x, y)
    expected_xlabel = "$\\left(\\rm{s}\\right)$"
    assert ax.xaxis.get_label().get_text() == expected_xlabel
    expected_ylabel = ""
    assert ax.yaxis.get_label().get_text() == expected_ylabel
