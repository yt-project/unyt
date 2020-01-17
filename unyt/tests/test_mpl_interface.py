"""Test Matplotlib ConversionInterface"""
from unyt._on_demand_imports import _matplotlib, NotAModule
from unyt import s, K


def test_label():
    if isinstance(_matplotlib.pyplot, NotAModule):
        return
    plt = _matplotlib.pyplot
    x = [0, 1, 2] * s
    y = [3, 4, 5] * K
    _, ax = plt.subplots()
    ax.plot(x, y)
    expected_xlabel = "$\\left(\\rm{s}\\right)$"
    assert ax.xaxis.get_label().get_text() == expected_xlabel
    expected_ylabel = "$\\left(\\rm{K}\\right)$"
    assert ax.yaxis.get_label().get_text() == expected_ylabel


def test_convert_unit():
    if isinstance(_matplotlib.pyplot, NotAModule):
        return
    plt = _matplotlib.pyplot
    x = [0, 1, 2] * s
    y = [1000, 2000, 3000] * K
    _, ax = plt.subplots()
    ax.plot(x, y, yunits="Celcius")
    expected = y.to("Celcius")
    line = ax.lines[0]
    original_y_array = line.get_data()[1]
    converted_y_array = line.convert_yunits(original_y_array)
    results = converted_y_array == expected
    assert results.all()


def test_convert_equivalency():
    if isinstance(_matplotlib.pyplot, NotAModule):
        return
    plt = _matplotlib.pyplot
    x = [0, 1, 2] * s
    y = [1000, 2000, 3000] * K
    _, ax = plt.subplots()
    ax.plot(x, y, yunits=("J", "thermal"))
    expected = y.to("J", "thermal")
    line = ax.lines[0]
    original_y_array = line.get_data()[1]
    converted_y_array = line.convert_yunits(original_y_array)
    results = converted_y_array == expected
    assert results.all()
