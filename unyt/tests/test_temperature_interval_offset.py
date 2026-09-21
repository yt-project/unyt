import pytest

from unyt import K, degC, degF, delta_degC, delta_degF
from unyt.exceptions import InvalidUnitOperation


def test_interval_to_offset_raises():
    with pytest.raises(InvalidUnitOperation):
        (3 * delta_degC).to(degC)
    with pytest.raises(InvalidUnitOperation):
        (3 * degC).to(delta_degC)
    with pytest.raises(InvalidUnitOperation):
        (3 * delta_degF).to(degF)
    with pytest.raises(InvalidUnitOperation):
        (3 * degF).to(delta_degF)


def test_kelvin_celsius_affine_stays():
    assert (0 * degC).to(K) == 273.15 * K
    assert (273.15 * K).to(degC) == 0 * degC


def test_interval_to_kelvin_stays():
    assert (3 * delta_degC).to(K) == 3 * K
