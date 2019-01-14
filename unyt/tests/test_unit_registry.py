"""
Test unit lookup tables and registry




"""

# -----------------------------------------------------------------------------
# Copyright (c) 2018, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------

import pytest

from unyt.dimensions import length
from unyt.exceptions import SymbolNotFoundError, UnitParseError
from unyt.unit_object import Unit
from unyt.unit_registry import UnitRegistry


def test_add_modify_error():
    from unyt import m

    ureg = UnitRegistry()

    with pytest.raises(UnitParseError):
        ureg.add("tayne", 1, length)
    with pytest.raises(UnitParseError):
        ureg.add("tayne", "blah", length)
    with pytest.raises(UnitParseError):
        ureg.add("tayne", 1.0, length, offset=1)
    with pytest.raises(UnitParseError):
        ureg.add("tayne", 1.0, length, offset="blah")

    ureg.add("tayne", 1.1, length)

    with pytest.raises(SymbolNotFoundError):
        ureg.remove("tayn")
    with pytest.raises(SymbolNotFoundError):
        ureg.modify("tayn", 1.2)

    ureg.modify("tayne", 1.0 * m)

    assert ureg["tayne"][:3] == ureg["m"][:3]


def test_keys():
    ureg = UnitRegistry()
    assert sorted(ureg.keys()) == sorted(ureg.lut.keys())


def test_prefixable_units():
    ureg = UnitRegistry()
    pu = ureg.prefixable_units
    assert "m" in pu
    assert "pc" in pu
    assert "mol" in pu
    ureg.add("foobar", 1.0, length, prefixable=True)
    assert "foobar" in ureg.prefixable_units
    mfoobar = Unit("mfoobar", registry=ureg)
    foobar = Unit("foobar", registry=ureg)
    assert (1 * foobar) / (1 * mfoobar) == 1000


def test_registry_contains():
    ureg = UnitRegistry()
    assert "m" in ureg
    assert "cm" in ureg
    assert "erg" in ureg
    assert "Merg" in ureg
    assert "foobar" not in ureg
    assert Unit("m", registry=ureg) in ureg
