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
from unyt.exceptions import (
    SymbolNotFoundError,
    UnitParseError,
)
from unyt.unit_registry import UnitRegistry


def test_add_modify_error():
    from unyt import m
    ureg = UnitRegistry()

    with pytest.raises(UnitParseError):
        ureg.add('tayne', 1, length)
    with pytest.raises(UnitParseError):
        ureg.add('tayne', 'blah', length)
    with pytest.raises(UnitParseError):
        ureg.add('tayne', 1.0, length, offset=1)
    with pytest.raises(UnitParseError):
        ureg.add('tayne', 1.0, length, offset='blah')

    ureg.add('tayne', 1.1, length)

    with pytest.raises(SymbolNotFoundError):
        ureg.remove('tayn')
    with pytest.raises(SymbolNotFoundError):
        ureg.modify('tayn', 1.2)

    ureg.modify('tayne', 1.0*m)

    assert ureg['tayne'][:3] == ureg['m'][:3]


def test_keys():
    ureg = UnitRegistry()
    assert sorted(ureg.keys()) == sorted(ureg.lut.keys())
