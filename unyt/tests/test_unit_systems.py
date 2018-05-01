"""
Test unit systems.

"""

# -----------------------------------------------------------------------------
# Copyright (c) 2018, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------

import pytest

from unyt.exceptions import (
    EquivalentDimsError,
    IllDefinedUnitSystem,
)
from unyt.unit_object import Unit
from unyt.unit_systems import (
    UnitSystem,
    unit_system_registry,
)
from unyt.unit_registry import UnitRegistry
from unyt import dimensions
from unyt import (
    gauss,
    Tesla,
)


def test_unit_systems():
    goofy_unit_system = UnitSystem(
        "goofy", "ly", "lbm", "hr", temperature_unit="R",
        angle_unit="arcsec", current_mks_unit="mA",
        luminous_intensity_unit="cd")
    assert goofy_unit_system["temperature"] == Unit("R")
    assert goofy_unit_system[dimensions.solid_angle] == Unit("arcsec**2")
    assert goofy_unit_system["energy"] == Unit("lbm*ly**2/hr**2")
    goofy_unit_system["energy"] = "eV"
    assert goofy_unit_system["energy"] == Unit("eV")
    assert goofy_unit_system["magnetic_field_mks"] == Unit("lbm/(hr**2*mA)")
    assert "goofy" in unit_system_registry


def test_unit_system_id():
    reg1 = UnitRegistry()
    reg2 = UnitRegistry()
    assert reg1.unit_system_id == reg2.unit_system_id
    reg1.modify('g', 2.0)
    assert reg1.unit_system_id != reg2.unit_system_id
    reg1 = UnitRegistry()
    reg1.add('dinosaurs', 12.0, dimensions.length)
    assert reg1.unit_system_id != reg2.unit_system_id
    reg1 = UnitRegistry()
    reg1.remove('g')
    assert reg1.unit_system_id != reg2.unit_system_id
    reg1.add('g', 1.0e-3, dimensions.mass)
    assert reg1.unit_system_id == reg2.unit_system_id


def test_cgs_mks_unit_conversions():
    t = 1*Tesla
    g = 1*gauss
    assert t.to_equivalent('G', 'cgs') == 1e4*gauss
    assert t.to_equivalent('G', 'CGS') == 1e4*gauss
    assert g.to_equivalent("T", "mks") == 1e-4*Tesla
    assert g.to_equivalent("T", "MKS") == 1e-4*Tesla
    assert g.to_equivalent("T", "si") == 1e-4*Tesla
    assert g.to_equivalent("T", "SI") == 1e-4*Tesla


def test_bad_unit_system():
    with pytest.raises(IllDefinedUnitSystem):
        UnitSystem('atomic', 'nm', 'fs', 'nK', 'rad')
