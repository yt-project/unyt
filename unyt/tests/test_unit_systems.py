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

from unyt.unit_object import Unit, unit_system_registry
from unyt.unit_systems import UnitSystem
from unyt import dimensions


def test_unit_systems():
    goofy_unit_system = UnitSystem(
        "goofy", "ly", "lbm", "hr", temperature_unit="R",
        angle_unit="arcsec", current_mks_unit="mA")
    assert goofy_unit_system["temperature"] == Unit("R")
    assert goofy_unit_system[dimensions.solid_angle] == Unit("arcsec**2")
    assert goofy_unit_system["energy"] == Unit("lbm*ly**2/hr**2")
    goofy_unit_system["energy"] = "eV"
    assert goofy_unit_system["energy"] == Unit("eV")
    assert goofy_unit_system["magnetic_field_mks"] == Unit("lbm/(hr**2*mA)")
    assert "goofy" in unit_system_registry
