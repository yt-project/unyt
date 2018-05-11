"""
Unit system class.

"""

# -----------------------------------------------------------------------------
# Copyright (c) 2018, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------

from collections import OrderedDict
from six import string_types
from unyt import dimensions
from unyt.exceptions import (
    MissingMKSCurrent,
    IllDefinedUnitSystem
)
from unyt.unit_object import (
    Unit,
    _get_system_unit_string
)

unit_system_registry = {}

cmks = dimensions.current_mks


class UnitSystem(object):
    """
    Create a UnitSystem for facilitating conversions to a default set of units.

    Parameters
    ----------
    name : string
        The name of the unit system. Will be used as the key in the
        *unit_system_registry* dict to reference the unit system by.
    length_unit : string
        The base length unit of this unit system.
    mass_unit : string
        The base mass unit of this unit system.
    time_unit : string
        The base time unit of this unit system.
    temperature_unit : string, optional
        The base temperature unit of this unit system. Defaults to "K".
    angle_unit : string, optional
        The base angle unit of this unit system. Defaults to "rad".
    mks_system: boolean, optional
        Whether or not this unit system has SI-specific units.
        Default: False
    current_mks_unit : string, optional
        The base current unit of this unit system. Defaults to "A".
    luminous_intensity_unit : string, optional
        The base luminous intensity unit of this unit system.
        Defaults to "cd".
    registry : :class:`yt.units.unit_registry.UnitRegistry` object
        The unit registry associated with this unit system. Only
        useful for defining unit systems based on code units.
    """
    def __init__(self, name, length_unit, mass_unit, time_unit,
                 temperature_unit="K", angle_unit="rad",
                 current_mks_unit="A", luminous_intensity_unit="cd",
                 registry=None):
        self.registry = registry
        if current_mks_unit is not None:
            current_mks_unit = Unit(current_mks_unit,
                                    registry=self.registry)
        self.units_map = OrderedDict([
            (dimensions.length, Unit(length_unit, registry=self.registry)),
            (dimensions.mass, Unit(mass_unit, registry=self.registry)),
            (dimensions.time, Unit(time_unit, registry=self.registry)),
            (dimensions.temperature, Unit(
                temperature_unit, registry=self.registry)),
            (dimensions.angle, Unit(angle_unit, registry=self.registry)),
            (dimensions.current_mks, current_mks_unit),
            (dimensions.luminous_intensity, Unit(
                luminous_intensity_unit, registry=self.registry))])
        for dimension, unit in self.units_map.items():
            # CGS sets the current_mks unit to none, so catch it here
            if unit is None:
                continue
            if unit.dimensions is not dimension:
                raise IllDefinedUnitSystem(self.units_map)
        self._dims = ["length", "mass", "time", "temperature", "angle",
                      "current_mks", "luminous_intensity"]
        self.registry = registry
        self.base_units = self.units_map.copy()
        unit_system_registry[name] = self
        self.name = name

    def __getitem__(self, key):
        if isinstance(key, string_types):
            key = getattr(dimensions, key)
        um = self.units_map
        if key not in um or um[key] is None or um[key].dimensions is not key:
            if cmks in key.free_symbols and self.units_map[cmks] is None:
                raise MissingMKSCurrent(self.name)
            units = _get_system_unit_string(key, self.units_map)
            self.units_map[key] = Unit(units, registry=self.registry)
        return self.units_map[key]

    def __setitem__(self, key, value):
        if isinstance(key, string_types):
            if key not in self._dims:
                self._dims.append(key)
            key = getattr(dimensions, key)
        if self.units_map[cmks] is None and cmks in key.free_symbols:
            raise MissingMKSCurrent(self.name)
        self.units_map[key] = Unit(value, registry=self.registry)

    def __str__(self):
        return self.name

    def __repr__(self):
        repr = "%s Unit System\n" % self.name
        repr += " Base Units:\n"
        for dim in self.base_units:
            if self.base_units[dim] is not None:
                repr += "  %s: %s\n" % (
                    str(dim).strip("()"), self.base_units[dim])
        repr += " Other Units:\n"
        for key in self._dims:
            dim = getattr(dimensions, key)
            if dim not in self.base_units:
                repr += "  %s: %s\n" % (key, self.units_map[dim])
        return repr[:-1]


#: The CGS unit system
cgs_unit_system = UnitSystem("cgs", "cm", "g", "s", current_mks_unit=None)
cgs_unit_system["energy"] = "erg"
cgs_unit_system["specific_energy"] = "erg/g"
cgs_unit_system["pressure"] = "dyne/cm**2"
cgs_unit_system["force"] = "dyne"
cgs_unit_system["magnetic_field_cgs"] = "gauss"
cgs_unit_system["charge_cgs"] = "esu"
cgs_unit_system["current_cgs"] = "statA"
cgs_unit_system["power"] = "erg/s"

#: The MKS unit system
mks_unit_system = UnitSystem("mks", "m", "kg", "s")
mks_unit_system["energy"] = "J"
mks_unit_system["specific_energy"] = "J/kg"
mks_unit_system["pressure"] = "Pa"
mks_unit_system["force"] = "N"
mks_unit_system["magnetic_field_mks"] = "T"
mks_unit_system["charge_mks"] = "C"
mks_unit_system["power"] = "W"

#: The imperial unit system
imperial_unit_system = UnitSystem("imperial", "ft", "lb", "s",
                                  temperature_unit="R")
imperial_unit_system["force"] = "lbf"
imperial_unit_system["energy"] = "ft*lbf"
imperial_unit_system["pressure"] = "lbf/ft**2"
imperial_unit_system["power"] = "hp"

#: The galactic unit system
galactic_unit_system = UnitSystem("galactic", "kpc", "Msun", "Myr")
galactic_unit_system["energy"] = "keV"
galactic_unit_system["magnetic_field_cgs"] = "uG"

#: The solar unit system
solar_unit_system = UnitSystem("solar", "AU", "Mearth", "yr")

#: Geometrized unit system
geometrized_unit_system = UnitSystem("geometrized", "l_geom",
                                     "m_geom", "t_geom")

#: Planck unit system
planck_unit_system = UnitSystem("planck", "l_pl", "m_pl", "t_pl",
                                temperature_unit="T_pl")
planck_unit_system["energy"] = "E_pl"
planck_unit_system["charge_mks"] = "q_pl"
