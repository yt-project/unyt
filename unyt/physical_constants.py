"""
Predefined useful physical constants

Note that all of these names can be imported from the top-level unyt namespace.
For example::

    >>> from unyt.physical_constants import gravitational_constant, solar_mass
    >>> from unyt import AU
    >>> from math import pi
    >>>
    >>> period = 2 * pi * ((1 * AU)**3 / (gravitational_constant * solar_mass))**0.5
    >>> period.in_units('day')
    unyt_quantity(365.26893606, 'day')

.. show_all_constants::

"""

# -----------------------------------------------------------------------------
# Copyright (c) 2018, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------


from unyt.array import unyt_quantity as _unyt_quantity
from unyt.exceptions import UnitsNotReducible
from unyt._unit_lookup_table import physical_constants as _physical_constants
from unyt.unit_object import Unit as _Unit
from unyt.unit_registry import default_unit_registry as _default_unit_registry


def _generate_constants(namespace, registry):
    for constant_name in _physical_constants:
        value, unit_name, alternate_names = _physical_constants[constant_name]
        for name in alternate_names + [constant_name]:
            dim = _Unit(unit_name, registry=registry).dimensions
            quan = _unyt_quantity(value, registry.unit_system[dim], registry=registry)
            namespace[name] = quan
            namespace[name + "_mks"] = _unyt_quantity(
                value, unit_name, registry=registry
            )
            try:
                namespace[name + "_cgs"] = quan.in_cgs()
            except UnitsNotReducible:
                pass
            if name == "h":
                # backward compatibility for unyt 1.0, which defined hmks
                namespace["hmks"] = namespace["h_mks"].copy()
                namespace["hcgs"] = namespace["h_cgs"].copy()


_generate_constants(globals(), registry=_default_unit_registry)
