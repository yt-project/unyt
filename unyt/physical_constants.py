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
    unyt_quantity(365.26236846, 'day')

.. show_all_constants::

"""


from itertools import chain as _chain

from unyt._unit_lookup_table import physical_constants as _physical_constants
from unyt.array import unyt_quantity
from unyt.unit_registry import default_unit_registry as _default_unit_registry

_mks_csts = dict.fromkeys(_physical_constants.keys())
_alt2csts = dict.fromkeys(
    _chain.from_iterable(
        alt_names for (_, _, alt_names) in _physical_constants.values()
    )
)


for constant_name, (value, unit_name, alternate_names) in _physical_constants.items():
    _alt2csts[constant_name] = constant_name
    _alt2csts.update({alt: constant_name for alt in alternate_names})
    _mks_csts[constant_name] = unyt_quantity(
        value, unit_name, registry=_default_unit_registry
    )


def __getattr__(name):
    from unyt.exceptions import UnitsNotReducible

    if name in _mks_csts:
        try:
            return _mks_csts[name].in_base(_default_unit_registry.unit_system)
        except UnitsNotReducible:
            return _mks_csts[name]

    if name.endswith("_mks") and name[:-4] in _alt2csts:
        return _mks_csts[_alt2csts[name[:-4]]]
    elif name.endswith("_cgs") and name[:-4] in _alt2csts:
        try:
            return _mks_csts[_alt2csts[name[:-4]]].in_cgs()
        except UnitsNotReducible:
            pass
    elif name in _alt2csts:
        return _mks_csts[_alt2csts[name]]
    elif name == "hmks":
        # backward compatibility for unyt 1.0
        return _mks_csts["h"]
    elif name == "hcgs":
        # backward compatibility for unyt 1.0
        return _mks_csts["h"].in_cgs()

    raise AttributeError
