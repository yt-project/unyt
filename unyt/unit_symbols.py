# -*- coding: utf-8 -*-
"""
Predefined useful aliases to physical units

Note that all of these names can be imported from the top-level unyt namespace.
For example::

    >>> from unyt import cm, g, s
    >>> data = [3, 4, 5]*g*cm/s
    >>> print(data)
    [3 4 5] cm*g/s

.. show_all_units::

"""

# -----------------------------------------------------------------------------
# Copyright (c) 2018, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------

from unyt._unit_lookup_table import name_alternatives as _name_alternatives
from unyt.unit_object import Unit as _Unit
from unyt.unit_registry import default_unit_registry as _registry

_ns = globals()
for _canonical_name, _alt_names in _name_alternatives.items():
    _u = _Unit(_canonical_name, registry=_registry)
    for _alt_name in _alt_names:
        _ns[_alt_name] = _u
