"""
The unyt package.

Note that the symbols defined in :mod:`unyt.physical_constants` and
:mod:`unyt.unit_symbols` are importable from this module. For example::

    >>> from unyt import km, clight
    >>> print((km/clight).to('ns'))
    3335.64095198152 ns

In addition, the following functions and classes are importable from the
top-level ``unyt`` namespace:

* :func:`unyt.array.loadtxt`
* :func:`unyt.array.savetxt`
* :func:`unyt.test`
* :func:`unyt.array.uconcatenate`
* :func:`unyt.array.ucross`
* :func:`unyt.array.udot`
* :func:`unyt.array.uhstack`
* :func:`unyt.array.uintersect1d`
* :func:`unyt.array.unorm`
* :func:`unyt.array.ustack`
* :func:`unyt.array.uunion1d`
* :func:`unyt.array.uvstack`
* :class:`unyt.array.unyt_array`
* :class:`unyt.array.unyt_quantity`
* :func:`unyt.unit_object.define_unit`
* :class:`unyt.unit_object.Unit`
* :class:`unyt.unit_registry.UnitRegistry`
* :class:`unyt.unit_systems.UnitSystem`
* :func:`unyt.testing.assert_allclose_units`
* :func:`unyt.array.allclose_units`
* :func:`unyt.dimensions.accepts`
* :func:`unyt.dimensions.returns`
"""

# -----------------------------------------------------------------------------
# Copyright (c) 2018, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------

from unyt import physical_constants, unit_symbols
from unyt.array import (  # NOQA: F401
    allclose_units,
    loadtxt,
    savetxt,
    uconcatenate,
    ucross,
    udot,
    uhstack,
    uintersect1d,
    unorm,
    unyt_array,
    unyt_quantity,
    ustack,
    uunion1d,
    uvstack,
)
from unyt.dimensions import accepts, returns  # NOQA: F401
from unyt.testing import assert_allclose_units  # NOQA: F401
from unyt.unit_object import Unit, define_unit  # NOQA: F401
from unyt.unit_registry import UnitRegistry  # NOQA: F401
from unyt.unit_systems import UnitSystem  # NOQA: F401

from ._version import get_versions

try:
    from unyt.mpl_interface import matplotlib_support  # NOQA: F401
except ImportError:
    pass
else:
    matplotlib_support = matplotlib_support()


# function to only import quantities into this namespace
# we go through the trouble of doing this instead of "import *"
# to avoid including extraneous variables (e.g. floating point
# constants used to *construct* a physical constant) in this namespace
def import_units(module, namespace):
    """Import Unit objects from a module into a namespace"""
    for key, value in module.__dict__.items():
        if isinstance(value, (unyt_quantity, Unit)):
            namespace[key] = value


import_units(unit_symbols, globals())
import_units(physical_constants, globals())

del import_units

__version__ = get_versions()["version"]
del get_versions


def test():  # pragma: no cover
    """Execute the unit tests on an installed copy of unyt.

    Note that this function requires pytest to run. If pytest is not
    installed this function will raise ImportError.
    """
    import os

    import pytest

    pytest.main([os.path.dirname(os.path.abspath(__file__))])
