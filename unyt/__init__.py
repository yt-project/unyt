"""
The unyt package. See the unyt documentation for full details.


"""

# -----------------------------------------------------------------------------
# Copyright (c) 2018, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------

from unyt import unit_symbols
from unyt import physical_constants

from unyt.array import (  # NOQA
    loadtxt,
    savetxt,
    uconcatenate,
    ucross,
    udot,
    uhstack,
    uintersect1d,
    unorm,
    ustack,
    uunion1d,
    uvstack,
    unyt_array,
    unyt_quantity
)


# function to only import quantities into this namespace
# we go through the trouble of doing this instead of "import *"
# to avoid including extraneous variables (e.g. floating point
# constants used to *construct* a physical constant) in this namespace
def import_quantities(module, global_namespace):
    for key, value in module.__dict__.items():
        if isinstance(value, unyt_quantity):
            global_namespace[key] = value


import_quantities(unit_symbols, globals())
import_quantities(physical_constants, globals())
