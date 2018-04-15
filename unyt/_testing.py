"""
Utilities for writing tests

"""

# -----------------------------------------------------------------------------
# Copyright (c) 2018, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------

from numpy.testing import assert_allclose

from unyt.array import (
    unyt_array,
    unyt_quantity
)
from unyt.exceptions import (
    UnitConversionError,
    UnitOperationError,
)


def assert_allclose_units(actual, desired, rtol=1e-7, atol=0, **kwargs):
    """Raise an error if two objects are not equal up to desired tolerance

    This is a wrapper for :func:`numpy.testing.assert_allclose` that also
    verifies unit consistency

    Parameters
    ----------
    actual : array-like
        Array obtained (possibly with attached units)
    desired : array-like
        Array to compare with (possibly with attached units)
    rtol : float, optional
        Relative tolerance, defaults to 1e-7
    atol : float or quantity, optional
        Absolute tolerance. If units are attached, they must be consistent
        with the units of ``actual`` and ``desired``. If no units are attached,
        assumes the same units as ``desired``. Defaults to zero.

    Notes
    -----
    Also accepts additional keyword arguments accepted by
    :func:`numpy.testing.assert_allclose`, see the documentation of that
    function for details.

    """
    # Create a copy to ensure this function does not alter input arrays
    act = unyt_array(actual)
    des = unyt_array(desired)

    try:
        des = des.in_units(act.units)
    except (UnitOperationError, UnitConversionError):
        raise AssertionError(
            "Units of actual (%s) and desired (%s) do not have "
            "equivalent dimensions" % (act.units, des.units))

    rt = unyt_array(rtol)
    if not rt.units.is_dimensionless:
        raise AssertionError("Units of rtol (%s) are not "
                             "dimensionless" % rt.units)

    if not isinstance(atol, unyt_array):
        at = unyt_quantity(atol, des.units)

    try:
        at = at.in_units(act.units)
    except UnitOperationError:
        raise AssertionError("Units of atol (%s) and actual (%s) do not have "
                             "equivalent dimensions" % (at.units, act.units))

    # units have been validated, so we strip units before calling numpy
    # to avoid spurious errors
    act = act.value
    des = des.value
    rt = rt.value
    at = at.value

    return assert_allclose(act, des, rt, at, **kwargs)
