"""
Exception classes defined by unyt



"""

# -----------------------------------------------------------------------------
# Copyright (c) 2018, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------


# Data access exceptions:

class UnitOperationError(ValueError):
    def __init__(self, operation, unit1, unit2=None):
        self.operation = operation
        self.unit1 = unit1
        self.unit2 = unit2
        ValueError.__init__(self)

    def __str__(self):
        err = ("The %s operator for unyt_arrays with units \"%s\" "
               "(dimensions \"%s\") " %
               (self.operation, self.unit1, self.unit1.dimensions))
        if self.unit2 is not None:
            err += ("and \"%s\" (dimensions \"%s\") " %
                    (self.unit2, self.unit2.dimensions))
        err += "is not well defined."
        return err


class UnitConversionError(Exception):
    def __init__(self, unit1, dimension1, unit2, dimension2):
        self.unit1 = unit1
        self.unit2 = unit2
        self.dimension1 = dimension1
        self.dimension2 = dimension2
        Exception.__init__(self)

    def __str__(self):
        err = ("Unit dimensionalities do not match. Tried to convert between "
               "%s (dim %s) and %s (dim %s)." %
               (self.unit1, self.dimension1, self.unit2, self.dimension2))
        return err


class UnitsNotReducible(Exception):
    def __init__(self, unit, units_base):
        self.unit = unit
        self.units_base = units_base
        Exception.__init__(self)

    def __str__(self):
        err = ("The unit '%s' cannot be reduced to a single expression within "
               "the %s base system of units." % (self.unit, self.units_base))
        return err


class EquivalentDimsError(UnitOperationError):
    def __init__(self, old_units, new_units, base):
        self.old_units = old_units
        self.new_units = new_units
        self.base = base

    def __str__(self):
        err = ("It looks like you're trying to convert between '%s' and '%s'. "
               "Try using \"to_equivalent('%s', '%s')\" instead." %
               (self.old_units, self.new_units, self.new_units, self.base))
        return err


class UfuncUnitError(Exception):
    def __init__(self, ufunc, unit1, unit2):
        self.ufunc = ufunc
        self.unit1 = unit1
        self.unit2 = unit2
        Exception.__init__(self)

    def __str__(self):
        err = ("The NumPy %s operation is only allowed on objects with "
               "identical units. Convert one of the arrays to the other\'s "
               "units first. Received units (%s) and (%s)." %
               (self.ufunc, self.unit1, self.unit2))
        return err


class IterableUnitCoercionError(Exception):
    def __init__(self, quantity_list):
        self.quantity_list = quantity_list

    def __str__(self):
        err = ("Received a list or tuple of quantities with nonuniform units: "
               "%s" % self.quantity_list)
        return err


class InvalidUnitEquivalence(Exception):
    def __init__(self, equiv, unit1, unit2):
        self.equiv = equiv
        self.unit1 = unit1
        self.unit2 = unit2

    def __str__(self):
        return ("The unit equivalence '%s' does not exist for the units '%s' "
                "and '%s'." % (self.equiv, self.unit1, self.unit2))


class InvalidUnitOperation(Exception):
    pass


class SymbolNotFoundError(Exception):
    pass


class UnitParseError(Exception):
    pass
