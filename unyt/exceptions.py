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
        err = ("Cannot convert between %s (dim %s) and %s (dim %s)." %
               (self.unit1, self.dimension1, self.unit2, self.dimension2))
        return err


class MissingMKSCurrent(Exception):
    def __init__(self, unit_system_name):
        self.unit_system_name = unit_system_name

    def __str__(self):
        err = ("The %s unit system does not have a MKS current base unit" %
               self.unit_system_name)
        return err


class MKSCGSConversionError(Exception):
    def __init__(self, unit):
        self.unit = unit

    def __str__(self):
        err = ("The %s unit cannot be safely converted." % self.unit)
        return err


class UnitsNotReducible(Exception):
    def __init__(self, unit, units_base):
        self.unit = unit
        self.units_base = units_base
        Exception.__init__(self)

    def __str__(self):
        err = ("The unit \"%s\" (dimensions \"%s\" cannot be reduced to an "
               "expression within the %s system of units." %
               (self.unit, self.unit.dimensions, self.units_base))
        return err


class EquivalentDimsError(UnitOperationError):
    def __init__(self, old_units, new_units, base):
        self.old_units = old_units
        self.new_units = new_units
        self.base = base

    def __str__(self):
        err = ("It looks like you're trying to convert between \"%s\" and "
               "\"%s\". Try using \"to_equivalent('%s', '%s')\" instead." %
               (self.old_units, self.new_units, self.new_units, self.base))
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
        from unyt.unit_object import Unit
        if isinstance(self.unit2, Unit):
            msg = ("The unit equivalence '%s' does not exist for the units "
                   "'%s' and '%s'.")
        else:
            msg = ("The unit equivalence '%s' does not exist for units '%s' "
                   "to convert to a new unit with dimensions '%s'.")
        return msg % (self.equiv, self.unit1, self.unit2)


class InvalidUnitOperation(Exception):
    pass


class SymbolNotFoundError(Exception):
    pass


class UnitParseError(Exception):
    pass


class IllDefinedUnitSystem(Exception):
    def __init__(self, units_map):
        self.units_map = units_map

    def __str__(self):
        return ("Cannot create unit system with inconsistent mapping from "
                "dimensions to units. Received:\n%s" % self.units_map)
