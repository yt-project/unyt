"""
A class that represents a unit symbol.


"""

# -----------------------------------------------------------------------------
# Copyright (c) 2018, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------


from six import text_type
from sympy import (
    Expr,
    Mul,
    Add,
    Number,
    Pow,
    Symbol,
    Integer,
    Float,
    Basic,
    Rational,
    sqrt
)
from sympy.core.numbers import One
from sympy import (
    sympify,
    latex
)
from sympy.parsing.sympy_parser import (
    parse_expr,
    auto_number,
    rationalize
)
from keyword import iskeyword as _iskeyword
from unyt.dimensions import (
    base_dimensions,
    temperature,
    dimensionless,
    angle
)
from unyt._unit_lookup_table import (
    unit_prefixes,
    prefixable_units,
    latex_prefixes,
)
from unyt.unit_registry import (
    default_unit_registry,
    UnitRegistry,
    UnitParseError
)
from unyt.exceptions import (
    InvalidUnitOperation,
    MissingMKSCurrent,
)
from unyt.equivalencies import (
    equivalence_registry,
    em_conversions,
)
import copy
import token
import numpy as np
from numbers import Number as numeric_type

sympy_one = sympify(1)

global_dict = {
    'Symbol': Symbol,
    'Integer': Integer,
    'Float': Float,
    'Rational': Rational,
    'sqrt': sqrt
}


def _auto_positive_symbol(tokens, local_dict, global_dict):
    """
    Inserts calls to ``Symbol`` for undefined variables.
    Passes in positive=True as a keyword argument.
    Adapted from sympy.sympy.parsing.sympy_parser.auto_symbol
    """
    result = []
    prevTok = (None, None)

    tokens.append((None, None))  # so zip traverses all tokens
    for tok, nextTok in zip(tokens, tokens[1:]):
        tokNum, tokVal = tok
        nextTokNum, nextTokVal = nextTok
        if tokNum == token.NAME:
            name = tokVal

            if (name in ['True', 'False', 'None']
                or _iskeyword(name)
                or name in local_dict
                # Don't convert attribute access
                or (prevTok[0] == token.OP and prevTok[1] == '.')
                # Don't convert keyword arguments
                or (prevTok[0] == token.OP and prevTok[1] in ('(', ',')
                    and nextTokNum == token.OP and nextTokVal == '=')):
                result.append((token.NAME, name))
                continue
            elif name in global_dict:
                obj = global_dict[name]
                if isinstance(obj, (Basic, type)) or callable(obj):
                    result.append((token.NAME, name))
                    continue

            result.extend([
                (token.NAME, 'Symbol'),
                (token.OP, '('),
                (token.NAME, repr(str(name))),
                (token.OP, ','),
                (token.NAME, 'positive'),
                (token.OP, '='),
                (token.NAME, 'True'),
                (token.OP, ')'),
            ])
        else:
            result.append((tokNum, tokVal))

        prevTok = (tokNum, tokVal)

    return result


def _get_latex_representation(expr, registry):
    symbol_table = {}
    for ex in expr.free_symbols:
        try:
            symbol_table[ex] = registry.lut[str(ex)][3]
        except KeyError:
            symbol_table[ex] = r"\rm{" + str(ex).replace('_', '\ ') + "}"

    # invert the symbol table dict to look for keys with identical values
    invert_symbols = {}
    for key, value in symbol_table.items():
        if value not in invert_symbols:
            invert_symbols[value] = [key]
        else:
            invert_symbols[value].append(key)

    # if there are any units with identical latex representations, substitute
    # units to avoid  uncanceled terms in the final latex expression.
    for val in invert_symbols:
        symbols = invert_symbols[val]
        for i in range(1, len(symbols)):
            expr = expr.subs(symbols[i], symbols[0])
    prefix = None
    if isinstance(expr, Mul):
        coeffs = expr.as_coeff_Mul()
        if coeffs[0] == 1 or not isinstance(coeffs[0], Float):
            pass
        else:
            expr = coeffs[1]
            prefix = Float(coeffs[0], 2)
    latex_repr = latex(expr, symbol_names=symbol_table, mul_symbol="dot",
                       fold_frac_powers=True, fold_short_frac=True)

    if prefix is not None:
        latex_repr = latex(prefix, mul_symbol="times") + '\\ ' + latex_repr

    if latex_repr == '1':
        return ''
    else:
        return latex_repr


unit_text_transform = (_auto_positive_symbol, rationalize, auto_number)


class Unit(Expr):
    """
    A symbolic unit, using sympy functionality. We only add "dimensions" so
    that sympy understands relations between different units.

    """

    # Set some assumptions for sympy.
    is_positive = True    # make sqrt(m**2) --> m
    is_commutative = True
    is_number = False

    # Extra attributes
    __slots__ = ["expr", "is_atomic", "base_value", "base_offset",
                 "dimensions", "registry", "_latex_repr"]

    __array_priority__ = 3.0

    def __new__(cls, unit_expr=sympy_one, base_value=None, base_offset=0.0,
                dimensions=None, registry=None, latex_repr=None,
                **assumptions):
        """
        Create a new unit. May be an atomic unit (like a gram) or combinations
        of atomic units (like g / cm**3).

        Parameters
        ----------
        unit_expr : Unit object, sympy.core.expr.Expr object, or str
            The symbolic unit expression.
        base_value : float
            The unit's value in yt's base units.
        base_offset : float
            The offset necessary to normalize temperature units to a common
            zero point.
        dimensions : sympy.core.expr.Expr
            A sympy expression representing the dimensionality of this unit.
            It must contain only mass, length, time, temperature and angle
            symbols.
        registry : UnitRegistry object
            The unit registry we use to interpret unit symbols.
        latex_repr : string
            A string to render the unit as LaTeX

        Additional keyword arguments are passed as assumptions to the Sympy
        Expr initializer

        """
        # Simplest case. If user passes a Unit object, just use the expr.
        unit_key = None
        if isinstance(unit_expr, (str, bytes, text_type)):
            if isinstance(unit_expr, bytes):
                unit_expr = unit_expr.decode("utf-8")

            if registry and unit_expr in registry.unit_objs:
                return registry.unit_objs[unit_expr]
            else:
                unit_key = unit_expr
                if not unit_expr:
                    # Bug catch...
                    # if unit_expr is an empty string, parse_expr fails hard...
                    unit_expr = "1"
                try:
                    unit_expr = parse_expr(unit_expr, global_dict=global_dict,
                                           transformations=unit_text_transform)
                except SyntaxError as e:
                    msg = ("Unit expression %s raised an error "
                           "during parsing:\n%s" % (unit_expr, repr(e)))
                    raise UnitParseError(msg)
        elif isinstance(unit_expr, Unit):
            # grab the unit object's sympy expression.
            unit_expr = unit_expr.expr
        elif hasattr(unit_expr, 'units') and hasattr(unit_expr, 'value'):
            # something that looks like a unyt_array, grab the unit and value
            if unit_expr.shape != ():
                raise UnitParseError(
                    'Cannot create a unit from a non-scalar unyt_array, '
                    'received: %s' % (unit_expr, ))
            value = unit_expr.value
            if value == 1:
                unit_expr = unit_expr.units.expr
            else:
                unit_expr = unit_expr.value*unit_expr.units.expr
        # Make sure we have an Expr at this point.
        if not isinstance(unit_expr, Expr):
            raise UnitParseError("Unit representation must be a string or "
                                 "sympy Expr. %s has type %s."
                                 % (unit_expr, type(unit_expr)))

        # this is slightly faster if unit_expr is the same object as
        # sympy_one than just checking for == equality
        is_one = (unit_expr is sympy_one or unit_expr == sympy_one)
        if dimensions is None and is_one:
            dimensions = dimensionless

        if registry is None:
            # Caller did not set the registry, so use the default.
            registry = default_unit_registry

        # done with argument checking...

        # see if the unit is atomic.
        is_atomic = False
        if isinstance(unit_expr, Symbol):
            is_atomic = True

        #
        # check base_value and dimensions
        #

        if base_value is not None:
            # check that base_value is a float or can be converted to one
            try:
                base_value = float(base_value)
            except ValueError:
                raise UnitParseError("Could not use base_value as a float. "
                                     "base_value is '%s' (type %s)."
                                     % (base_value, type(base_value)))

            # check that dimensions is valid
            if dimensions is not None:
                _validate_dimensions(dimensions)
        else:
            # lookup the unit symbols
            unit_data = _get_unit_data_from_expr(unit_expr, registry.lut)
            base_value = unit_data[0]
            dimensions = unit_data[1]
            if len(unit_data) > 2:
                base_offset = unit_data[2]
                latex_repr = unit_data[3]
            else:
                base_offset = 0.0

        # Create obj with superclass construct.
        obj = Expr.__new__(cls, **assumptions)

        # Attach attributes to obj.
        obj.expr = unit_expr
        obj.is_atomic = is_atomic
        obj.base_value = base_value
        obj.base_offset = base_offset
        obj.dimensions = dimensions
        obj._latex_repr = latex_repr
        obj.registry = registry

        if unit_key is not None:
            registry.unit_objs[unit_key] = obj

        # Return `obj` so __init__ can handle it.

        return obj

    @property
    def latex_repr(self):
        """A LaTeX representation for the unit

        Examples
        --------
        >>> from unyt import g, cm
        >>> (g/cm**3).units.latex_repr
        '\\\\frac{\\\\rm{g}}{\\\\rm{cm}^{3}}'
        """
        if self._latex_repr is not None:
            return self._latex_repr
        if self.expr.is_Atom:
            expr = self.expr
        else:
            expr = self.expr.copy()
        self._latex_repr = _get_latex_representation(expr, self.registry)
        return self._latex_repr

    @property
    def units(self):
        return self

    # Some sympy conventions
    def __getnewargs__(self):
        return (self.expr, self.is_atomic, self.base_value, self.dimensions,
                self.registry)

    def __hash__(self):
        return super(Unit, self).__hash__()

    def _hashable_content(self):
        return (self.expr, self.is_atomic, self.base_value, self.dimensions,
                self.registry)

    # end sympy conventions

    def __repr__(self):
        if self.expr == sympy_one:
            return "(dimensionless)"
        # @todo: don't use dunder method?
        return self.expr.__repr__()

    def __str__(self):
        if self.expr == sympy_one:
            return "dimensionless"
        # @todo: don't use dunder method?
        return self.expr.__str__()

    # for sympy.printing
    def _sympystr(self, *args):
        return str(self.expr)

    #
    # Start unit operations
    #

    def __add__(self, u):
        raise InvalidUnitOperation("addition with unit objects is not allowed")

    def __radd__(self, u):
        raise InvalidUnitOperation("addition with unit objects is not allowed")

    def __sub__(self, u):
        raise InvalidUnitOperation(
            "subtraction with unit objects is not allowed")

    def __rsub__(self, u):
        raise InvalidUnitOperation(
            "subtraction with unit objects is not allowed")

    def __iadd__(self, u):
        raise InvalidUnitOperation(
            "in-place operations with unit objects are not allowed")

    def __isub__(self, u):
        raise InvalidUnitOperation(
            "in-place operations with unit objects are not allowed")

    def __imul__(self, u):
        raise InvalidUnitOperation(
            "in-place operations with unit objects are not allowed")

    def __idiv__(self, u):
        raise InvalidUnitOperation(
            "in-place operations with unit objects are not allowed")

    def __itruediv__(self, u):
        raise InvalidUnitOperation(
            "in-place operations with unit objects are not allowed")

    def __rmul__(self, u):
        return self.__mul__(u)

    def __mul__(self, u):
        """ Multiply Unit with u (Unit object). """
        from unyt.array import unyt_quantity, unyt_array
        if not isinstance(u, Unit):
            cls = type(u)
            if ((cls in (np.ndarray, np.matrix, np.ma.masked_array) or
                 isinstance(u, (numeric_type, list, tuple)))):
                try:
                    units = u.units*self
                except AttributeError:
                    units = self
                data = np.array(u, dtype='float64')
                if data.shape == ():
                    return unyt_quantity(data, units, bypass_validation=True)
                return unyt_array(data, units, bypass_validation=True)
            elif isinstance(u, unyt_array):
                return cls(u, u.units*self, bypass_validation=True)
            else:
                raise InvalidUnitOperation(
                    "Tried to multiply a Unit object with '%s' (type %s). "
                    "This behavior is undefined." % (u, type(u)))

        base_offset = 0.0
        if self.base_offset or u.base_offset:
            if u.dimensions in (temperature, angle) and self.is_dimensionless:
                base_offset = u.base_offset
            elif (self.dimensions in (temperature, angle) and
                  u.is_dimensionless):
                base_offset = self.base_offset
            else:
                raise InvalidUnitOperation(
                    "Quantities with dimensions of angle or units of "
                    "Fahrenheit or Celsius cannot be multiplied.")

        return Unit(self.expr * u.expr,
                    base_value=(self.base_value * u.base_value),
                    base_offset=base_offset,
                    dimensions=(self.dimensions * u.dimensions),
                    registry=self.registry)

    def __div__(self, u):
        """ Divide Unit by u (Unit object). """
        if not isinstance(u, Unit):
            if isinstance(u, (numeric_type, list, tuple, np.ndarray)):
                from unyt.array import unyt_quantity
                return unyt_quantity(1.0, self)/u
            else:
                raise InvalidUnitOperation(
                    "Tried to divide a Unit object by '%s' (type %s). This "
                    "behavior is undefined." % (u, type(u)))

        base_offset = 0.0
        if self.base_offset or u.base_offset:
            if u.dimensions in (temperature, angle) and self.is_dimensionless:
                base_offset = u.base_offset
            elif (self.dimensions in (temperature, angle) and
                  u.is_dimensionless):
                base_offset = self.base_offset
            else:
                raise InvalidUnitOperation(
                    "Quantities with units of Farhenheit "
                    "and Celsius cannot be multiplied.")

        return Unit(self.expr / u.expr,
                    base_value=(self.base_value / u.base_value),
                    base_offset=base_offset,
                    dimensions=(self.dimensions / u.dimensions),
                    registry=self.registry)

    __truediv__ = __div__

    def __rdiv__(self, u):
        return u * self**-1

    def __rtruediv__(self, u):
        return u * self**-1

    def __pow__(self, p):
        """ Take Unit to power p (float). """
        try:
            p = Rational(str(p)).limit_denominator()
        except (ValueError, TypeError):
            raise InvalidUnitOperation("Tried to take a Unit object to the "
                                       "power '%s' (type %s). Failed to cast "
                                       "it to a float." % (p, type(p)))

        return Unit(self.expr**p, base_value=(self.base_value**p),
                    dimensions=(self.dimensions**p),
                    registry=self.registry)

    def __eq__(self, u):
        """ Test unit equality. """
        if not isinstance(u, Unit):
            return False
        return (self.base_value == u.base_value and
                self.dimensions == u.dimensions)

    def __ne__(self, u):
        """ Test unit inequality. """
        if not isinstance(u, Unit):
            return True
        if self.base_value != u.base_value:
            return True
        # use 'is' comparison dimensions to avoid expensive sympy operation
        if self.dimensions is u.dimensions:
            return False
        # fall back to expensive sympy comparison
        return self.dimensions != u.dimensions

    def copy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        expr = str(self.expr)
        base_value = copy.deepcopy(self.base_value)
        base_offset = copy.deepcopy(self.base_offset)
        dimensions = copy.deepcopy(self.dimensions)
        lut = copy.deepcopy(self.registry.lut)
        registry = UnitRegistry(lut=lut)
        return Unit(expr, base_value, base_offset, dimensions, registry)

    #
    # End unit operations
    #

    def same_dimensions_as(self, other_unit):
        """Test if the dimensions of *other_unit* are the same as this unit

        Examples
        --------
        >>> from unyt import Msun, kg, mile
        >>> Msun.units.same_dimensions_as(kg.units)
        True
        >>> Msun.units.same_dimensions_as(mile.units)
        False
        """
        # test first for 'is' equality to avoid expensive sympy operation
        if self.dimensions is other_unit.dimensions:
            return True
        return (self.dimensions / other_unit.dimensions) == sympy_one

    @property
    def is_dimensionless(self):
        """Is this a dimensionless unit?

        Returns
        -------
        True for a dimensionless unit, False otherwise

        Examples
        --------
        >>> from unyt import count, kg
        >>> count.units.is_dimensionless
        True
        >>> kg.units.is_dimensionless
        False
        """
        return self.dimensions is sympy_one

    @property
    def is_code_unit(self):
        """Is this a "code" unit?

        Returns
        -------
        True if the unit name begins with "code" False otherwise

        """
        for atom in self.expr.atoms():
            if str(atom).startswith("code") or atom.is_Number:
                pass
            else:
                return False
        return True

    def list_equivalencies(self):
        """Lists the possible equivalencies associated with this unit object

        Examples
        --------
        >>> from unyt import km
        >>> km.units.list_equivalencies()
        spectral: length <-> frequency <-> energy
        schwarzschild: mass <-> length
        compton: mass <-> length
        """
        from unyt.equivalencies import equivalence_registry
        for k, v in equivalence_registry.items():
            if self.has_equivalent(k):
                print(v())

    def has_equivalent(self, equiv):
        """
        Check to see if this unit object as an equivalent unit in *equiv*.

        Example
        -------
        >>> from unyt import km
        >>> km.units.has_equivalent('spectral')
        True
        """
        try:
            this_equiv = equivalence_registry[equiv]()
        except KeyError:
            raise KeyError("No such equivalence \"%s\"." % equiv)
        old_dims = self.dimensions
        return old_dims in this_equiv._dims

    def get_base_equivalent(self, unit_system="cgs"):
        """Create and return dimensionally-equivalent units in a specified base.

        >>> from unyt import g, cm
        >>> (g/cm**3).units.get_base_equivalent('mks')
        kg/m**3
        >>> (g/cm**3).units.get_base_equivalent('solar')
        Mearth/AU**3
        """
        from unyt.unit_systems import unit_system_registry
        if hasattr(unit_system, "unit_registry"):
            unit_system = unit_system.unit_registry.unit_system_id
        elif unit_system == "code":
            unit_system = self.registry.unit_system_id
        unit_system = unit_system_registry[str(unit_system)]
        try:
            units_string = unit_system[self.dimensions]
        except MissingMKSCurrent:
            units_string = em_conversions[self.dimensions][0]
        return Unit(units_string, registry=self.registry)

    def get_cgs_equivalent(self):
        """Create and return dimensionally-equivalent cgs units.

        Example
        -------
        >>> from unyt import kg, m
        >>> (kg/m**3).units.get_cgs_equivalent()
        g/cm**3
        """
        return self.get_base_equivalent(unit_system="cgs")

    def get_mks_equivalent(self):
        """Create and return dimensionally-equivalent mks units.

        Example
        -------
        >>> from unyt import g, cm
        >>> (g/cm**3).units.get_mks_equivalent()
        kg/m**3
        """
        return self.get_base_equivalent(unit_system="mks")

    def get_conversion_factor(self, other_units):
        """Get the conversion factor and offset (if any) from one unit to another

        Parameters
        ----------
        other_units: unit object
           The units we want the conversion factor for

        Returns
        -------
        conversion_factor : float
            old_units / new_units
        offset : float or None
            Offset between this unit and the other unit. None if there is
            no offset.

        Examples
        --------
        >>> from unyt import km, cm, degree_fahrenheit, degree_celsius
        >>> km.units.get_conversion_factor(cm.units)
        (100000.0, None)
        >>> degree_celsius.units.get_conversion_factor(degree_fahrenheit.units)
        (1.7999999999999998, -31.999999999999886)
        """
        return _get_conversion_factor(self, other_units)

    def latex_representation(self):
        """A LaTeX representation for the unit

        Examples
        --------
        >>> from unyt import g, cm
        >>> (g/cm**3).units.latex_representation()
        '\\\\frac{\\\\rm{g}}{\\\\rm{cm}^{3}}'
        """
        return self.latex_repr

#
# Unit manipulation functions
#


def _get_conversion_factor(old_units, new_units):
    """
    Get the conversion factor between two units of equivalent dimensions. This
    is the number you multiply data by to convert from values in `old_units` to
    values in `new_units`.

    Parameters
    ----------
    old_units: str or Unit object
        The current units.
    new_units : str or Unit object
        The units we want.

    Returns
    -------
    conversion_factor : float
        `old_units / new_units`
    offset : float or None
        Offset between the old unit and new unit.

    """
    ratio = old_units.base_value / new_units.base_value
    if old_units.base_offset == 0 and new_units.base_offset == 0:
        return (ratio, None)
    else:
        if old_units.dimensions in (temperature, angle):
            return ratio, ratio*old_units.base_offset - new_units.base_offset
        else:
            raise InvalidUnitOperation(
                "Fahrenheit and Celsius are not absolute temperature scales "
                "and cannot be used in compound unit symbols.")

#
# Helper functions
#


def _get_unit_data_from_expr(unit_expr, unit_symbol_lut):
    """
    Grabs the total base_value and dimensions from a valid unit expression.

    Parameters
    ----------
    unit_expr: Unit object, or sympy Expr object
        The expression containing unit symbols.
    unit_symbol_lut: dict
        Provides the unit data for each valid unit symbol.

    """
    # The simplest case first
    if isinstance(unit_expr, Unit):
        return (unit_expr.base_value, unit_expr.dimensions)

    # Now for the sympy possibilities
    if isinstance(unit_expr, Number):
        if unit_expr is sympy_one:
            return (1.0, sympy_one)
        return (float(unit_expr), sympy_one)

    if isinstance(unit_expr, Symbol):
        return _lookup_unit_symbol(str(unit_expr), unit_symbol_lut)

    if isinstance(unit_expr, Pow):
        unit_data = _get_unit_data_from_expr(
            unit_expr.args[0], unit_symbol_lut)
        power = unit_expr.args[1]
        if isinstance(power, Symbol):
            raise UnitParseError("Invalid unit expression '%s'." % unit_expr)
        conv = float(unit_data[0]**power)
        unit = unit_data[1]**power
        return (conv, unit)

    if isinstance(unit_expr, Mul):
        base_value = 1.0
        dimensions = 1
        for expr in unit_expr.args:
            unit_data = _get_unit_data_from_expr(expr, unit_symbol_lut)
            base_value *= unit_data[0]
            dimensions *= unit_data[1]

        return (float(base_value), dimensions)

    raise UnitParseError("Cannot parse for unit data from '%s'. Please supply"
                         " an expression of only Unit, Symbol, Pow, and Mul"
                         "objects." % str(unit_expr))


def _lookup_unit_symbol(symbol_str, unit_symbol_lut):
    """
    Searches for the unit data tuple corresponding to the given symbol.

    Parameters
    ----------
    symbol_str : str
        The unit symbol to look up.
    unit_symbol_lut : dict
        Dictionary with symbols as keys and unit data tuples as values.

    """
    if symbol_str in unit_symbol_lut:
        # lookup successful, return the tuple directly
        return unit_symbol_lut[symbol_str]

    # could still be a known symbol with a prefix
    possible_prefix = symbol_str[0]

    if symbol_str[:2] == 'da':
        possible_prefix = 'da'

    if possible_prefix in unit_prefixes:
        # the first character could be a prefix, check the rest of the symbol
        symbol_wo_pref = symbol_str[1:]

        # deca is the only prefix with length 2
        if symbol_str[:2] == 'da':
            symbol_wo_pref = symbol_str[2:]
            possible_prefix = 'da'

        unit_is_si_prefixable = (symbol_wo_pref in unit_symbol_lut and
                                 symbol_wo_pref in prefixable_units)

        if unit_is_si_prefixable is True:
            # lookup successful, it's a symbol with a prefix
            unit_data = unit_symbol_lut[symbol_wo_pref]
            prefix_value = unit_prefixes[possible_prefix]

            if possible_prefix in latex_prefixes:
                latex_repr = symbol_str.replace(
                    possible_prefix, '{'+latex_prefixes[possible_prefix]+'}')
            else:
                # Need to add some special handling for comoving units
                # this is fine for now, but it wouldn't work for a general
                # unit that has an arbitrary LaTeX representation
                if symbol_wo_pref != 'cm' and symbol_wo_pref.endswith('cm'):
                    sub_symbol_wo_prefix = symbol_wo_pref[:-2]
                    sub_symbol_str = symbol_str[:-2]
                else:
                    sub_symbol_wo_prefix = symbol_wo_pref
                    sub_symbol_str = symbol_str
                latex_repr = unit_data[3].replace(
                    '{' + sub_symbol_wo_prefix + '}',
                    '{' + sub_symbol_str + '}')

            # Leave offset and dimensions the same, but adjust scale factor and
            # LaTeX representation
            ret = (unit_data[0] * prefix_value, unit_data[1], unit_data[2],
                   latex_repr)

            unit_symbol_lut[symbol_str] = ret

            return ret

    # no dice
    if symbol_str.startswith('code_'):
        raise UnitParseError(
            "Code units have not been defined. \n"
            "Try creating the array or quantity using ds.arr or ds.quan "
            "instead.")
    else:
        raise UnitParseError("Could not find unit symbol '%s' in the provided "
                             "symbols." % symbol_str)


def _validate_dimensions(dimensions):
    if isinstance(dimensions, Mul):
        for dim in dimensions.args:
            _validate_dimensions(dim)
    elif isinstance(dimensions, Symbol):
        if dimensions not in base_dimensions:
            raise UnitParseError("Dimensionality expression contains an "
                                 "unknown symbol '%s'." % dimensions)
    elif isinstance(dimensions, Pow):
        if not isinstance(dimensions.args[1], Number):
            raise UnitParseError("Dimensionality expression '%s' contains a "
                                 "unit symbol as a power." % dimensions)
    elif isinstance(dimensions, (Add, Number)):
        if not isinstance(dimensions, One):
            raise UnitParseError("Only dimensions that are instances of Pow, "
                                 "Mul, or symbols in the base dimensions are "
                                 "allowed.  Got dimensions '%s'" % dimensions)
    elif not isinstance(dimensions, Basic):
        raise UnitParseError("Bad dimensionality expression '%s'." %
                             dimensions)


def _get_system_unit_string(dimensions, base_units):
    # The dimensions of a unit object is the product of the base dimensions.
    # Use sympy to factor the dimensions into base CGS unit symbols.
    units = []
    my_dims = dimensions.expand()
    if my_dims is dimensionless:
        return ""
    if my_dims in base_units:
        return base_units[my_dims]
    for factor in my_dims.as_ordered_factors():
        dim = list(factor.free_symbols)[0]
        unit_string = str(base_units[dim])
        if factor.is_Pow:
            power_string = "**(%s)" % factor.as_base_exp()[1]
        else:
            power_string = ""
        units.append("(%s)%s" % (unit_string, power_string))
    return " * ".join(units)


def _define_unit(registry, symbol, value, tex_repr=None, offset=None,
                 prefixable=False):
    from unyt.array import unyt_quantity, _iterable
    if symbol in registry:
        raise RuntimeError("The symbol \"%s\" is already in the unit registry!"
                           % symbol)
    if not isinstance(value, unyt_quantity):
        if _iterable(value) and len(value) == 2:
            value = unyt_quantity(value[0], value[1])
        else:
            raise RuntimeError("\"value\" needs to be a (value, unit) tuple!")
    base_value = float(value.in_base(unit_system='cgs-ampere'))
    dimensions = value.units.dimensions
    registry.add(symbol, base_value, dimensions, tex_repr=tex_repr,
                 offset=offset)
    if prefixable:
        prefixable_units.append(symbol)


def define_unit(symbol, value, tex_repr=None, offset=None, prefixable=False):
    """
    Define a new unit and add it to the default unit registry.

    Parameters
    ----------
    symbol : string
        The symbol for the new unit.
    value : tuple or ~unyt.array.unyt_quantity
        The definition of the new unit in terms of some other units. For
        example, one would define a new "mph" unit with (1.0, "mile/hr")
    tex_repr : string, optional
        The LaTeX representation of the new unit. If one is not supplied, it
        will be generated automatically based on the symbol string.
    offset : float, optional
        The default offset for the unit. If not set, an offset of 0 is assumed.
    prefixable : boolean, optional
        Whether or not the new unit can use SI prefixes. Default: False

    Examples
    --------
    >>> from unyt import day
    >>> two_weeks = 14.0*day
    >>> define_unit("fortnight", two_weeks)
    """
    _define_unit(default_unit_registry, symbol, value, tex_repr=tex_repr,
                 offset=offset, prefixable=prefixable)
