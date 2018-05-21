"""
A registry for units that can be added to and modified.


"""

# -----------------------------------------------------------------------------
# Copyright (c) 2018, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------


import json

from unyt.exceptions import (
    SymbolNotFoundError,
    UnitParseError,
)
from unyt._unit_lookup_table import default_unit_symbol_lut
from hashlib import md5
import six
from sympy import (
    sympify,
    srepr,
)


class UnitRegistry:
    """A registry for unit symbols"""
    _unit_system_id = None

    def __init__(self, add_default_symbols=True, lut=None):
        self._unit_object_cache = {}
        if lut:
            self.lut = lut
        else:
            self.lut = {}

        if add_default_symbols:
            self.lut.update(default_unit_symbol_lut)

    def __getitem__(self, key):
        return self.lut[key]

    def __contains__(self, item):
        return item in self.lut

    def pop(self, item):
        if item in self.lut:
            del self.lut[item]

    @property
    def unit_system_id(self):
        """
        This is a unique identifier for the unit registry created
        from a FNV hash. It is needed to register a dataset's code
        unit system in the unit system registry.
        """
        if self._unit_system_id is None:
            hash_data = bytearray()
            for k, v in sorted(self.lut.items()):
                hash_data.extend(k.encode('ascii'))
                hash_data.extend(repr(v).encode('ascii'))
            m = md5()
            m.update(hash_data)
            self._unit_system_id = str(m.hexdigest())
        return self._unit_system_id

    def add(self, symbol, base_value, dimensions, tex_repr=None, offset=None):
        """
        Add a symbol to this registry.

        """
        from unyt.unit_object import _validate_dimensions

        self._unit_system_id = None

        # Validate
        if not isinstance(base_value, float):
            raise UnitParseError("base_value (%s) must be a float, got a %s."
                                 % (base_value, type(base_value)))

        if offset is not None:
            if not isinstance(offset, float):
                raise UnitParseError(
                    "offset value (%s) must be a float, got a %s."
                    % (offset, type(offset)))
        else:
            offset = 0.0

        _validate_dimensions(dimensions)

        if tex_repr is None:
            # make educated guess that will look nice in most cases
            tex_repr = r"\rm{" + symbol.replace('_', '\ ') + "}"

        # Add to lut
        self.lut.update({symbol: (base_value, dimensions, offset, tex_repr)})

    def remove(self, symbol):
        """
        Remove the entry for the unit matching `symbol`.

        """
        self._unit_system_id = None

        if symbol not in self.lut:
            raise SymbolNotFoundError(
                "Tried to remove the symbol '%s', but it does not exist "
                "in this registry." % symbol)

        del self.lut[symbol]

    def modify(self, symbol, base_value):
        """
        Change the base value of a unit symbol.  Useful for adjusting code
        units after parsing parameters.

        """
        self._unit_system_id = None

        if symbol not in self.lut:
            raise SymbolNotFoundError(
                "Tried to modify the symbol '%s', but it does not exist "
                "in this registry." % symbol)

        if hasattr(base_value, "in_base"):
            new_dimensions = base_value.units.dimensions
            base_value = base_value.in_base('mks')
            base_value = base_value.value
        else:
            new_dimensions = self.lut[symbol][1]

        self.lut[symbol] = ((float(base_value), new_dimensions) +
                            self.lut[symbol][2:])

    def keys(self):
        """
        Print out the units contained in the lookup table.

        """
        return self.lut.keys()

    def to_json(self):
        """
        Returns a json-serialized version of the unit registry
        """
        sanitized_lut = {}
        for k, v in six.iteritems(self.lut):
            san_v = list(v)
            repr_dims = srepr(v[1])
            san_v[1] = repr_dims
            sanitized_lut[k] = tuple(san_v)

        return json.dumps(sanitized_lut)

    @classmethod
    def from_json(cls, json_text):
        """
        Returns a UnitRegistry object from a json-serialized unit registry
        """
        data = json.loads(json_text)
        lut = {}
        for k, v in six.iteritems(data):
            unsan_v = list(v)
            unsan_v[1] = sympify(v[1])
            lut[k] = tuple(unsan_v)

        return cls(lut=lut, add_default_symbols=False)

    def list_same_dimensions(self, unit_object):
        """
        Return a list of base unit names that this registry knows about that
        are of equivalent dimensions to *unit_object*.
        """
        equiv = [k for k, v in self.lut.items()
                 if v[1] is unit_object.dimensions]
        equiv = list(sorted(set(equiv)))
        return equiv


#: The default unit registry
default_unit_registry = UnitRegistry()
