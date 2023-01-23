from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

from matplotlib.units import AxisInfo, ConversionInterface

from unyt.array import unyt_array, unyt_quantity
from unyt.unit_object import Unit

if TYPE_CHECKING:
    import sys

    from matplotlib.axes import Axis

    if sys.version_info >= (3, 9):
        from collections.abc import MutableMapping
    else:
        from typing import MutableMapping


class unyt_arrayConverter(ConversionInterface):
    """Matplotlib interface for unyt_array"""

    _instance = None
    _labelstyle = "()"
    _axisnames: "MutableMapping[Axis, str]" = WeakKeyDictionary()

    # ensure that unyt_arrayConverter is a singleton
    def __new__(cls):
        if unyt_arrayConverter._instance is None:
            unyt_arrayConverter._instance = super().__new__(cls)
        return unyt_arrayConverter._instance

    # When matplotlib first encounters a type in its units.registry, it will
    # call default_units to obtain the units. Then it calls axisinfo to
    # customize the axis - in our case, just set the label. Then matplotlib calls
    # convert.

    @staticmethod
    def axisinfo(unit, axis):
        """Set the axis label based on unit

        Parameters
        ----------

        unit : Unit object, string, or tuple
            This parameter comes from unyt_arrayConverter.default_units() or from
            user code such as Axes.plot(), Axis.set_units(), etc. In user code, it
            is possible to convert the plotted units by specifing the new unit as
            a string, such as "ms", or as a tuple, such as ("J", "thermal")
            following the call signature of unyt_array.convert_to_units().
        axis : Axis object

        Returns
        -------

        AxisInfo object with the label formatted as in-line math latex
        """
        if isinstance(unit, tuple):
            unit = unit[0]
        unit_obj = unit if isinstance(unit, Unit) else Unit(unit)
        name = unyt_arrayConverter._axisnames.get(axis, "")
        if unit_obj.is_dimensionless:
            label = name
        else:
            name += " "
            unit_str = unit_obj.latex_representation()
            if unyt_arrayConverter._labelstyle == "[]":
                label = name + "$\\left[" + unit_str + "\\right]$"
            elif unyt_arrayConverter._labelstyle == "/":
                axsym = "$q_{\\rm" + axis.axis_name + "}$"
                name = axsym if name == " " else name
                if "/" in unit_str:
                    label = name + "$\\;/\\;\\left(" + unit_str + "\\right)$"
                else:
                    label = name + "$\\;/\\;" + unit_str + "$"
            else:
                label = name + "$\\left(" + unit_str + "\\right)$"
        return AxisInfo(label=label.strip())

    @staticmethod
    def default_units(x, axis):
        """Return the Unit object of the unyt_array x

        Parameters
        ----------

        x : unyt_array
        axis : Axis object

        Returns
        -------

        Unit object
        """
        # In the case where the first matplotlib command is setting limits,
        # x may be a tuple of length two (with the same units).
        if isinstance(x, tuple):
            name = getattr(x[0], "name", "")
            units = x[0].units
        else:
            name = getattr(x, "name", "")
            units = x.units

        # maintain a mapping between Axis and name since Axis does not point to
        # its underlying data and we want to propagate the name to the axis
        # label in the subsequent call to axisinfo
        unyt_arrayConverter._axisnames[axis] = name if name is not None else ""
        return units

    @staticmethod
    def convert(value, unit, axis):
        """Convert the units of value to unit

        Parameters
        ----------

        value : unyt_array, unyt_quantity, or sequence there of
        unit : Unit, string or tuple
            This parameter comes from unyt_arrayConverter.default_units() or from
            user code such as Axes.plot(), Axis.set_units(), etc. In user code, it
            is possible to convert the plotted units by specifing the new unit as
            a string, such as "ms", or as a tuple, such as ("J", "thermal")
            following the call signature of unyt_array.convert_to_units().
        axis : Axis object

        Returns
        -------

        unyt_array

        Raises
        ------

        UnitConversionError if unit does not have the same dimensions as value or
        if we don't know how to convert value.
        """
        converted_value = value
        if isinstance(unit, str) or isinstance(unit, Unit):
            unit = (unit,)
        if isinstance(value, (unyt_array, unyt_quantity)):
            converted_value = value.to(*unit)
        else:
            value_type = type(value)
            converted_value = []
            for obj in value:
                converted_value.append(obj.to(*unit))
            converted_value = value_type(converted_value)
        return converted_value
