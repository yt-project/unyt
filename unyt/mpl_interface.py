"""
Register unyt_array with Matplotlib if it is available


"""

# -----------------------------------------------------------------------------
# Copyright (c) 2020, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------


try:
    from matplotlib.units import (
        ConversionInterface,
        AxisInfo,
        registry,
        ConversionError,
    )
except ImportError:
    pass
else:
    from unyt import unyt_array, unyt_quantity, Unit

    class unyt_arrayConverter(ConversionInterface):
        """Matplotlib interface for unyt_array"""

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
            unit_obj = Unit(unit)
            if unit_obj.is_dimensionless:
                label = ""
            else:
                unit_str = unit_obj.latex_representation()
                label = "$\\left(" + unit_str + "\\right)$"
            return AxisInfo(label=label)

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
            return x.units

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

            ConversionError if unit does not have the same dimensions as value or
            if we don't know how to convert value.
            """
            if isinstance(unit, str) or isinstance(unit, Unit):
                unit = (unit,)
            if isinstance(value, (unyt_array, unyt_quantity)):
                converted_value = value.to(*unit)
            elif isinstance(value, (list, tuple)):
                value_type = type(value)
                converted_value = []
                for obj in value:
                    converted_value.append(obj.to(*unit))
                converted_value = value_type(converted_value)
            else:
                raise ConversionError(f"unable to convert {value}")
            return converted_value

    class MplUnitsCM:
        """Context manager for experimenting with Unyt in Matplotlib"""

        def __enter__(self):
            registry[unyt_array] = unyt_arrayConverter()
            registry[unyt_quantity] = unyt_arrayConverter()

        def __exit__(self, exc_type, exc_val, exc_tb):
            registry.pop(unyt_array)
            registry.pop(unyt_quantity)

        def enable(self):
            self.__enter__()

        def disable(self):
            self.__exit__(None, None, None)

        def __repr__(self):
            return "<MplUnitsCM>"
