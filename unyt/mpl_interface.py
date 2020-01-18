"""Matplotlib ConversionInterface"""
try:
    from matplotlib.units import ConversionInterface, AxisInfo, registry
except ImportError:
    pass
else:
    from unyt import unyt_array, Unit

    class unyt_arrayConverter(ConversionInterface):
        """Matplotlib interface for unyt_array"""

        @staticmethod
        def axisinfo(unit, axis):
            """return default axis label"""
            if isinstance(unit, tuple):
                unit = unit[0]
            unit_str = Unit(unit).latex_representation()
            label = "$\\left(" + unit_str + "\\right)$"
            return AxisInfo(label=label)

        @staticmethod
        def default_units(x, axis):
            """Return the default unit for x or None"""
            return x.units

        @staticmethod
        def convert(value, unit, axis):
            """Convert"""
            if isinstance(unit, str) or isinstance(unit, Unit):
                unit = (unit,)
            return value.to(*unit)

    registry[unyt_array] = unyt_arrayConverter()
