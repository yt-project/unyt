"""
Matplotlib offers support for custom classes, such as unyt_array, allowing customization
of axis information and unit conversion. In the case of unyt, the axis label is set
based on the unyt_array.name and unyt_array.units attributes. It is also possible to
convert the plotted units.

This feature is optional and has to be enabled using the matplotlib_support context
manager.
"""

from unyt.array import unyt_array, unyt_quantity

from ._on_demand_imports import _matplotlib

__all__ = ["matplotlib_support"]


class matplotlib_support:
    """Context manager for enabling the feature

    When used in a with statement, the feature is enabled during the context and
    then disabled after it exits.

    Parameters
    ----------

    label_style : str
        One of the following set, ``{'()', '[]', '/'}``. These choices
        correspond to the following unit labels:

        * ``'()'`` -> ``'(unit)'``
        * ``'[]'`` -> ``'[unit]'``
        * ``'/'`` -> ``'q_x / unit'``
    """

    @property
    def array_converter(self):
        from ._mpl_array_converter import unyt_arrayConverter

        unyt_arrayConverter._labelstyle = self.label_style
        return unyt_arrayConverter

    def __init__(self, label_style="()"):
        self._labelstyle = label_style
        self._enabled = False

    def __call__(self):
        self.__enter__()

    @property
    def label_style(self):
        """str: One of the following set, ``{'()', '[]', '/'}``.
        These choices correspond to the following unit labels:

            * ``'()'`` -> ``'(unit)'``
            * ``'[]'`` -> ``'[unit]'``
            * ``'/'`` -> ``'q_x / unit'``
        """
        return self._labelstyle

    @label_style.setter
    def label_style(self, label_style="()"):
        self._labelstyle = label_style
        self.array_converter._labelstyle = label_style

    def __enter__(self):
        _matplotlib.units.registry[unyt_array] = self.array_converter()
        _matplotlib.units.registry[unyt_quantity] = self.array_converter()
        self._enabled = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        _matplotlib.units.registry.pop(unyt_array)
        _matplotlib.units.registry.pop(unyt_quantity)
        self._enabled = False

    def enable(self):
        self.__enter__()

    def disable(self):
        if self._enabled:
            self.__exit__(None, None, None)
