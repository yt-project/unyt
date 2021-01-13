"""
dask_array class and helper functions for unyt.



"""

from dask.array.core import Array, finalize  # TO DO: handle optional dep.
from unyt.array import unyt_array
import numpy as np


_use_simple_decorator = [
    'min', 'max', 'argmin', 'argmax', 'sum', 'trace', 'mean', 'std', 'cumsum',
    'squeeze', 'rechunk', 'clip', 'view', 'swapaxes', 'round', 'copy',
    '__deepcopy__', 'repeat', 'astype', 'reshape', 'topk'
    ]


def _simple_unyt_decorator(dask_func, current_unyt_dask):
    # a decorator for the simpler functions that can just copy over the current
    # unit info and unit conversion factor after applying the dask function.
    # this includes functions that return single values (e.g., min()) or
    # functions that change the array in ways that do not affect units
    # like reshaping or rounding.
    def wrapper(*args, **kwargs):
        da = dask_func(*args, **kwargs)  # will return standard dask array
        return _attach_unyt(da, current_unyt_dask)
    return wrapper


class unyt_dask_array(Array):
    """
        a dask.array.core.Array subclass that tracks units. Easiest to use the
        unyt_from_dask helper function to generate new instances.

        Parameters
        ----------

        All parameters are those for dask.array.core.Array

        Examples
        --------
    """

    def __init__(self, dask_graph, name, chunks, dtype=None, meta=None, shape=None):
        self.units = None
        self.unyt_name = None
        self.dask_name = name
        self.factor = 1.
        self._unyt_array = None

    def _attach_units(self, units=None,
                      registry=None,
                      dtype=None,
                      bypass_validation=False,
                      input_units=None,
                      name=None):
        x_np = np.array([1.])
        self._unyt_array = unyt_array(x_np, units, registry, dtype, bypass_validation, input_units, name)
        self.units = self._unyt_array.units
        self.unyt_name = self._unyt_array.name

    def to(self, units, equivalence=None, **kwargs):
        # tracks any time units are converted with a running conversion factor
        # that gets applied after calling dask methods.
        init_val = self._unyt_array.value[0]
        self._unyt_array = self._unyt_array.to(units, equivalence, **kwargs)
        self.factor = self.factor * self._unyt_array.value[0] / init_val
        self.units = units
        self.unyt_name = self._unyt_array.name

    def __getattribute__(self, name):
        result = super().__getattribute__(name)
        if name in _use_simple_decorator:
            return _simple_unyt_decorator(result, self)

        return result

    def __abs__(self):
        # hmm, this doesn't get caught if in _use_simple_decorator...
        return _attach_unyt(super().__abs__(), self)

    def __pow__(self, other):
        # unyt and dask implement this directly:
        self._unyt_array = self._unyt_array.__pow__(other)
        self.units = self._unyt_array.units
        self.factor = self.factor ** other
        return _attach_unyt(super().__pow__(other), self)

    # operations involving other arrays (TO DO: use the unyt_array classes
    # better to do all the unit checks...)
    def __add__(self, other):
        # yikes, what to do about self.factor here????
        return _attach_unyt(super().__add__(other), self)

    def __sub__(self, other):
        # yikes, what to do about self.factor here????
        return _attach_unyt(super().__sub__(other), self)

    def __mul__(self, other):
        if isinstance(other, unyt_dask_array):
            self.factor = self.factor * other.factor
            self._unyt_array = self._unyt_array * other._unyt_array
            self.units = self._unyt_array.units

        return _attach_unyt(super().__mul__(other), self)

    def __rmul__(self, other):
        if isinstance(other, unyt_dask_array):
            self.factor = self.factor * other.factor
            self._unyt_array = self._unyt_array * other._unyt_array
            self.units = self._unyt_array.units

        return _attach_unyt(super().__rmul__(other), self)

    def __div__(self, other):
        if isinstance(other, unyt_dask_array):
            self.factor = self.factor / other.factor
            self._unyt_array = self._unyt_array / other._unyt_array
            self.units = self._unyt_array.units
        return _attach_unyt(super().__div__(other), self)

    def __rdiv__(self, other):
        if isinstance(other, unyt_dask_array):
            self.factor = other.factor / self.factor
            self._unyt_array = other._unyt_array / self._unyt_array
            self.units = self._unyt_array.units
        return _attach_unyt(super().__rdiv__(other), self)

    def __truediv__(self, other):
        if isinstance(other, unyt_dask_array):
            self.factor = self.factor / other.factor
            self._unyt_array = self._unyt_array / other._unyt_array
            self.units = self._unyt_array.units
        return _attach_unyt(super().__truediv__(other), self)

    def __dask_postcompute__(self):
        # a dask hook to catch after .compute(), see
        # https://docs.dask.org/en/latest/custom-collections.html#example-dask-collection
        return _finalize_unyt, ((self.units, self.factor))


def _finalize_unyt(results, unit_name, factor):
    """
    the function to call from the __dask_postcompute__ hook.

    Parameters
    ----------
    results : the dask results object
    unit_name : the units of the result
    factor : the current value of the unit conversion factor

    Returns
    -------
    unyt_array

    """

    # here, we first call the standard finalize function for a dask array
    # and then return a standard unyt_array from the now in-memory result.
    return unyt_array(finalize(results) * factor, unit_name)


def _attach_unyt(dask_array, unyt_da_instance):
    """
    helper function to return a new unyt_dask_array instance after an
    intermediate dask graph construction... would be better to have a
    unyt_dask_array.__new__ but still working out some conflicts with
    dask.array.core.Array.__new__

    this works for now...

    Parameters
    ----------
    dask_array : dask.array.core.Array object
        the result of applying a dask array method
    unyt_da_instance : unyt_dask_array
        the current unyt_dask_array instance

    Returns
    -------
    a new unyt_dask_array instance with the proper dask graph

    """

    # first create our new instance:
    (cls, da_args) = dask_array.__reduce__()
    out = unyt_dask_array(*da_args)

    # now use copy over the unit info from the old instance
    out.units = unyt_da_instance.units
    out.unyt_name = unyt_da_instance.unyt_name
    out.dask_name = unyt_da_instance.dask_name
    out.factor = unyt_da_instance.factor
    out._unyt_array = unyt_da_instance._unyt_array

    return out


def unyt_from_dask(dask_array,
                   units=None,
                   registry=None,
                   dtype=None,
                   bypass_validation=False,
                   input_units=None,
                   name=None):
    """
    creates a unyt_dask_array from a standard dask array.

    Parameters
    ----------
    dask_array : a standard dask array

    remaining arguments get passed to unyt.unyt_array, check there for a
    description.

    Examples
    --------

    """

    # reduce the dask array to pull out the arguments required for instantiating
    # a new dask.array.core.Array object and then initialize our new unyt_dask
    # array
    (cls, args) = dask_array.__reduce__()
    da = unyt_dask_array(*args)

    # attach the unyt info to the array
    da._attach_units(units, registry, dtype, bypass_validation, input_units, name)

    return da
