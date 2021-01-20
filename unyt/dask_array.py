"""
dask_array class and helper functions for unyt.



"""

from dask.array.core import Array, finalize  # TO DO: handle optional dep.
from unyt.array import unyt_quantity, unyt_array

_use_simple_decorator = [
    'min', 'max', 'argmin', 'argmax', 'sum', 'trace', 'mean', 'std', 'cumsum',
    'squeeze', 'rechunk', 'clip', 'view', 'swapaxes', 'round', 'copy',
    '__deepcopy__', 'repeat', 'astype', 'reshape', 'topk'
    ]

_unyt_methods = ['to', ]

# operators that require explicit declaration
# binary operators that need to enforce unit equivalency
_bin_ops_apply_factors = ['__add__', '__sub__']

# operators that should apply the operation to the units
_ops_on_unit = ['__pow__', '__sqrt__']
_ops_pass = ['__abs__']


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

    def __new__(clss,
                dask_graph,
                name,
                chunks,
                dtype=None,
                meta=None,
                shape=None,
                units=None,
                registry=None,
                bypass_validation=False,
                input_units=None,
                unyt_name=None,
                ):

        # get the base dask array
        obj = super(unyt_dask_array, clss).__new__(clss, dask_graph, name, chunks, dtype, meta, shape)

        # attach our unyt sidecar quantity
        dtype = obj.dtype
        obj._unyt_quantity = unyt_quantity(1., units, registry, dtype, bypass_validation, input_units, unyt_name)
        obj.units = obj._unyt_quantity.units
        obj.unyt_name = obj._unyt_quantity.name
        obj.factor = 1.  # dtype issue here?

        return obj

    def to(self, units, equivalence=None, **kwargs):
        # TO DO: append the unyt_quantity docstring...
        # tracks any time units are converted with a running conversion factor
        # that gets applied after calling dask methods.
        init_val = self._unyt_quantity.value
        self._unyt_quantity = self._unyt_quantity.to(units, equivalence, **kwargs)
        self.factor = self.factor * self._unyt_quantity.value / init_val
        self.units = units
        self.unyt_name = self._unyt_quantity.name
        return _attach_unyt(self, self)

    def __getattribute__(self, name):
        result = super().__getattribute__(name)
        if name in _use_simple_decorator:
            return _simple_unyt_decorator(result, self)

        return result

    def __dask_postcompute__(self):
        # a dask hook to catch after .compute(), see
        # https://docs.dask.org/en/latest/custom-collections.html#example-dask-collection
        return _finalize_unyt, ((self.units, self.factor))

    # straightforward operations where the operation applies to the unit
    # These methods bypass __getattribute__, but maybe there's a way
    # to more programmatically generate these (e.g., see
    # https://stackoverflow.com/a/57211141/9357244)
    def __abs__(self):
        return _attach_unyt(super().__abs__(), self)

    def __pow__(self, other):
        # unyt and dask implement this directly:
        self._unyt_quantity = self._unyt_quantity.__pow__(other)
        self.units = self._unyt_quantity.units
        self.factor = self.factor ** other
        return _attach_unyt(super().__pow__(other), self)

    def __mul__(self, other):
        if isinstance(other, unyt_dask_array):
            self.factor = self.factor * other.factor
            self._unyt_quantity = self._unyt_quantity * other._unyt_quantity
            self.units = self._unyt_quantity.units

        return _attach_unyt(super().__mul__(other), self)

    def __rmul__(self, other):
        if isinstance(other, unyt_dask_array):
            self.factor = self.factor * other.factor
            self._unyt_quantity = self._unyt_quantity * other._unyt_quantity
            self.units = self._unyt_quantity.units

        return _attach_unyt(super().__rmul__(other), self)

    def __div__(self, other):
        if isinstance(other, unyt_dask_array):
            self.factor = self.factor / other.factor
            self._unyt_quantity = self._unyt_quantity / other._unyt_quantity
            self.units = self._unyt_quantity.units
        return _attach_unyt(super().__div__(other), self)

    def __rdiv__(self, other):
        if isinstance(other, unyt_dask_array):
            self.factor = other.factor / self.factor
            self._unyt_quantity = other._unyt_quantity / self._unyt_quantity
            self.units = self._unyt_quantity.units
        return _attach_unyt(super().__rdiv__(other), self)

    def __truediv__(self, other):
        if isinstance(other, unyt_dask_array):
            self.factor = self.factor / other.factor
            self._unyt_quantity = self._unyt_quantity / other._unyt_quantity
            self.units = self._unyt_quantity.units
        return _attach_unyt(super().__truediv__(other), self)

    # operations involving other arrays (TO DO: use the unyt_array classes
    # better to do all the unit checks...)

    def _apply_factors_to_graphs(self, other):
        # applies the factors to the dask graphs of each dask array, returns
        # new unyt_dask arrays for each. Used when two incoming unyt_dask arrays
        # have the same units but different factors from prior conversions.
        new_self = unyt_from_dask(super().__mul__(self.factor), self.units)
        new_other = unyt_from_dask(other * other.factor, self.units)
        return new_self, new_other

    def __add__(self, other):
        if self.units == other.units and self.factor != other.factor:
            # same units, but there was a previous conversion
            new_self, new_other = self._apply_factors_to_graphs(other)
            return unyt_from_dask(new_self.__add__(new_other), new_self.units)
        elif self.units != other.units:
            raise ValueError("unyt_dask arrays must have the same units to add")
        else:
            # same units and same factor, the factor can be applied on compute
            return _attach_unyt(super().__add__(other), self)

    def __sub__(self, other):
        if self.units == other.units and self.factor != other.factor:
            # same units, but there was a previous conversion
            new_self, new_other = self._apply_factors_to_graphs(other)
            return unyt_from_dask(new_self.__sub__(new_other), new_self.units)
        elif self.units != other.units:
            raise ValueError("unyt_dask arrays must have the same units to subtract")
        else:
            # same units and same factor, the factor can be applied on compute
            return _attach_unyt(super().__sub__(other), self)

    def _get_unit_state(self):
        # retuns just the unit state of the object
        return self.units, self._unyt_quantity, self.unyt_name, self.factor

    def _set_unit_state(self, units, unyt_quantity, unyt_name, factor):
        # sets just the unit state of the object
        self.units = units
        self._unyt_quantity = unyt_quantity
        self.unyt_name = unyt_name
        self.factor = factor

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
    intermediate dask graph construction

    Parameters
    ----------
    dask_array : dask.array.core.Array object or a unyt_dask_array
        the result of applying a dask array method
    unyt_da_instance : unyt_dask_array
        the current unyt_dask_array instance

    Returns
    -------
    a new unyt_dask_array instance with the proper dask graph

    """

    # extra info: this is primarily needed because the calls to superclass
    # methods like min(), max(), etc. will return base dask Array objects.

    # first create our new instance:
    out = unyt_from_dask(dask_array)

    # now copy over unit state
    out._set_unit_state(*unyt_da_instance._get_unit_state())

    return out


def unyt_from_dask(dask_array,
                   units=None,
                   registry=None,
                   bypass_validation=False,
                   unyt_name=None,
                   ):
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

    da = unyt_dask_array(*args,
                         units=units,
                         registry=registry,
                         bypass_validation=bypass_validation,
                         unyt_name=unyt_name)

    return da
