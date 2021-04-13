"""
a dask array class (unyt_dask_array) and helper functions for unyt.



"""

from dask.array.core import Array, finalize, is_valid_array_chunk  # TO DO: handle optional dep.

import unyt.array as ua
from numpy import ndarray

# the following attributes hang off of dask.array.core.Array and do not modify units
_use_simple_decorator = [
    "min",
    "max",
    "argmin",
    "argmax",
    "sum",
    "mean",
    "std",
    "cumsum",
    "squeeze",
    "rechunk",
    "clip",
    "view",
    "swapaxes",
    "round",
    "copy",
    "__deepcopy__",
    "repeat",
    "astype",
    "reshape",
    "topk",
]


def _simple_dask_decorator(dask_func, current_unyt_dask):
    """
    a decorator for the simpler dask functions that can just copy over the current
    unit info and unit conversion factor after applying the dask function. this
    includes functions that return single values (e.g., min()) or functions that
    change the array in ways that do not affect units like reshaping or rounding.

    Parameters
    ----------

    dask_func: func handle
       the dask function handle to call
    current_unyt_dask: unyt_dask_array
       the current instance of a unyt_dask_array

    Returns a new unyt_dask_array instance with appropriate units and factor
    """
    def wrapper(*args, **kwargs):
        da = dask_func(*args, **kwargs)  # will return standard dask array
        un_qua = current_unyt_dask._unyt_quantity
        factor = current_unyt_dask.factor
        return _create_with_quantity(da, un_qua, factor)

    return wrapper


_unyt_funcs_to_track = [
    "to",
    "in_units",
    "in_cgs",
    "in_base",
    "in_mks"
]

def _track_factor(unyt_func_name, current_unyt_dask):
    """
    a decorator to use with unyt functions to track the conversion factor

    Parameters
    ----------

    unyt_func_name: str
        the name of the function to call from the sidecar _unyt_quantity. Must be
        an attribute of unyt.unyt_quantity.
    current_unyt_dask: unyt_dask_array
        the current instance of a unyt_dask_array


    Returns a new unyt_dask_array instance with appropriate units and factor
    """
    def wrapper(*args, **kwargs):

        # current value of sidecar quantity
        init_val = current_unyt_dask._unyt_quantity.value

        # get the unyt function handle and call it
        the_func = getattr(current_unyt_dask._unyt_quantity, unyt_func_name)
        new_unyt_quantity = the_func(*args, **kwargs)

        # calculate the cumulative conversion factor and pull out the new name and units
        factor = current_unyt_dask.factor * new_unyt_quantity.value / init_val
        unyt_name = new_unyt_quantity.name
        units = new_unyt_quantity.units

        # create a new unyt dask object and set the unit state with our new
        # converted factor, return the new object
        new_obj = unyt_from_dask(current_unyt_dask)
        new_obj._set_unit_state(units, new_unyt_quantity, unyt_name, factor)

        return new_obj
    return wrapper


# helper sanitizing functions for handling ufunc, array operations
def _extract_unyt(obj):
    # returns the hidden _unyt_quantity if it exists in obj, otherwise return obj
    if isinstance(obj, unyt_dask_array):
        return obj._unyt_quantity
    return obj


def _extract_dask(obj):
    # returns a plain dask array if the obj is a unyt_dask_array, otherwise return obj
    if isinstance(obj, unyt_dask_array):
        return obj.to_dask()
    return obj


def _apply_factors_if_dask(obj):
    # returns a new unyt_dask_array with factor applied otherwise return obj
    if isinstance(obj, unyt_dask_array):
        return unyt_from_dask(obj * obj.factor, obj.units)
    return obj


def _prep_ufunc(ufunc, *input, extract_dask=False, **kwargs):
    # this function:
    # (1) sanitizes inputs for calls to __array_func__, __array__ and _elementwise
    # (2) applies the function to the hidden unyt quantities
    # (3) (optional) makes inputs extra clean: converts unyt_dask_array args to plain dask array objects

    # apply unyt factors to any that are unyt_dask_array objects
    new_inputs = [_apply_factors_if_dask(i) for i in input]

    # now we apply the operation to the hidden unyt_quantities
    unyt_inputs = [_extract_unyt(i) for i in new_inputs]
    unyt_result = ufunc(*unyt_inputs, **kwargs)

    if extract_dask:
        new_inputs = [_extract_dask(i) for i in new_inputs]

    return new_inputs, unyt_result


def _post_ufunc(dask_superfunk, unyt_result):
    # a decorator to attach hidden unyt quantity to result of a ufunc, array or elemwise calculation
    def wrapper(*args, **kwargs):
        dask_result = dask_superfunk(*args, **kwargs)
        if hasattr(unyt_result, 'units'):
            return _create_with_quantity(dask_result, unyt_result)
        return dask_result
    return wrapper

# note: the unyt_dask_array class has no way of catching daskified reductions (yet?).
# operations like dask.array.min() get routed through dask.array.reductions.min()
# and will return plain arrays or float/int values. When these operations exist as
# attributes, they can be called and will return unyt objects. i.e., :
# import dask; import unyt
# x_da = unyt_from_dask(dask.array.ones((10, 10), chunks=(2, 2)), unyt.m)
# dask.array.min(x_da).compute()  #  returns a plain float
# x_da.min().compute()  #  returns a unyt quantity


class unyt_dask_array(Array):
    """
    a dask.array.core.Array subclass that tracks units. Easiest to use the
    unyt_from_dask helper function to generate new instances.

    Parameters
    ----------

    All parameters are those for dask.array.core.Array

    """

    def __new__(
        clss,
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
        obj = super(unyt_dask_array, clss).__new__(
            clss,
            dask_graph,
            name,
            chunks,
            dtype,
            meta,
            shape,
        )

        # attach our unyt sidecar quantity
        dtype = obj.dtype
        obj._unyt_quantity = ua.unyt_quantity(
            1.0,
            units,
            registry,
            dtype,
            bypass_validation,
            input_units,
            unyt_name,
        )

        obj.units = obj._unyt_quantity.units
        obj.unyt_name = obj._unyt_quantity.name
        obj.factor = 1.0  # dtype issue here?

        return obj

    def _prep_for_ufunc(self, ufunc, *input, extract_dask=False, **kwargs):

        # apply unyt factors to any that are unyt_dask_array objects
        new_inputs = [self._apply_factors_if_dask(i) for i in input]

        # now we apply the operation to the hidden unyt_quantities
        unyt_inputs = [self._extract_unyt(i) for i in new_inputs]
        unyt_result = ufunc(*unyt_inputs, **kwargs)

        if extract_dask:
            new_inputs = [self._extract_dask(i) for i in new_inputs]

        return new_inputs, unyt_result

    def _elemwise(self, ufunc, *args, **kwargs):
        args, unyt_result = _prep_ufunc(ufunc, *args, **kwargs)
        return _post_ufunc(super()._elemwise, unyt_result)(ufunc, *args, **kwargs)

    def __array_ufunc__(self, numpy_ufunc, method, *inputs, **kwargs):
        inputs, unyt_result = _prep_ufunc(numpy_ufunc, *inputs, extract_dask=True, **kwargs)
        return _post_ufunc(super().__array_ufunc__, unyt_result)(numpy_ufunc, method, *inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        args, unyt_result = _prep_ufunc(func, *args, extract_dask=True, **kwargs)
        types = [type(i) for i in args]
        return _post_ufunc(super().__array_function__, unyt_result)(func, types, args, kwargs)

    def _extract_unyt(self, obj):
        # returns the hidden _unyt_quantity if it exists in obj, otherwise return obj
        if isinstance(obj, unyt_dask_array):
            return obj._unyt_quantity
        return obj

    def _extract_dask(self, obj):
        # returns a plain dask array if the obj is a unyt_dask_array, otherwise return obj
        if isinstance(obj, unyt_dask_array):
            return obj.to_dask()
        return obj

    def _apply_factors_if_dask(self, obj):
        # returns a new unyt_dask_array with factor applied otherwise pass through
        if isinstance(obj, unyt_dask_array):
            return unyt_from_dask(super().__mul__(obj.factor), obj.units)
        return obj

    def to_dask(self):
        """ return a plain dask array. Only copies high level graphs, should be cheap...
        """
        (cls, args) = self.__reduce__()
        return super().__new__(Array, *args)

    def __getattribute__(self, name):

        if name in _unyt_funcs_to_track:
            return _track_factor(name, self)

        result = super().__getattribute__(name)
        if name in _use_simple_decorator:
            return _simple_dask_decorator(result, self)

        return result

    def __dask_postcompute__(self):
        # a dask hook to catch after .compute(), see
        # https://docs.dask.org/en/latest/custom-collections.html#example-dask-collection
        return _finalize_unyt, ((self.units, self.factor))

    # straightforward operations where the operation applies to the unit
    # These methods bypass __getattribute__ and numpy hooks, but maybe there's a way
    # to more programmatically generate these (e.g., see
    # https://stackoverflow.com/a/57211141/9357244)
    def __abs__(self):
        return _create_with_quantity(super().__abs__(), self._unyt_quantity, self.factor)

    def __pow__(self, other):
        # unyt and dask implement this directly:
        un_qua = self._unyt_quantity.__pow__(other)
        factor = self.factor ** other
        return _create_with_quantity(super().__pow__(other), un_qua, factor)

    def __mul__(self, other):
        if isinstance(other, unyt_dask_array):
            factor = self.factor * other.factor
            un_qua = self._unyt_quantity * other._unyt_quantity
        else:
            factor = self.factor
            un_qua = self._unyt_quantity

        return _create_with_quantity(super().__mul__(other), un_qua, factor)

    def __rmul__(self, other):
        if isinstance(other, unyt_dask_array):
            factor = self.factor * other.factor
            un_qua = self._unyt_quantity * other._unyt_quantity
        else:
            factor = self.factor
            un_qua = self._unyt_quantity

        return _create_with_quantity(super().__rmul__(other), un_qua, factor)

    def __div__(self, other):
        if isinstance(other, unyt_dask_array):
            factor = self.factor / other.factor
            un_qua = self._unyt_quantity / other._unyt_quantity
        else:
            factor = self.factor
            un_qua = self._unyt_quantity
        return _create_with_quantity(super().__div__(other), un_qua, factor)

    def __rdiv__(self, other):
        if isinstance(other, unyt_dask_array):
            factor = other.factor / self.factor
            un_qua = other._unyt_quantity / self._unyt_quantity
        else:
            factor = self.factor
            un_qua = self._unyt_quantity
        return _create_with_quantity(super().__rdiv__(other), un_qua, factor)

    def __truediv__(self, other):
        if isinstance(other, unyt_dask_array):
            factor = self.factor / other.factor
            un_qua = self._unyt_quantity / other._unyt_quantity
        else:
            factor = self.factor
            un_qua = self._unyt_quantity
        return _create_with_quantity(super().__truediv__(other), un_qua, factor)

    def _apply_factors_to_graphs(self, other=None):
        # applies the factors to the dask graphs of each dask array, returns
        # new unyt_dask arrays for each. Used when two incoming unyt_dask arrays
        # have the same units but different factors from prior conversions.
        new_self = unyt_from_dask(super().__mul__(self.factor), self.units)
        new_other = None
        if other is not None:
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
            return _create_with_quantity(super().__add__(other), self._unyt_quantity, self.factor)

    def __sub__(self, other):
        if self.units == other.units and self.factor != other.factor:
            # same units, but there was a previous conversion
            new_self, new_other = self._apply_factors_to_graphs(other)
            return unyt_from_dask(new_self.__sub__(new_other), new_self.units)
        elif self.units != other.units:
            raise ValueError("unyt_dask arrays must have the same units to subtract")
        else:
            # same units and same factor, the factor can be applied on compute
            return _create_with_quantity(super().__sub__(other), self._unyt_quantity, self.factor)

    def _get_unit_state(self):
        # returns the unit state of the object
        return self.units, self._unyt_quantity, self.unyt_name, self.factor

    def _set_unit_state(self, units, new_unyt_quantity, unyt_name, factor):
        # sets just the unit state of the object
        self.units = units
        self._unyt_quantity = new_unyt_quantity
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
    # and then return a standard unyt_array from the now in-memory result if
    # the result is an array, otherwise return a unyt_quantity.
    result = finalize(results) * factor

    if type(result) == ndarray:
        return ua.unyt_array(finalize(results) * factor, unit_name)
    else:
        return ua.unyt_quantity(result, unit_name)


def _create_with_quantity(dask_array, new_unyt_quantity, factor=None):
    """
    this function instantiates a new unyt_dask_array instance and then sets
    the unit state, including units and conversion factor. Used to wrap
    dask operations and track the cumulative conversion factor.

    Parameters
    ----------
    dask_array : a standard dask array
    new_unyt_quantity : a standard unity quantity
    factor : float
        the cumulative conversion factor, default to 1.0
    remaining arguments get passed to unyt.unyt_array, check there for a
    description.
    """
    out = unyt_from_dask(dask_array)

    # attach the unyt_quantity
    units = new_unyt_quantity.units
    unyt_name = new_unyt_quantity.name
    if factor is None:
        factor = 1.

    out._set_unit_state(units, new_unyt_quantity, unyt_name, factor)
    return out


def unyt_from_dask(
    dask_array,
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

    >>> from unyt import dask_array
    >>> import dask.array as da
    >>> x = da.random.random((10000, 10000), chunks=(1000, 1000))
    >>> x_da = dask_array.unyt_from_dask(x, 'm')
    >>> x_da.units
    m
    >>> x_da.mean().units()
    m
    >>> x_da.mean().compute()
    unyt_array(0.50001502, 'm')
    >>> x_da.to('cm').mean().compute()
    unyt_array(50.00150242, 'cm')
    >>> (x_da.to('cm')**2).mean().compute()
    unyt_array(3333.37805754, 'cm**2')

    """

    # reduce the dask array to pull out the arguments required for instantiating
    # a new dask.array.core.Array object and then initialize our new unyt_dask
    # array
    (cls, args) = dask_array.__reduce__()

    da = unyt_dask_array(
        *args,
        units=units,
        registry=registry,
        bypass_validation=bypass_validation,
        unyt_name=unyt_name
    )

    return da
