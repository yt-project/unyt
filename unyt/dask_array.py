"""
a dask array class (unyt_dask_array) and helper functions for unyt.



"""

import unyt.array as ua
from numpy import ndarray
from dask.array.core import Array, finalize

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
    unit info after applying the dask function. this includes functions that return
    single values (e.g., min()) or functions that change the array in ways that do
    not affect units like reshaping or rounding.

    Parameters
    ----------

    dask_func: func handle
       the dask function handle to call
    current_unyt_dask: unyt_dask_array
       the current instance of a unyt_dask_array

    Returns a new unyt_dask_array instance with appropriate units
    """
    def wrapper(*args, **kwargs):
        da = dask_func(*args, **kwargs)  # will return standard dask array
        return _create_with_quantity(da, current_unyt_dask._unyt_quantity)

    return wrapper


_unyt_funcs_to_track = [
    "to",
    "in_units",
    "in_cgs",
    "in_base",
    "in_mks"
]

def _track_conversion(unyt_func_name, current_unyt_dask):
    """
    a decorator to use with unyt functions that convert units

    Parameters
    ----------

    unyt_func_name: str
        the name of the function to call from the sidecar _unyt_quantity. Must be
        an attribute of unyt.unyt_quantity.
    current_unyt_dask: unyt_dask_array
        the current instance of a unyt_dask_array


    Returns a new unyt_dask_array instance with appropriate units
    """
    def wrapper(*args, **kwargs):

        # current value of sidecar quantity
        init_val = current_unyt_dask._unyt_quantity.value

        # get the unyt function handle and call it
        the_func = getattr(current_unyt_dask._unyt_quantity, unyt_func_name)
        new_unyt_quantity = the_func(*args, **kwargs)

        # calculate the conversion factor and pull out the new name and units
        factor = new_unyt_quantity.value / init_val
        units = new_unyt_quantity.units

        # apply the factor, return new
        new_obj = unyt_from_dask(current_unyt_dask * factor, units)

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


def _prep_ufunc(ufunc, *input, extract_dask=False, **kwargs):
    # this function:
    # (1) sanitizes inputs for calls to __array_func__, __array__ and _elementwise
    # (2) applies the function to the hidden unyt quantities
    # (3) (optional) makes inputs extra clean: converts unyt_dask_array args to plain dask array objects

    # apply the operation to the hidden unyt_quantities
    unyt_inputs = [_extract_unyt(i) for i in input]
    unyt_result = ufunc(*unyt_inputs, **kwargs)

    if extract_dask:
        new_inputs = [_extract_dask(i) for i in input]
        return new_inputs, unyt_result
    return input, unyt_result


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

        # set the unit conversion attributes so they are discoverable. no name
        # conflicts for now, but this could be an issue if _unyt_funcs_to_track
        # is expanded.
        for attr in _unyt_funcs_to_track:
            setattr(obj, attr, getattr(obj._unyt_quantity, attr))

        return obj

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

    def __repr__(self):
        disp_str = super().__repr__().replace('dask.array', 'unyt_dask_array')
        units_str = f", units={self.units.__str__()}>"
        return disp_str.replace(">", units_str)

    def to_dask(self):
        """ return a plain dask array. Only copies high level graphs, should be cheap...
        """
        (cls, args) = self.__reduce__()
        return super().__new__(Array, *args)

    def __getattribute__(self, name):

        if name in _unyt_funcs_to_track:
            return _track_conversion(name, self)

        result = super().__getattribute__(name)
        if name in _use_simple_decorator:
            return _simple_dask_decorator(result, self)

        return result

    def __dask_postcompute__(self):
        # a dask hook to catch after .compute(), see
        # https://docs.dask.org/en/latest/custom-collections.html#example-dask-collection
        return _finalize_unyt, ((self.units, ))

    # straightforward operations where the operation applies to the unit
    # These methods bypass __getattribute__ and numpy hooks, but maybe there's a way
    # to more programmatically generate these (e.g., see
    # https://stackoverflow.com/a/57211141/9357244)
    def __abs__(self):
        return _create_with_quantity(super().__abs__(), self._unyt_quantity)

    def __pow__(self, other):
        # unyt and dask implement this directly:
        un_qua = self._unyt_quantity ** other
        return _create_with_quantity(self.to_dask()**other, un_qua)

    def __mul__(self, other):
        if isinstance(other, unyt_dask_array):
            un_qua = self._unyt_quantity * other._unyt_quantity
            other = other.to_dask()
        else:
            un_qua = self._unyt_quantity

        return _create_with_quantity(self.to_dask() * other, un_qua)

    def __rmul__(self, other):
        if isinstance(other, unyt_dask_array):
            un_qua = self._unyt_quantity * other._unyt_quantity
            other = other.to_dask()
        else:
            un_qua = self._unyt_quantity

        return _create_with_quantity(self.to_dask() * other, un_qua)

    def __div__(self, other):
        if isinstance(other, unyt_dask_array):
            un_qua = self._unyt_quantity / other._unyt_quantity
            other = other.to_dask()
        else:
            un_qua = self._unyt_quantity
        return _create_with_quantity(self.to_dask() / other, un_qua)

    def __rdiv__(self, other):
        if isinstance(other, unyt_dask_array):
            un_qua = other._unyt_quantity / self._unyt_quantity
            other = other.to_dask()
        else:
            un_qua = self._unyt_quantity
        return _create_with_quantity(other / self.to_dask(), un_qua)

    def __truediv__(self, other):
        if isinstance(other, unyt_dask_array):
            un_qua = self._unyt_quantity / other._unyt_quantity
            other = other.to_dask()
        else:
            un_qua = self._unyt_quantity
        return _create_with_quantity(self.to_dask() / other, un_qua)

    def _sanitize_other(self, other, units_must_match=True):
        if type(other) == unyt_dask_array:
            if units_must_match and self.units != other.units:
                raise ValueError("units must match for this operation.")
            return other
        elif type(other) == ua.unyt_quantity:
            return other.value
        else:
            raise ValueError("Argument must be unyt_dask_array or unyt_quantity object.")

    def __add__(self, other):
        new_other = self._sanitize_other(other)
        return _create_with_quantity(self.to_dask() + new_other, self._unyt_quantity)

    def __sub__(self, other):
        new_other = self._sanitize_other(other)
        return _create_with_quantity(self.to_dask() - new_other, self._unyt_quantity)

    def _get_unit_state(self):
        # returns the unit state of the object
        return self.units, self._unyt_quantity, self.unyt_name

    def _set_unit_state(self, units, new_unyt_quantity, unyt_name):
        # sets just the unit state of the object
        self.units = units
        self._unyt_quantity = new_unyt_quantity
        self.unyt_name = unyt_name


def _finalize_unyt(results, unit_name):
    """
    the function to call from the __dask_postcompute__ hook.

    Parameters
    ----------
    results : the dask results object
    unit_name : the units of the result

    Returns
    -------
    unyt_array or unyt_quantity

    """

    # here, we first call the standard finalize function for a dask array
    # and then return a standard unyt_array from the now in-memory result if
    # the result is an array, otherwise return a unyt_quantity.
    result = finalize(results)

    if type(result) == ndarray:
        return ua.unyt_array(result, unit_name)
    else:
        return ua.unyt_quantity(result, unit_name)


def _create_with_quantity(dask_array, new_unyt_quantity):
    """
    this function instantiates a new unyt_dask_array instance and then sets
    the unit state, including units. Used to wrap dask operations

    Parameters
    ----------
    dask_array : a standard dask array
    new_unyt_quantity : a standard unity quantity
    remaining arguments get passed to unyt.unyt_array, check there for a
    description.
    """
    out = unyt_from_dask(dask_array)

    # attach the unyt_quantity
    units = new_unyt_quantity.units
    unyt_name = new_unyt_quantity.name

    out._set_unit_state(units, new_unyt_quantity, unyt_name)
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
