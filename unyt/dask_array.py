"""
a dask array class (unyt_dask_array) and helper functions for unyt.



"""

import sys
from functools import wraps

import numpy as np

import unyt.array as ua

if "pytest" in sys.modules:
    # should only happen if pytest is installed *and* already imported,
    # so we can skip collecting doctests from this module when dask isn't installed
    # while avoiding making pytest itself a hard dependency to this module.
    # This check is constructed to work with direct invocation (pytest unyt)
    # as well as through python -m pytest
    import pytest

    pytest.importorskip("dask")
    del pytest

from dask.array.core import Array as DaskArray, finalize as dask_finalize  # noqa: E402

# the following attributes hang off of dask.array.core.Array and do not modify units
_use_unary_decorator = {
    "min",
    "max",
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
    "repeat",
    "astype",
    "reshape",
    "topk",
}


def _unary_dask_decorator(dask_func, current_unyt_dask):
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

    @wraps(dask_func)
    def wrapper(*args, **kwargs):
        da = dask_func(*args, **kwargs)  # will return standard dask array
        return _create_with_quantity(da, current_unyt_dask._unyt_array)

    return wrapper


# the following list are the unyt_quantity/array attributes that will also
# be attributes of the unyt_dask_array. These methods are wrapped with a unit
# conversion decorator.
_unyt_funcs_to_track = {"to", "in_units", "in_cgs", "in_base", "in_mks"}


# helper sanitizing functions for handling ufunc, array operations
def _extract_unyt(obj):
    # returns the hidden _unyt_quantity if it exists in obj, otherwise return obj
    if _is_iterable(obj):
        return [_extract_unyt(ob) for ob in obj]
    return obj._unyt_array if isinstance(obj, unyt_dask_array) else obj


def _extract_dask(obj):
    # returns a plain dask array if the obj is a unyt_dask_array, otherwise return obj
    if _is_iterable(obj):
        return [_extract_dask(ob) for ob in obj]
    return obj.to_dask() if isinstance(obj, unyt_dask_array) else obj


def _extract_unyt_val(obj):
    # returns the value of a unyt_quantity if obj is a unyt_quantity
    if _is_iterable(obj):
        return [_extract_unyt_val(ob) for ob in obj]
    return obj.to_value() if isinstance(obj, ua.unyt_quantity) else obj


def _check_unyt_inputs(ui_0, ui_1):
    # returns True if the args need to be converted to the same unit, based on
    # if the units have the same dimensions but different units.

    if hasattr(ui_0, "units") and hasattr(ui_1, "units"):
        un_0, un_1 = ui_0.units, ui_1.units
        return un_0 != un_1 and un_0.dimensions == un_1.dimensions
    return False


def _is_iterable(obj):
    return type(obj) is tuple or type(obj) is list  # noqa E721


def _sanitize_unit_args(*input):
    # returns sanitized inputs and unyt_inputs for calling the ufunc
    unyt_inputs = _extract_unyt(input)

    if len(unyt_inputs) > 1:
        # note 1: even though we rely on the ufunc applied to the unyt quantities
        # to get the final units of our unyt_dask_array after an operation,
        # we need to first ensure that if our arguments have the same dimensions
        # they are in the same units. This happens internally for unyt_quantities
        # but we also need to apply those internal unit conversions to our
        # dask_unyt_array objects, so we do those checks manually here.
        # note 2: we do NOT check for validity of the operation here. The subsequent
        # call to the ufunc with the unyt_inputs will enforce the unyt rules
        # (e.g., addition must have same dimensions).
        ui_0, ui_1 = unyt_inputs[0], unyt_inputs[1]
        if _check_unyt_inputs(ui_0, ui_1):
            # convert to the unit with the larger base
            input = list(input)
            if ui_0.units.base_value < ui_1.units.base_value:
                input[0] = input[0].to(ui_1.units)
            else:
                input[1] = input[1].to(ui_0.units)
            unyt_inputs = _extract_unyt(input)

    return input, unyt_inputs


def _prep_ufunc(ufunc, *input, extract_dask=False, **kwargs):
    # this function:
    # (1) sanitizes inputs for calls to __array_func__, __array__ and _elementwise
    # (2) applies the function to the hidden unyt quantities
    # (3) (optional) makes inputs extra clean: converts unyt_dask_array args to
    #     plain dask array objects

    # apply the operation to the hidden unyt_quantities
    input, unyt_inputs = _sanitize_unit_args(*input)
    unyt_result = ufunc(*unyt_inputs, **kwargs)

    if extract_dask:
        input = _extract_dask(input)

    input = _extract_unyt_val(input)
    return input, unyt_result


def _post_ufunc(dask_superfunk, unyt_result):
    # a decorator to attach hidden unyt quantity to result of a ufunc, array or
    # elemwise calculation
    def wrapper(*args, **kwargs):
        dask_result = dask_superfunk(*args, **kwargs)
        if hasattr(unyt_result, "units"):
            return _create_with_quantity(dask_result, unyt_result)
        return dask_result

    return wrapper


def _special_dec(the_func):
    # decorator for special operations like __mul__ , __truediv__, which will
    # bypass __getattribute__
    def wrapper(self, *args, **kwargs):
        funcname = the_func.__name__
        ufunc = getattr(ua.unyt_quantity, funcname)
        ufunc_args = (self,) + args
        args, unyt_result = _prep_ufunc(ufunc, *ufunc_args, extract_dask=True, **kwargs)

        # remove the first arg, cause we need to supply self as first arg
        _ = args.pop(0)
        daskresult = the_func(self, *args, **kwargs)

        if hasattr(unyt_result, "units"):
            return _create_with_quantity(daskresult, unyt_result)
        return daskresult  # __lt__, __le__, etc. will hit this

    return wrapper


class unyt_dask_array(DaskArray):
    """
    a dask.array.core.Array subclass that tracks units. This class is only
    recommended for advanced usage, most cases should use the unyt_from_dask
    helper function to instantiate a new unyt_dask_array.

    Parameters
    ----------

    The following parameters are those for a standard dask.array.core.Array:

    dask : dict
        Task dependency graph
    name : string
        Name of array in dask
    shape : tuple of ints
        Shape of the entire array
    chunks: iterable of tuples
        block sizes along each dimension
    dtype : str or dtype
        Typecode or data-type for the new Dask Array
    meta : empty ndarray
        empty ndarray created with same NumPy backend, ndim and dtype as the
        Dask Array being created (overrides dtype)

    The following keyword-only parameters are the same as for a standard
    unyt.unyt_array (check there for definitions):

    units
    registry
    bypass_validation
    unyt_name

    """

    def __new__(
        cls,
        dask_graph,
        name,
        chunks,
        dtype=None,
        meta=None,
        shape=None,
        *,
        units=None,
        registry=None,
        bypass_validation=False,
        unyt_name=None,
    ):
        # get the base dask array
        obj = super().__new__(
            cls,
            dask_graph,
            name,
            chunks,
            dtype,
            meta,
            shape,
        )

        # attach our unyt sidecar quantity
        dtype = obj.dtype
        obj._unyt_array = ua.unyt_array(
            [1.0],
            units,
            registry,
            dtype,
            bypass_validation=bypass_validation,
            name=unyt_name,
        )

        obj.units = obj._unyt_array.units
        obj.unyt_name = obj._unyt_array.name

        # set the unit conversion attributes so they are discoverable. no name
        # conflicts for now, but this could be an issue if _unyt_funcs_to_track
        # is expanded.
        for attr in _unyt_funcs_to_track:
            setattr(obj, attr, getattr(obj._unyt_array, attr))

        return obj

    def _elemwise(self, ufunc, *args, **kwargs):
        args, unyt_result = _prep_ufunc(ufunc, *args, **kwargs)
        return _post_ufunc(super()._elemwise, unyt_result)(ufunc, *args, **kwargs)

    def __array_ufunc__(self, numpy_ufunc, method, *inputs, **kwargs):
        inputs, unyt_result = _prep_ufunc(
            numpy_ufunc, *inputs, extract_dask=True, **kwargs
        )
        wrapped_func = _post_ufunc(super().__array_ufunc__, unyt_result)
        return wrapped_func(numpy_ufunc, method, *inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        args, unyt_result = _prep_ufunc(func, *args, extract_dask=True, **kwargs)
        types = [type(i) for i in args]
        wrapped_func = _post_ufunc(super().__array_function__, unyt_result)
        return wrapped_func(func, types, args, kwargs)

    def __getitem__(self, index):
        return _unary_dask_decorator(super().__getitem__, self)(index)

    def __setitem__(self, key, value):
        # requires dask >= 2021.4.1
        super().__setitem__(key, value)

    def __repr__(self):
        disp_str = super().__repr__().replace("dask.array", "unyt_dask_array")
        units_str = f", units={self.units}>"
        return disp_str.replace(">", units_str)

    def _repr_html_(self):
        # controls jupyter notebook display of an array. called from _repr_html_
        base_table = super()._repr_html_()
        table = base_table.split("\n")
        new_table = []
        for row in table:
            if "</tbody>" in row:
                u = self.units
                newrow = f"    <tr><th> Units </th><td> {u} </td> <td> {u} </td></tr>"
                new_table.append(newrow)
            new_table.append(row)
        return "\n".join(new_table)

    def to_dask(self):
        """
        convert to a plain dask array

        Returns
        -------
        dask.array object

        Examples
        --------

        >>> from unyt import dask_array
        >>> import dask.array as da
        >>> x = da.random.random((10000, 10000), chunks=(1000, 1000))
        >>> x_da = dask_array.unyt_from_dask(x, 'm')
        >>> x_da.to_dask()
        ... # doctest: +NORMALIZE_WHITESPACE
        dask.array<random_sample, shape=(10000, 10000), dtype=float64,
             chunksize=(1000, 1000), chunktype=numpy.ndarray>
        """
        (_, args) = super().__reduce__()
        return DaskArray(*args)

    def __reduce__(self):
        (_, args) = super().__reduce__()
        unyt_state = {
            "_unyt_array": self._unyt_array,
            "units": self.units,
            "unyt_name": self.unyt_name,
        }
        # reminder: the 3rd object returned by a __reduce__ tuple is the object
        # state. when there is no __setstate__, it must be a dict and it will be
        # added to the __dict__ attribute.
        return unyt_dask_array, args, unyt_state

    def __getattribute__(self, name):
        if name in _unyt_funcs_to_track:
            return self._track_conversion(name)

        result = super().__getattribute__(name)
        if name in _use_unary_decorator:
            return _unary_dask_decorator(result, self)

        return result

    def _track_conversion(self, unyt_func_name):
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
            init_val = self._unyt_array.value

            # get the unyt function handle and call it
            the_func = getattr(self._unyt_array, unyt_func_name)
            new_unyt_quantity = the_func(*args, **kwargs)

            # calculate the conversion factor and pull out the new name and units
            # might be able to use _get_conversion_factor here too...
            factor = new_unyt_quantity.value / init_val

            # apply the factor, return new
            new_unyt_quantity[:] = 1  # reset sidecar value
            new_obj = _create_with_quantity(self * factor, new_unyt_quantity)

            return new_obj

        # functools wrap fails here because unyt_func_name is a string, copy manually:
        wrapper.__doc__ = getattr(self._unyt_array, unyt_func_name).__doc__
        return wrapper

    def __dask_postcompute__(self):
        # a dask hook to catch after .compute(), see
        # https://docs.dask.org/en/latest/custom-collections.html#example-dask-collection
        return _finalize_unyt, ((self.units,))

    def _set_unit_state(self, units, new_unyt_array, unyt_name):
        # sets just the unit state of the object
        self.units = units
        self._unyt_array = new_unyt_array
        self.unyt_name = unyt_name

    # These methods bypass __getattribute__ and numpy hooks, so they are defined
    # explicitly here (but are handled generically by the _special_dec decorator).

    @_special_dec
    def __abs__(self):
        return super().__abs__()

    @_special_dec
    def __pow__(self, other):
        return super().__pow__(other)

    @_special_dec
    def __mul__(self, other):
        return super().__mul__(other)

    @_special_dec
    def __rmul__(self, other):
        return super().__rmul__(other)

    @_special_dec
    def __truediv__(self, other):
        return super().__truediv__(other)

    @_special_dec
    def __rtruediv__(self, other):
        return super().__rtruediv__(other)

    @_special_dec
    def __add__(self, other):
        return super().__add__(other)

    @_special_dec
    def __sub__(self, other):
        return super().__sub__(other)

    @_special_dec
    def __lt__(self, other):
        return super().__lt__(other)

    @_special_dec
    def __le__(self, other):
        return super().__le__(other)

    @_special_dec
    def __gt__(self, other):
        return super().__gt__(other)

    @_special_dec
    def __ge__(self, other):
        return super().__ge__(other)

    @_special_dec
    def __eq__(self, other):
        return super().__eq__(other)

    @_special_dec
    def __ne__(self, other):
        return super().__ne__(other)

    def prod(self, *args, **kwargs):
        _, unit = ua._apply_power_mapping(
            np.multiply, self._unyt_array.units, self.size, self.shape, kwargs
        )
        dask_result = super().prod(*args, **kwargs)
        return _create_with_quantity(dask_result, ua.unyt_array([1.0], unit))


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
    result = dask_finalize(results)

    if type(result) == np.ndarray:
        return ua.unyt_array(result, unit_name)
    else:
        return ua.unyt_quantity(result, unit_name)


def _create_with_quantity(dask_array, new_unyt_array):
    """
    this function instantiates a new unyt_dask_array instance and then sets
    the unit state, including units. Used to wrap dask operations

    Parameters
    ----------
    dask_array : a standard dask array or a unyt_dask_array
    new_unyt_array : a standard unyt array
    remaining arguments get passed to unyt.unyt_array, check there for a
    description.
    """
    if isinstance(dask_array, unyt_dask_array):
        dask_array = dask_array.to_dask()
    out = unyt_from_dask(dask_array)

    # attach the unyt_array
    units = new_unyt_array.units
    unyt_name = new_unyt_array.name

    out._set_unit_state(units, new_unyt_array, unyt_name)
    return out


def unyt_from_dask(
    dask_array,
    units=None,
    *,
    registry=None,
    bypass_validation=False,
    unyt_name=None,
):
    """
    creates a unyt_dask_array from a standard dask array. This function will
    create a new copy of the dask graph associated with an array.

    Parameters
    ----------
    dask_array : a standard dask array
    units : String unit name, unit symbol object, or astropy unit
        The units of the array. Powers must be specified using python
        syntax (cm**3, not cm^3).
    registry : :class:`unyt.unit_registry.UnitRegistry`
        The registry to create units from. If units is already associated
        with a unit registry and this is specified, this will be used instead
        of the registry associated with the unit object.
    bypass_validation : boolean
        If True, all input validation is skipped. Using this option may produce
        corrupted, invalid units or array data, but can lead to significant
        speedups in the input validation logic adds significant overhead. If
        set, units *must* be a valid unit object. Defaults to False.
    unyt_name : string
        The name of the array. Defaults to None. This attribute does not propagate
        through mathematical operations, but is preserved under indexing
        and unit conversions.

    Notes
    -----
        All of the above parameters with the exception of the dask_array are
        the same as for a standard unyt.unyt_array.

    Examples
    --------

    >>> from unyt import dask_array, unyt_quantity
    >>> import dask.array as da
    >>> x = da.ones((1000, 1000), chunks=(100, 100))
    >>> x_da = dask_array.unyt_from_dask(x, 'm')
    >>> type(x_da)
    <class 'unyt.dask_array.unyt_dask_array'>
    >>> x_da.units
    m
    >>> x_da.mean().units
    m
    >>> x_da.mean().compute()
    unyt_quantity(1., 'm')
    >>> x_da.to('cm').mean().compute()
    unyt_quantity(100., 'cm')
    >>> (x_da.to('cm')/unyt_quantity(10, 's')).mean().compute()
    unyt_quantity(10., 'cm/s')

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
        unyt_name=unyt_name,
    )

    return da


# note: the unyt_dask_array class has no way of catching daskified reductions.
# operations like dask.array.min() get routed through dask.array.reductions.min()
# and will return plain arrays or float/int values.
#
# When these operations exist as attributes, they can be called and will return
# unyt objects. i.e., :
#
# import dask; import unyt
# x_da = unyt_from_dask(dask.array.ones((10, 10), chunks=(2, 2)), unyt.m)
# dask.array.min(x_da).compute()  #  returns a plain float
# x_da.min().compute()  #  returns a unyt quantity
#
# but when the functions do not exist as attributes, like dask.array.nanmin(),
# it is difficult to handle without manually wrapping all of those reductions
# functions and exposing them here. The following function, reduce_with_units,
# is a compromise: it is a simple helper function for calling
# a general dask.array.reductions method on a unyt_dask array to correctly
# handle units.
#
# check https://github.com/yt-project/unyt/issues/269 for further discussion

_nan_ops = {"nansum", "nanmean", "nanmedian", "nanstd", "nanmax", "nanmin", "nancumsum"}
_passthrough_reductions = {"diagonal", "median"}
_allowed_funcs = _nan_ops.union(_passthrough_reductions, _use_unary_decorator)


def reduce_with_units(dask_func, unyt_dask_in, *args, **kwargs):
    """
    Call a dask.array.reduction function and preserve units.

    Parameters
    ----------
    dask_func : function
        a function handle from dask.array.reduction (e.g., dask.array.min,
        dask.array.var, dask.array.nanstd) to call.
    unyt_dask_in : unyt_dask_array
        the unyt dask array, first argument to dask_func
    args:
        any arguments to the dask_func
    kwargs:
        any keyword arguments to the dask_func

    Returns
    -------
    unyt_dask_array:
        the result of dask_func with units preserved

    Examples
    --------
    >>> import dask.array
    >>> from unyt.dask_array import unyt_from_dask, reduce_with_units
    >>> a = dask.array.ones((10000,), chunks=(100,))
    >>> a = unyt_from_dask(a, 'm')
    >>> b = reduce_with_units(dask.array.median, a, axis=0)
    >>> b.compute()
    unyt_quantity(1., 'm')

    """

    if dask_func.__name__ in _allowed_funcs:
        # e.g., min, max, nanstd. functions that return the same units can
        # be called directly as the dask function will treat unyt_dask_in as
        # a standard dask array and then we copy over the initial units.
        return _unary_dask_decorator(dask_func, unyt_dask_in)(
            unyt_dask_in, *args, **kwargs
        )
    else:
        # the operation may change the units
        npfunc = getattr(np, dask_func.__name__, None)
        if npfunc:
            newargs, unyt_result = _prep_ufunc(
                npfunc, unyt_dask_in, *args, extract_dask=True, **kwargs
            )
            return _post_ufunc(dask_func, unyt_result)(*newargs)
        else:
            raise ValueError("could not deduce np equivalent of dask reduction")
