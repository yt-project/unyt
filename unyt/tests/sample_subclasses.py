"""
Tests for support of subclasses of unyt_array and unyt_quantity.
"""

import warnings
from collections.abc import Callable, Collection, Iterable
from numbers import Number as numeric_type

import numpy as np

import unyt
from unyt import unyt_array, unyt_quantity
from unyt._array_functions import (
    allclose as unyt_allclose,
    around as unyt_around,
    array2string as unyt_array2string,
    array_equal as unyt_array_equal,
    array_equiv as unyt_array_equiv,
    array_repr as unyt_array_repr,
    block as unyt_block,
    choose as unyt_choose,
    clip as unyt_clip,
    column_stack as unyt_column_stack,
    concatenate as unyt_concatenate,
    convolve as unyt_convolve,
    copyto as unyt_copyto,
    correlate as unyt_correlate,
    cross as unyt_cross,
    diff as unyt_diff,
    dot as unyt_dot,
    dstack as unyt_dstack,
    ediff1d as unyt_ediff1d,
    einsum as unyt_einsum,
    fft_fft as unyt_fft_fft,
    fft_fft2 as unyt_fft_fft2,
    fft_fftn as unyt_fft_fftn,
    fft_fftshift as unyt_fft_fftshift,
    fft_hfft as unyt_fft_hfft,
    fft_ifft as unyt_fft_ifft,
    fft_ifft2 as unyt_fft_ifft2,
    fft_ifftn as unyt_fft_ifftn,
    fft_ifftshift as unyt_fft_ifftshift,
    fft_ihfft as unyt_fft_ihfft,
    fft_irfft as unyt_fft_irfft,
    fft_irfft2 as unyt_fft_irfft2,
    fft_irfftn as unyt_fft_irfftn,
    fft_rfft as unyt_fft_rfft,
    fft_rfft2 as unyt_fft_rfft2,
    fft_rfftn as unyt_fft_rfftn,
    fill_diagonal as unyt_fill_diagonal,
    geomspace as unyt_geomspace,
    histogram as unyt_histogram,
    histogram2d as unyt_histogram2d,
    histogram_bin_edges as unyt_histogram_bin_edges,
    histogramdd as unyt_histogramdd,
    hstack as unyt_hstack,
    inner as unyt_inner,
    insert as unyt_insert,
    interp as unyt_interp,
    intersect1d as unyt_intersect1d,
    isclose as unyt_isclose,
    isin as unyt_in1d,
    isin as unyt_isin,
    kron as unyt_kron,
    linalg_det as unyt_linalg_det,
    linalg_eig as unyt_linalg_eig,
    linalg_eigh as unyt_linalg_eigh,
    linalg_eigvals as unyt_linalg_eigvals,
    linalg_eigvalsh as unyt_linalg_eigvalsh,
    linalg_inv as unyt_linalg_inv,
    linalg_lstsq as unyt_linalg_lstsq,
    linalg_pinv as unyt_linalg_pinv,
    linalg_solve as unyt_linalg_solve,
    linalg_svd as unyt_linalg_svd,
    linalg_tensorinv as unyt_linalg_tensorinv,
    linalg_tensorsolve as unyt_linalg_tensorsolve,
    linspace as unyt_linspace,
    logspace as unyt_logspace,
    nanpercentile as unyt_nanpercentile,
    nanquantile as unyt_nanquantile,
    norm as unyt_linalg_norm,  # not linalg_norm, doesn't follow usual pattern
    outer as unyt_outer,
    pad as unyt_pad,
    percentile as unyt_percentile,
    place as unyt_place,
    prod as unyt_prod,
    ptp as unyt_ptp,
    put as unyt_put,
    put_along_axis as unyt_put_along_axis,
    putmask as unyt_putmask,
    quantile as unyt_quantile,
    savetxt as unyt_savetxt,
    searchsorted as unyt_searchsorted,
    select as unyt_select,
    setdiff1d as unyt_setdiff1d,
    sinc as unyt_sinc,
    sort_complex as unyt_sort_complex,
    stack as unyt_stack,
    take as unyt_take,
    tensordot as unyt_tensordot,
    trace as unyt_trace,
    trapezoid as unyt_trapezoid,
    tril as unyt_tril,
    triu as unyt_triu,
    union1d as unyt_union1d,
    unwrap as unyt_unwrap,
    var as unyt_var,
    vdot as unyt_vdot,
    vstack as unyt_vstack,
    where as unyt_where,
    _trapezoid_func,
)
from unyt.array import _iterable, multiple_output_operators
from packaging.version import Version
from importlib.metadata import version

NUMPY_VERSION = Version(version("numpy"))
if NUMPY_VERSION >= Version("2.0.0dev0"):
    from unyt._array_functions import linalg_outer as unyt_linalg_outer
if NUMPY_VERSION < Version("2.0.0dev0"):
    from unyt._array_functions import asfarray as unyt_asfarray

_HANDLED_FUNCTIONS = {}


class ExtraAttributeError(Exception):
    """
    Raised when function arguments have incompatible ``extra_attr``.
    """

    def __init__(self, message: str = "Tried to mix different extra_attr!") -> None:
        self.message = message


# first we define helper functions to handle repetitive operations in wrapping unyt &
# numpy functions (we will actually wrap the functions below):


def _copy_extra_attr_if_present(
    from_obj: object, to_obj: object, copy_units=False
) -> object:
    """
    Copy extra_attr across two objects.
    """
    if not (
        isinstance(to_obj, subclass_uarray) and isinstance(from_obj, subclass_uarray)
    ):
        return to_obj
    if copy_units:
        to_obj.units = from_obj.units
    if hasattr(from_obj, "extra_attr"):
        to_obj.extra_attr = from_obj.extra_attr
    return to_obj


def _propagate_extra_attr_to_result(func: Callable) -> Callable:
    """
    Wrapper that copies the extra attribute from first input argument to first
    output.

    Many functions take one input (or have a first input that has a close
    correspondence to the output) and one output. This helper copies the
    ``extra_attr`` attribute from the first input argument to the output. Can
    be used as a decorator on functions (the first argument is then the first
    argument of the function) or methods (the first argument is then ``self``).
    If the output is not a ``subclass_uarray`` it is not promoted (and then no
    attributes are copied).
    """

    def wrapped(obj, *args, **kwargs):
        return _copy_extra_attr_if_present(obj, func(obj, *args, **kwargs))

    return wrapped


def _prepare_array_func_args(*args, _default_cm: bool = True, **kwargs) -> dict:
    """
    For this minimalist class we check that the extra_attr matches and raise if not.
    """
    args_extra_attrs = [
        (hasattr(arg, "extra_attr"), getattr(arg, "extra_attr", None)) for arg in args
    ]
    kwarg_extra_attrs = {
        k: (hasattr(kwarg, "extra_attr"), getattr(kwarg, "extra_attr", None))
        for k, kwarg in kwargs.items()
    }
    extra_attr_values_where_present = [ea[1] for ea in args_extra_attrs if ea[0]] + [
        ea[1] for ea in kwarg_extra_attrs.values() if ea[0]
    ]
    # here we check that all of the extra_attr match (could be True, False or None):
    if not len(set(extra_attr_values_where_present)) <= 1:
        raise ExtraAttributeError
    # we could modify the args and kwargs before returning them here to "prepare" them
    return {
        "args": args,
        "kwargs": kwargs,
        "extra_attr": (
            extra_attr_values_where_present[0]
            if len(extra_attr_values_where_present) > 0
            else None
        ),
    }


def _return_helper(
    res: np.ndarray,
    helper_result: dict,
    out: np.ndarray | None = None,
) -> "subclass_uarray":
    """
    Helper function to attach our extra_attr to return values of wrapped
    functions.
    """
    res = _promote_unyt_to_subclass(res)
    if isinstance(res, subclass_uarray):  # also recognizes subclass_uquantity
        res.extra_attr = helper_result["extra_attr"]
    if isinstance(out, subclass_uarray):  # also recognizes subclass_uquantity
        out.extra_attr = helper_result["extra_attr"]
    return res


def _ensure_result_is_subclass_uarray_or_uquantity(func: Callable) -> Callable:
    """
    Wrapper that converts any ``unyt_array`` or ``unyt_quantity``
    instances in function output to subclass equivalents.

    If the wrapped function returns a ``tuple`` (as many numpy functions do) it
    is iterated over (but not recursively) and each element with a unyt class
    type is upgraded to its subclass equivalent. If anything but a ``tuple``
    is returned, that object is promoted to the subclass equivalent if it is
    of a unyt class type.
    """

    def wrapped(*args, **kwargs) -> object:
        # omit docstring so that sphinx picks up docstring of wrapped function
        result = func(*args, **kwargs)
        if isinstance(result, tuple):
            return tuple(
                _ensure_array_or_quantity_matches_shape(_promote_unyt_to_subclass(item))
                for item in result
            )
        else:
            return _ensure_array_or_quantity_matches_shape(
                _promote_unyt_to_subclass(result)
            )

    return wrapped


def _promote_unyt_to_subclass(input_object: object) -> object:
    """
    Upgrades the input unyt instance to its subclass equivalent.
    """
    if isinstance(input_object, unyt_quantity) and not isinstance(
        input_object, subclass_uquantity
    ):
        return input_object.view(subclass_uquantity)
    elif isinstance(input_object, unyt_array) and not isinstance(
        input_object, subclass_uarray
    ):
        return input_object.view(subclass_uarray)
    else:
        return input_object


def _ensure_array_or_quantity_matches_shape(input_object: object) -> object:
    """
    Convert scalars to ``subclass_uquantity`` and arrays to
    ``subclass_uarray``.
    """
    if (
        isinstance(input_object, subclass_uarray)
        and not isinstance(input_object, subclass_uquantity)
        and input_object.shape == ()
    ):
        return input_object.view(subclass_uquantity)
    elif isinstance(input_object, subclass_uquantity) and input_object.shape != ():
        return input_object.view(subclass_uarray)
    else:
        return input_object


def _default_unary_wrapper(unyt_func: Callable) -> Callable:
    """
    Wrapper helper for unary functions with typical behaviour.

    Can be used as a decorator.
    """

    def wrapper(*args, **kwargs):
        helper_result = _prepare_array_func_args(*args, **kwargs)
        res = unyt_func(*helper_result["args"], **helper_result["kwargs"])
        if "out" in kwargs:
            return _return_helper(res, helper_result, out=kwargs["out"])
        else:
            return _return_helper(res, helper_result)

    return wrapper


def _default_binary_wrapper(unyt_func: Callable) -> Callable:
    """
    Wrapper helper for binary functions with typical behaviour.

    Can be used as a decorator.
    """

    def wrapper(*args, **kwargs):
        helper_result = _prepare_array_func_args(*args, **kwargs)
        res = unyt_func(*helper_result["args"], **helper_result["kwargs"])
        if "out" in kwargs:
            return _return_helper(res, helper_result, out=kwargs["out"])
        else:
            return _return_helper(res, helper_result)

    return wrapper


# here an alias, but often comparisons have their own particular default logic:
_default_comparison_wrapper = _default_binary_wrapper


def _default_oplist_wrapper(unyt_func: Callable) -> Callable:
    """
    Wrapper helper for functions accepting a list of operands with typical behaviour.

    Can be used as a decorator.
    """

    def wrapper(*args, **kwargs):
        helper_result = _prepare_array_func_args(*args, **kwargs)
        helper_result_oplist = _prepare_array_func_args(*args[0])
        res = unyt_func(
            helper_result_oplist["args"],
            *helper_result["args"][1:],
            **helper_result["kwargs"],
        )
        return _return_helper(res, helper_result_oplist)

    return wrapper


def implements(numpy_function: Callable) -> Callable:
    """
    Register an __array_function__ implementation for subclass_uarray objects.

    Intended for use as a decorator.
    """

    # See NEP 18 https://numpy.org/neps/nep-0018-array-function-protocol.html
    def decorator(func: Callable) -> Callable:
        """
        Actually register the specified function.
        """
        _HANDLED_FUNCTIONS[numpy_function] = func
        return func

    return decorator


# Next we wrap functions from unyt and numpy. There's not much point in writing docstrings
# or type hints for all of these.

# Now we wrap functions that unyt handles explicitly (below that will be those not handled
# explicitly):


@implements(np.array2string)
def array2string(a, *args, **kwargs):
    res = unyt_array2string(a, *args, **kwargs)
    return res + f" extra_attr={a.extra_attr}"


implements(np.dot)(_default_binary_wrapper(unyt_dot))
implements(np.vdot)(_default_binary_wrapper(unyt_vdot))
implements(np.inner)(_default_binary_wrapper(unyt_inner))
implements(np.outer)(_default_binary_wrapper(unyt_outer))
implements(np.kron)(_default_binary_wrapper(unyt_kron))
implements(np.histogram_bin_edges)(_default_unary_wrapper(unyt_histogram_bin_edges))
implements(np.linalg.inv)(_default_unary_wrapper(unyt_linalg_inv))
implements(np.linalg.tensorinv)(_default_unary_wrapper(unyt_linalg_tensorinv))
implements(np.linalg.pinv)(_default_unary_wrapper(unyt_linalg_pinv))
implements(np.linalg.svd)(_default_unary_wrapper(unyt_linalg_svd))


def _histogram(a, bins=10, range=None, density=None, weights=None, normed=None):
    helper_result = _prepare_array_func_args(
        a, bins=bins, range=range, density=density, weights=weights, normed=normed
    )
    kwargs = {
        "bins": helper_result["kwargs"]["bins"],
        "range": helper_result["kwargs"]["range"],
        "density": helper_result["kwargs"]["density"],
        "weights": helper_result["kwargs"]["weights"],
    }
    if NUMPY_VERSION < Version("1.24"):
        kwargs["normed"] = helper_result["kwargs"]["normed"]
    counts, bins = unyt_histogram(*helper_result["args"], **kwargs)
    counts = _promote_unyt_to_subclass(counts)
    if isinstance(counts, subclass_uarray):  # also recognizes subclass_uquantity
        counts.extra_attr = helper_result["extra_attr"]
    return counts, _return_helper(bins, helper_result)


if NUMPY_VERSION >= Version("1.24"):

    @implements(np.histogram)
    def histogram(a, bins=10, range=None, density=None, weights=None):
        return _histogram(a, bins=bins, range=range, density=density, weights=weights)

else:

    @implements(np.histogram)
    def histogram(a, bins=10, range=None, normed=None, density=None, weights=None):
        return _histogram(a, bins=bins, range=range, density=density, weights=weights)


def _histogram2d(x, y, *, bins=10, range=None, density=None, weights=None, normed=None):
    if range is not None:
        xrange, yrange = range
    else:
        xrange, yrange = None, None

    try:
        N = len(bins)
    except TypeError:
        N = 1
    if N != 2:
        xbins = ybins = bins
    elif N == 2:
        xbins, ybins = bins
    helper_result_x = _prepare_array_func_args(x, bins=xbins, range=xrange)
    helper_result_y = _prepare_array_func_args(y, bins=ybins, range=yrange)
    if not density:
        helper_result_w = _prepare_array_func_args(weights=weights, normed=normed)
        if (helper_result_x["kwargs"]["range"] is None) and (
            helper_result_y["kwargs"]["range"] is None
        ):
            safe_range = None
        else:
            safe_range = (
                helper_result_x["kwargs"]["range"],
                helper_result_y["kwargs"]["range"],
            )
        kwargs = {
            "bins": (
                helper_result_x["kwargs"]["bins"],
                helper_result_y["kwargs"]["bins"],
            ),
            "range": safe_range,
            "density": density,
            "weights": helper_result_w["kwargs"]["weights"],
        }
        if NUMPY_VERSION < Version("1.24"):
            kwargs["normed"] = helper_result_w["kwargs"]["normed"]
        counts, xbins, ybins = unyt_histogram2d(
            helper_result_x["args"][0],
            helper_result_y["args"][0],
            **kwargs,
        )
        if weights is not None:
            counts = _promote_unyt_to_subclass(counts)
            if isinstance(
                counts, subclass_uarray
            ):  # also recognizes subclass_uquantity
                counts.extra_attr = helper_result_w["extra_attr"]
    else:  # density=True
        # now x, y and weights must be compatible because they will combine
        # we unpack input to the helper to get everything checked for compatibility
        helper_result = _prepare_array_func_args(
            x,
            y,
            xbins=xbins,
            ybins=ybins,
            xrange=xrange,
            yrange=yrange,
            weights=weights,
            normed=normed,
        )
        if (helper_result["kwargs"]["xrange"] is None) and (
            helper_result["kwargs"]["yrange"] is None
        ):
            safe_range = None
        else:
            safe_range = (
                helper_result["kwargs"]["xrange"],
                helper_result["kwargs"]["yrange"],
            )
        kwargs = {
            "bins": (
                helper_result["kwargs"]["xbins"],
                helper_result["kwargs"]["ybins"],
            ),
            "range": safe_range,
            "density": density,
            "weights": helper_result["kwargs"]["weights"],
        }
        if NUMPY_VERSION < Version("1.24"):
            kwargs["normed"] = helper_result["kwargs"]["normed"]
        counts, xbins, ybins = unyt_histogram2d(
            helper_result["args"][0],
            helper_result["args"][1],
            **kwargs
        )
        counts = _promote_unyt_to_subclass(counts)
        if isinstance(counts, subclass_uarray):  # also recognizes subclass_uquantity
            counts.extra_attr = helper_result["extra_attr"]
    return (
        counts,
        _return_helper(xbins, helper_result_x),
        _return_helper(ybins, helper_result_y),
    )


if NUMPY_VERSION >= Version("1.24"):

    @implements(np.histogram2d)
    def histogram2d(x, y, bins=10, range=None, density=None, weights=None):
        return _histogram2d(
            x, y, bins=bins, range=range, density=density, weights=weights
        )

else:

    @implements(np.histogram2d)
    def histogram2d(x, y, bins=10, range=None, normed=None, weights=None, density=None):
        return _histogram2d(
            x,
            y,
            bins=bins,
            range=range,
            normed=normed,
            weights=weights,
            density=density,
        )


def _histogramdd(sample, bins=10, range=None, density=None, weights=None, normed=None):
    D = len(sample)
    if range is not None:
        ranges = range
    else:
        ranges = D * [None]

    try:
        len(bins)
    except TypeError:
        # bins is an integer
        bins = D * [bins]
    helper_results = [
        _prepare_array_func_args(s, bins=b, range=r)
        for s, b, r in zip(sample, bins, ranges)
    ]
    if not density:
        helper_result_w = _prepare_array_func_args(weights=weights)
        all_ranges = [
            helper_result["kwargs"]["range"] for helper_result in helper_results
        ]
        if set(all_ranges) == {None}:
            safe_range = None
        else:
            safe_range = [
                helper_result["kwargs"]["range"] for helper_result in helper_results
            ]
        kwargs = {
            "bins": [
                helper_result["kwargs"]["bins"] for helper_result in helper_results
            ],
            "range": safe_range,
            "density": density,
            "weights": helper_result_w["kwargs"]["weights"],
        }
        if NUMPY_VERSION < Version("1.24"):
            kwargs["normed"] = normed
        counts, bins = unyt_histogramdd(
            [helper_result["args"][0] for helper_result in helper_results],
            **kwargs,
        )
        if weights is not None:
            counts = _promote_unyt_to_subclass(counts)
            if isinstance(counts, subclass_uarray):
                counts.extra_attr = helper_result_w["extra_attr"]
    else:  # density=True
        # now sample and weights must be compatible because they will combine
        # we unpack input to the helper to get everything checked for compatibility
        helper_result = _prepare_array_func_args(
            *sample, bins=bins, range=range, weights=weights
        )
        kwargs = {
            "bins": helper_result["kwargs"]["bins"],
            "range": helper_result["kwargs"]["range"],
            "density": density,
            "weights": helper_result["kwargs"]["weights"],
        }
        if NUMPY_VERSION < Version("1.24"):
            kwargs["normed"] = normed
        counts, bins = unyt_histogramdd(
            helper_result["args"],
            **kwargs,
        )
        counts = _promote_unyt_to_subclass(counts)
        if isinstance(counts, subclass_uarray):  # also recognizes subclass_uquantity
            counts.extra_attr = helper_result["extra_attr"]
    return (
        counts,
        tuple(
            _return_helper(b, helper_result)
            for b, helper_result in zip(bins, helper_results)
        ),
    )


if NUMPY_VERSION >= Version("1.24"):

    @implements(np.histogramdd)
    def histogramdd(sample, bins=10, range=None, density=None, weights=None):
        return _histogramdd(
            sample, bins=bins, range=range, density=density, weights=weights
        )

else:

    @implements(np.histogramdd)
    def histogramdd(
        sample, bins=10, range=None, normed=None, weights=None, density=None
    ):
        return _histogramdd(
            sample,
            bins=bins,
            range=range,
            normed=normed,
            weights=weights,
            density=density,
        )

implements(np.concatenate)(_default_oplist_wrapper(unyt_concatenate))
implements(np.cross)(_default_binary_wrapper(unyt_cross))
implements(np.intersect1d)(_default_binary_wrapper(unyt_intersect1d))
implements(np.union1d)(_default_binary_wrapper(unyt_union1d))
implements(np.linalg.norm)(_default_unary_wrapper(unyt_linalg_norm))
implements(np.vstack)(_default_oplist_wrapper(unyt_vstack))
implements(np.hstack)(_default_oplist_wrapper(unyt_hstack))
implements(np.dstack)(_default_oplist_wrapper(unyt_dstack))
implements(np.column_stack)(_default_oplist_wrapper(unyt_column_stack))
implements(np.stack)(_default_oplist_wrapper(unyt_stack))
implements(np.around)(_default_unary_wrapper(unyt_around))


def _prepare_array_block_args(lst, recursing=False):
    """
    Block accepts only a nested list of array "blocks". We need to recurse on this.
    """
    helper_results = []
    if isinstance(lst, list):
        for item in lst:
            if isinstance(item, list):
                helper_results += _prepare_array_block_args(item, recursing=True)
            else:
                helper_results.append(_prepare_array_func_args(item))
    if recursing:
        return helper_results
    eas = [hr["extra_attr"] for hr in helper_results]
    if set(eas) == {True}:
        ret_ea = True
    elif set(eas) == {None}:
        ret_ea = None
    elif set(eas) == {False}:
        ret_ea = False
    else:
        # mixed values
        raise ExtraAttributeError
    ret_lst = lst
    return {
        "args": ret_lst,
        "kwargs": {},
        "extra_attr": ret_ea,
    }


@implements(np.block)
def block(arrays):
    # block is a special case since we need to recurse more than one level
    # down the list of arrays.
    helper_result_block = _prepare_array_block_args(arrays)
    res = unyt_block(helper_result_block["args"])
    return _return_helper(res, helper_result_block)


implements(np.fft.fft)(_default_unary_wrapper(unyt_fft_fft))
implements(np.fft.fft2)(_default_unary_wrapper(unyt_fft_fft2))
implements(np.fft.fftn)(_default_unary_wrapper(unyt_fft_fftn))
implements(np.fft.hfft)(_default_unary_wrapper(unyt_fft_hfft))
implements(np.fft.rfft)(_default_unary_wrapper(unyt_fft_rfft))
implements(np.fft.rfft2)(_default_unary_wrapper(unyt_fft_rfft2))
implements(np.fft.rfftn)(_default_unary_wrapper(unyt_fft_rfftn))
implements(np.fft.ifft)(_default_unary_wrapper(unyt_fft_ifft))
implements(np.fft.ifft2)(_default_unary_wrapper(unyt_fft_ifft2))
implements(np.fft.ifftn)(_default_unary_wrapper(unyt_fft_ifftn))
implements(np.fft.ihfft)(_default_unary_wrapper(unyt_fft_ihfft))
implements(np.fft.irfft)(_default_unary_wrapper(unyt_fft_irfft))
implements(np.fft.irfft2)(_default_unary_wrapper(unyt_fft_irfft2))
implements(np.fft.irfftn)(_default_unary_wrapper(unyt_fft_irfftn))
implements(np.fft.fftshift)(_default_unary_wrapper(unyt_fft_fftshift))
implements(np.fft.ifftshift)(_default_unary_wrapper(unyt_fft_ifftshift))
implements(np.sort_complex)(_default_unary_wrapper(unyt_sort_complex))
implements(np.isclose)(_default_comparison_wrapper(unyt_isclose))
implements(np.allclose)(_default_comparison_wrapper(unyt_allclose))
implements(np.array_equal)(_default_comparison_wrapper(unyt_array_equal))
implements(np.array_equiv)(_default_comparison_wrapper(unyt_array_equiv))
implements(np.linspace)(_default_binary_wrapper(unyt_linspace))
implements(np.logspace)(_default_binary_wrapper(unyt_logspace))
implements(np.geomspace)(_default_binary_wrapper(unyt_geomspace))
implements(np.copyto)(_default_binary_wrapper(unyt_copyto))
implements(np.prod)(_default_unary_wrapper(unyt_prod))
implements(np.var)(_default_unary_wrapper(unyt_var))
implements(np.trace)(_default_unary_wrapper(unyt_trace))
implements(np.percentile)(_default_unary_wrapper(unyt_percentile))
implements(np.quantile)(_default_unary_wrapper(unyt_quantile))
implements(np.nanpercentile)(_default_unary_wrapper(unyt_nanpercentile))
implements(np.nanquantile)(_default_unary_wrapper(unyt_nanquantile))
implements(np.linalg.det)(_default_unary_wrapper(unyt_linalg_det))
implements(np.diff)(_default_unary_wrapper(unyt_diff))
implements(np.ediff1d)(_default_unary_wrapper(unyt_ediff1d))
implements(np.ptp)(_default_unary_wrapper(unyt_ptp))
# implements(np.cumprod)(...) Omitted because unyt just raises if called.
# implements(np.cumulative_prod)(...) Omitted because unyt just raises if called.
implements(np.pad)(_default_unary_wrapper(unyt_pad))
implements(np.choose)(_default_binary_wrapper(unyt_choose))
implements(np.insert)(_default_unary_wrapper(unyt_insert))


@implements(np.linalg.lstsq)
def linalg_lstsq(a, b, rcond=None):
    helper_result = _prepare_array_func_args(a, b, rcond=rcond)
    ress = unyt_linalg_lstsq(*helper_result["args"], **helper_result["kwargs"])
    return (
        _return_helper(ress[0], helper_result),
        _return_helper(ress[1], helper_result),
        ress[2],
        _return_helper(ress[3], helper_result),
    )


implements(np.linalg.solve)(_default_binary_wrapper(unyt_linalg_solve))
implements(np.linalg.tensorsolve)(_default_binary_wrapper(unyt_linalg_tensorsolve))


@implements(np.linalg.eig)
def linalg_eig(a):
    helper_result = _prepare_array_func_args(a)
    ress = unyt_linalg_eig(*helper_result["args"], **helper_result["kwargs"])
    return (_return_helper(ress[0], helper_result), ress[1])


@implements(np.linalg.eigh)
def linalg_eigh(a, UPLO="L"):
    helper_result = _prepare_array_func_args(a, UPLO=UPLO)
    ress = unyt_linalg_eigh(*helper_result["args"], **helper_result["kwargs"])
    return (_return_helper(ress[0], helper_result), ress[1])


implements(np.linalg.eigvals)(_default_unary_wrapper(unyt_linalg_eigvals))
implements(np.linalg.eigvalsh)(_default_unary_wrapper(unyt_linalg_eigvalsh))


@implements(np.savetxt)
def savetxt(
    fname,
    X,
    fmt="%.18e",
    delimiter=" ",
    newline="\n",
    header="",
    footer="",
    comments="# ",
    encoding=None,
):
    warnings.warn(
        "numpy.savetxt does not preserve units or extra_attr information, "
        "and will only save the raw numerical data from the subclass_uarray object.\n"
        "If this is the intended behaviour, call `numpy.savetxt(file, arr.d)` "
        "to silence this warning.\n",
        stacklevel=4,
    )
    helper_result = _prepare_array_func_args(
        fname,
        X,
        fmt=fmt,
        delimiter=delimiter,
        newline=newline,
        header=header,
        footer=footer,
        comments=comments,
        encoding=encoding,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="numpy.savetxt does not preserve units",
        )
        unyt_savetxt(*helper_result["args"], **helper_result["kwargs"])
    return


@implements(np.apply_over_axes)
def apply_over_axes(func, a, axes):
    res = func(a, axes[0])
    if len(axes) > 1:
        # this function is recursive by nature,
        # here we intentionally do not call the base _implementation
        return np.apply_over_axes(func, res, axes[1:])
    else:
        return res


implements(np.fill_diagonal)(_default_binary_wrapper(unyt_fill_diagonal))
implements(np.isin)(_default_comparison_wrapper(unyt_isin))
implements(np.place)(_default_unary_wrapper(unyt_place))
implements(np.put)(_default_unary_wrapper(unyt_put))
implements(np.put_along_axis)(_default_unary_wrapper(unyt_put_along_axis))
implements(np.putmask)(_default_unary_wrapper(unyt_putmask))
implements(np.searchsorted)(_default_binary_wrapper(unyt_searchsorted))


@implements(np.select)
def select(condlist, choicelist, default=0):
    helper_result = _prepare_array_func_args(condlist, choicelist, default=default)
    helper_result_choicelist = _prepare_array_func_args(*choicelist)
    res = unyt_select(
        helper_result["args"][0],
        helper_result_choicelist["args"],
        **helper_result["kwargs"],
    )
    return _return_helper(res, helper_result)


implements(np.setdiff1d)(_default_binary_wrapper(unyt_setdiff1d))
implements(np.sinc)(_default_unary_wrapper(unyt_sinc))
implements(np.clip)(_default_unary_wrapper(unyt_clip))


@implements(np.where)
def where(condition, *args):
    helper_result = _prepare_array_func_args(condition, *args)
    if len(args) == 0:  # just condition
        res = unyt_where(*helper_result["args"], **helper_result["kwargs"])
    elif len(args) < 2:
        # error message borrowed from numpy 1.24.1
        raise ValueError("either both or neither of x and y should be given")
    res = unyt_where(*helper_result["args"], **helper_result["kwargs"])
    return _return_helper(res, helper_result)


implements(np.triu)(_default_unary_wrapper(unyt_triu))
implements(np.tril)(_default_unary_wrapper(unyt_tril))


@implements(np.einsum)
def einsum(
    subscripts,
    *operands,
    out=None,
    dtype=None,
    order="K",
    casting="safe",
    optimize=False,
):
    helper_result = _prepare_array_func_args(
        subscripts,
        operands,
        out=out,
        dtype=dtype,
        order=order,
        casting=casting,
        optimize=optimize,
    )
    helper_result_operands = _prepare_array_func_args(*operands)
    res = unyt_einsum(
        helper_result["args"][0],
        *helper_result_operands["args"],
        **helper_result["kwargs"],
    )
    return _return_helper(res, helper_result_operands, out=out)


implements(np.convolve)(_default_binary_wrapper(unyt_convolve))
implements(np.correlate)(_default_binary_wrapper(unyt_correlate))
implements(np.tensordot)(_default_binary_wrapper(unyt_tensordot))
implements(np.unwrap)(_default_unary_wrapper(unyt_unwrap))
implements(np.interp)(_default_unary_wrapper(unyt_interp))


@implements(np.array_repr)
def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
    helper_result = _prepare_array_func_args(
        arr,
        max_line_width=max_line_width,
        precision=precision,
        suppress_small=suppress_small,
    )
    rep = unyt_array_repr(*helper_result["args"], **helper_result["kwargs"])[:-1]
    if hasattr(arr, "extra_attr"):
        rep += f", extra_attr='{arr.extra_attr}'"
    rep += ")"
    return rep


if NUMPY_VERSION < Version("2.0.0dev0"):
    implements(np.asfarray)(_default_unary_wrapper(unyt_asfarray))
if NUMPY_VERSION >= Version("2.0.0dev0"):
    implements(np.linalg.outer)(_default_binary_wrapper(unyt_linalg_outer))
implements(_trapezoid_func)(_default_binary_wrapper(unyt_trapezoid))
implements(np.isin)(_default_comparison_wrapper(unyt_in1d))
implements(np.take)(_default_unary_wrapper(unyt_take))

# Now we wrap functions that unyt does not handle explicitly:
implements(np.average)(_propagate_extra_attr_to_result(np.average._implementation))
implements(np.max)(_propagate_extra_attr_to_result(np.max._implementation))
implements(np.min)(_propagate_extra_attr_to_result(np.min._implementation))
implements(np.mean)(_propagate_extra_attr_to_result(np.mean._implementation))
implements(np.median)(_propagate_extra_attr_to_result(np.median._implementation))
implements(np.sort)(_propagate_extra_attr_to_result(np.sort._implementation))
implements(np.sum)(_propagate_extra_attr_to_result(np.sum._implementation))
implements(np.partition)(_propagate_extra_attr_to_result(np.partition._implementation))
if NUMPY_VERSION >= Version("2.0.0dev0"):
    implements(np.linalg.cross)(_default_binary_wrapper(np.linalg.cross._implementation))


@implements(np.meshgrid)
def meshgrid(*xi, **kwargs):
    # meshgrid is a unique case: arguments never interact with each other, so we don't
    # want to use our _prepare_array_func_args helper (that will try to enforce
    # consistency).
    # However we can't just use _propagate_extra_attr_to_result because we
    # need to iterate over arguments.
    res = np.meshgrid._implementation(*xi, **kwargs)
    ret_type = tuple if NUMPY_VERSION >= Version("2.0.0dev0") else list
    return ret_type(_copy_extra_attr_if_present(x, r) for (x, r) in zip(xi, res))


class subclass_uarray(unyt_array):
    """
    Minimalist subclass of unyt_array.
    """

    def __new__(
        cls,
        input_array: Iterable,
        units: str | unyt.unit_object.Unit = None,
        *,
        registry: unyt.unit_registry.UnitRegistry = None,
        dtype: np.dtype | str = None,
        bypass_validation: bool = False,
        name: str = None,
        extra_attr: bool = None,
    ) -> "subclass_uarray":
        """
        Closely inspired by the :meth:`unyt.array.unyt_array.__new__` constructor.
        """

        if bypass_validation is True:
            obj = super().__new__(
                cls,
                input_array,
                units=units,
                registry=registry,
                dtype=dtype,
                bypass_validation=bypass_validation,
                name=name,
            )

            # dtype, units, registry & name are handled by unyt
            obj.extra_attr = extra_attr

            return obj

        if isinstance(input_array, subclass_uarray):
            if input_array.extra_attr != extra_attr:
                raise ValueError
            obj = input_array.view(cls)

            return obj

        elif isinstance(input_array, np.ndarray) and input_array.dtype != object:
            # Guard np.ndarray so it doesn't get caught by _iterable in next
            # case. ndarray with object dtype goes to next case to properly
            # handle e.g. ndarrays containing subclass_uquantity's
            pass

        elif _iterable(input_array) and input_array:
            # check compatibility:
            helper_result = _prepare_array_func_args(*input_array)
            input_array = helper_result["args"]

            extra_attr = (
                helper_result["extra_attr"] if extra_attr is None else extra_attr
            )

        obj = super().__new__(
            cls,
            input_array,
            units=units,
            registry=registry,
            dtype=dtype,
            bypass_validation=bypass_validation,
            name=name,
        )

        if isinstance(obj, unyt_array) and not isinstance(obj, cls):
            obj = obj.view(cls)

        # attach our attribute:
        obj.extra_attr = extra_attr

        return obj

    def __array_finalize__(self, obj: "subclass_uarray") -> None:
        super().__array_finalize__(obj)
        if obj is None:
            return
        self.extra_attr = getattr(obj, "extra_attr", None)

    def __str__(self) -> str:
        return super().__str__() + f" {self.extra_attr}"

    def __repr__(self) -> str:
        return super().__repr__() + f" {self.extra_attr}"

    def __reduce__(self) -> tuple:
        """
        Pickle reduction method.

        Here we add an extra element at the start of the state tuple to store
        the extra_attr.
        """
        np_ret = super().__reduce__()
        obj_state = np_ret[2]
        sub_state = (((self.extra_attr,),) + obj_state[:],)
        new_ret = np_ret[:2] + sub_state + np_ret[3:]
        return new_ret

    def __setstate__(self, state: tuple) -> None:
        """
        Pickle setstate method.

        Here we extract the extra info we added to the object
        state and pass the rest to :meth:`unyt.array.unyt_array.__setstate__`.
        """
        super().__setstate__(state[1:])
        (self.extra_attr,) = state[0]

    # Wrap functions that return copies of subclass_uarrays so that our
    # attribute gets passed through:
    astype = _propagate_extra_attr_to_result(unyt_array.astype)
    in_units = _propagate_extra_attr_to_result(unyt_array.in_units)
    byteswap = _propagate_extra_attr_to_result(unyt_array.byteswap)
    compress = _propagate_extra_attr_to_result(unyt_array.compress)
    diagonal = _propagate_extra_attr_to_result(unyt_array.diagonal)
    flatten = _propagate_extra_attr_to_result(unyt_array.flatten)
    ravel = _propagate_extra_attr_to_result(unyt_array.ravel)
    repeat = _propagate_extra_attr_to_result(unyt_array.repeat)
    swapaxes = _propagate_extra_attr_to_result(unyt_array.swapaxes)
    transpose = _propagate_extra_attr_to_result(unyt_array.transpose)
    view = _propagate_extra_attr_to_result(unyt_array.view)
    __copy__ = _propagate_extra_attr_to_result(unyt_array.__copy__)
    __deepcopy__ = _propagate_extra_attr_to_result(unyt_array.__deepcopy__)
    in_cgs = _propagate_extra_attr_to_result(unyt_array.in_cgs)
    take = _propagate_extra_attr_to_result(
        _ensure_result_is_subclass_uarray_or_uquantity(unyt_array.take)
    )
    reshape = _propagate_extra_attr_to_result(
        _ensure_result_is_subclass_uarray_or_uquantity(unyt_array.reshape)
    )
    __getitem__ = _propagate_extra_attr_to_result(
        _ensure_result_is_subclass_uarray_or_uquantity(unyt_array.__getitem__)
    )
    dot = _default_binary_wrapper(unyt_array.dot)

    # Also wrap some array "properties":
    T = property(_propagate_extra_attr_to_result(unyt_array.transpose))
    ua = property(_propagate_extra_attr_to_result(np.ones_like))
    unit_array = property(_propagate_extra_attr_to_result(np.ones_like))

    @classmethod
    def __unyt_ufunc_prepare__(cls, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        helper_result = _prepare_array_func_args(*inputs, **kwargs)
        return ufunc, method, helper_result["args"], helper_result["kwargs"]

    @classmethod
    def __unyt_ufunc_finalize__(
        cls, result, ufunc: np.ufunc, method: str, *inputs, **kwargs
    ):
        helper_result = _prepare_array_func_args(*inputs, **kwargs)
        # if we get a tuple we have multiple return values to deal with
        if isinstance(result, tuple):
            r = tuple(
                (
                    r.view(subclass_uquantity)
                    if r.shape == ()
                    else (
                        r.view(subclass_uarray)
                        if isinstance(r, unyt_array)
                        and not isinstance(r, subclass_uarray)
                        else r
                    )
                )
                for r in result
            )
            for r in result:
                if isinstance(r, subclass_uarray):  # also recognizes subclass_uquantity
                    r.extra_attr = helper_result["extra_attr"]
        elif isinstance(result, unyt_array):  # also recognizes subclass_uquantity
            if not isinstance(result, subclass_uarray):
                result = (
                    result.view(subclass_uquantity)
                    if result.shape == ()
                    else result.view(subclass_uarray)
                )
            result.extra_attr = helper_result["extra_attr"]
        if "out" in kwargs:
            out = kwargs.pop("out")
            if ufunc not in multiple_output_operators:
                out = out[0]
                if isinstance(out, unyt_array) and not isinstance(out, subclass_uarray):
                    out = (
                        out.view(subclass_uquantity)
                        if out.shape == ()
                        else out.view(subclass_uarray)
                    )
                if isinstance(
                    out, subclass_uarray
                ):  # also recognizes subclass_uquantity
                    out.extra_attr = helper_result["extra_attr"]
            else:
                out = tuple(
                    (
                        (
                            o.view(subclass_uquantity)
                            if o.shape == ()
                            else o.view(subclass_uarray)
                        )
                        if isinstance(o, unyt_array)
                        and not isinstance(o, subclass_uarray)
                        else o
                    )
                    for o in out
                )
                for o in out:
                    if isinstance(
                        o, subclass_uarray
                    ):  # also recognizes subclass_uquantity
                        o.extra_attr = helper_result["extra_attr"]
        return result

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs, **kwargs
    ) -> object:
        """
        Handles :mod:`numpy` ufunc calls on ``subclass_uarray`` input.

        ``numpy`` facilitates wrapping array classes by handing off to this
        function when a function of :class:`numpy.ufunc` type is called with
        arguments from an inheriting array class. Since we inherit from
        ``unyt_array`, we let it handle what to do with the units and
        take care of processing the extra_attr via our helper functions.
        """
        helper_result = _prepare_array_func_args(*inputs, **kwargs)

        result = _ensure_result_is_subclass_uarray_or_uquantity(
            super().__array_ufunc__
        )(ufunc, method, *inputs, **kwargs)
        # if we get a tuple we have multiple return values to deal with
        if isinstance(result, tuple):
            for r in result:
                if isinstance(r, subclass_uarray):  # also recognizes subclass_uquantity
                    r.extra_attr = helper_result["extra_attr"]
        elif isinstance(result, subclass_uarray):  # also recognizes subclass_uquantity
            result.extra_attr = helper_result["extra_attr"]
        if "out" in kwargs:
            out = kwargs.pop("out")
            if ufunc not in multiple_output_operators:
                out = out[0]
                if isinstance(
                    out, subclass_uarray
                ):  # also recognizes subclass_uquantity
                    out.extra_attr = helper_result["extra_attr"]
            else:
                for o in out:
                    if isinstance(
                        o, subclass_uarray
                    ):  # also recognizes subclass_uquantity
                        o.extra_attr = helper_result["extra_attr"]

        return result

    def __array_function__(
        self, func: Callable, types: Collection, args: tuple, kwargs: dict
    ):
        """
        Handles ``numpy`` function calls on ``subclass_uarray`` input.

        ``numpy`` facilitates wrapping array classes by handing off to this
        function when a numpy-defined function is called with arguments
        from an inheriting array class. Since we inherit from ``unyt_array``,
        we let it handle what to do with the units and take care of processing
        the extra_attr via our helper functions.
        """
        # Follow NEP 18 guidelines
        # https://numpy.org/neps/nep-0018-array-function-protocol.html
        from unyt._array_functions import (
            _HANDLED_FUNCTIONS as _UNYT_HANDLED_FUNCTIONS,
            _UNSUPPORTED_FUNCTIONS as _UNYT_UNSUPPORTED_FUNCTIONS,
        )

        # Let's claim to support everything supported by unyt.
        # If we can't do this in future, follow their pattern of
        # defining out own _UNSUPPORTED_FUNCTIONS in a _array_functions.py file
        _UNSUPPORTED_FUNCTIONS = _UNYT_UNSUPPORTED_FUNCTIONS

        if func in _UNSUPPORTED_FUNCTIONS:
            # following NEP 18, return NotImplemented as a sentinel value
            # which will lead to raising a TypeError, while
            # leaving other arguments a chance to take the lead
            return NotImplemented

        if not all(issubclass(t, subclass_uarray) or t is np.ndarray for t in types):
            # Note: this allows subclasses that don't override
            # __array_function__ to handle subclass_uarray objects
            return NotImplemented

        if func in _HANDLED_FUNCTIONS:
            function_to_invoke = _HANDLED_FUNCTIONS[func]
        elif func in _UNYT_HANDLED_FUNCTIONS:
            function_to_invoke = _UNYT_HANDLED_FUNCTIONS[func]
        else:
            # default to numpy's private implementation
            function_to_invoke = func._implementation
        return function_to_invoke(*args, **kwargs)

    def __mul__(
        self, b: int | float | np.ndarray | unyt.unit_object.Unit
    ) -> "subclass_uarray":
        """
        Multiply this ``subclass_uarray``.

        We delegate most cases to ``unyt_array``, but we need to handle
        the case where the second argument is a ``Unit``.
        """
        if getattr(b, "is_Unit", False):
            return _copy_extra_attr_if_present(
                self,
                _ensure_result_is_subclass_uarray_or_uquantity(b.__mul__)(
                    self.view(unyt_quantity)
                    if self.shape == ()
                    else self.view(unyt_array)
                ),
            )
        else:
            return super().__mul__(b)

    def __rmul__(
        self, b: int | float | np.ndarray | unyt.unit_object.Unit
    ) -> "subclass_uarray":
        """
        Multiply this ``subclass_uarray`` (as the right argument).

        We delegate most cases to ``unyt_array``, but we need to handle
        the case where the second argument is a ``Unit``.
        """
        if getattr(b, "is_Unit", False):
            return self.__mul__(b)
        else:
            return super().__rmul__(b)


class subclass_uquantity(subclass_uarray, unyt_quantity):
    """
    Minimalist subclass of unyt_quantity.
    """

    def __new__(
        cls,
        input_scalar: numeric_type,
        units: str | unyt.unit_object.Unit | None = None,
        *,
        registry: unyt.unit_registry.UnitRegistry | None = None,
        dtype: np.dtype | str | None = None,
        bypass_validation: bool = False,
        name: str | None = None,
        extra_attr: bool | None = None,
    ) -> "subclass_uquantity":
        """
        Closely inspired by the ``unyt_quantity.__new__`` constructor.
        """
        if bypass_validation is True:
            result = super().__new__(
                cls,
                np.asarray(input_scalar),
                units=units,
                registry=registry,
                dtype=dtype,
                bypass_validation=bypass_validation,
                name=name,
                extra_attr=extra_attr,
            )

        if not isinstance(input_scalar, (numeric_type, np.number, np.ndarray)):
            raise RuntimeError("subclass_uquantity values must be numeric")

        # Use values from kwargs, if None use values from input_scalar
        units = getattr(input_scalar, "units", None) if units is None else units
        name = getattr(input_scalar, "name", None) if name is None else name
        if isinstance(input_scalar, subclass_uquantity):
            if input_scalar.extra_attr != extra_attr:
                raise ValueError
        extra_attr = (
            getattr(input_scalar, "extra_attr", None)
            if extra_attr is None
            else extra_attr
        )
        result = super().__new__(
            cls,
            np.asarray(input_scalar),
            units=units,
            registry=registry,
            dtype=dtype,
            bypass_validation=bypass_validation,
            name=name,
            extra_attr=extra_attr,
        )
        if result.size > 1:
            raise RuntimeError("subclass_uquantity instances must be scalars")
        return result

    __round__ = _propagate_extra_attr_to_result(
        _ensure_result_is_subclass_uarray_or_uquantity(unyt_quantity.__round__)
    )
