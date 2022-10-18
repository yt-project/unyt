import numpy as np
from packaging.version import Version

from unyt.array import NULL_UNIT, unyt_array
from unyt.exceptions import UnitInconsistencyError

NUMPY_VERSION = Version(np.__version__)
_HANDLED_FUNCTIONS = {}


def implements(numpy_function):
    """Register an __array_function__ implementation for unyt_array objects."""
    # See NEP 18 https://numpy.org/neps/nep-0018-array-function-protocol.html
    def decorator(func):
        _HANDLED_FUNCTIONS[numpy_function] = func
        return func

    return decorator


@implements(np.array2string)
def array2string(a, *args, **kwargs):
    return (
        np.array2string._implementation(a, *args, **kwargs)
        + f", units={str(a.units)!r}"
    )


def product_helper(a, b, out, func):
    prod_units = getattr(a, "units", NULL_UNIT) * getattr(b, "units", NULL_UNIT)
    if out is None:
        return func._implementation(a.view(np.ndarray), b.view(np.ndarray)) * prod_units
    res = func._implementation(
        a.view(np.ndarray), b.view(np.ndarray), out=out.view(np.ndarray)
    )
    if getattr(out, "units", None) is not None:
        out.units = prod_units
    return unyt_array(res, prod_units, bypass_validation=True)


@implements(np.dot)
def dot(a, b, out=None):
    return product_helper(a, b, out, np.dot)


@implements(np.vdot)
def vdot(a, b):
    return np.vdot._implementation(a.view(np.ndarray), b.view(np.ndarray)) * (
        getattr(a, "units", NULL_UNIT) * getattr(b, "units", NULL_UNIT)
    )


@implements(np.inner)
def inner(a, b):
    return np.inner._implementation(a.view(np.ndarray), b.view(np.ndarray)) * (
        getattr(a, "units", NULL_UNIT) * getattr(b, "units", NULL_UNIT)
    )


@implements(np.outer)
def outer(a, b, out=None):
    return product_helper(a, b, out, np.outer)


@implements(np.kron)
def kron(a, b):
    return np.kron._implementation(a.view(np.ndarray), b.view(np.ndarray)) * (
        getattr(a, "units", NULL_UNIT) * getattr(b, "units", NULL_UNIT)
    )


@implements(np.linalg.inv)
def linalg_inv(a, *args, **kwargs):
    return np.linalg.inv._implementation(a.view(np.ndarray), *args, **kwargs) / a.units


@implements(np.linalg.tensorinv)
def linalg_tensorinv(a, *args, **kwargs):
    return np.linalg.tensorinv._implementation(a, *args, **kwargs) / a.units


@implements(np.linalg.pinv)
def linalg_pinv(a, *args, **kwargs):
    return np.linalg.pinv._implementation(a, *args, **kwargs).view(np.ndarray) / a.units


def _sanitize_range(_range, units):
    # helper function to histogram* functions
    ndim = len(units)
    if _range is None:
        return _range
    new_range = np.empty((ndim, 2))
    for i in range(ndim):
        ilim = _range[2 * i : 2 * (i + 1)]
        imin, imax = ilim
        if not (hasattr(imin, "units") and hasattr(imax, "units")):
            raise TypeError(
                f"Elements of range must both have a 'units' attribute. Got {_range}"
            )
        new_range[i] = imin.to(units[i]).value, imax.to(units[i]).value
    return new_range.squeeze()


@implements(np.histogram)
def histogram(
    a,
    bins=10,
    range=None,
    normed=None,
    weights=None,
    density=None,
):
    range = _sanitize_range(range, units=[a.units])
    counts, bins = np.histogram._implementation(
        a.view(np.ndarray),
        bins=bins,
        range=range,
        normed=normed,
        weights=weights,
        density=density,
    )
    return counts, bins * a.units


@implements(np.histogram2d)
def histogram2d(
    x,
    y,
    bins=10,
    range=None,
    normed=None,
    weights=None,
    density=None,
):
    range = _sanitize_range(range, units=[x.units, y.units])
    counts, xbins, ybins = np.histogram2d._implementation(
        x.view(np.ndarray),
        y.view(np.ndarray),
        bins=bins,
        range=range,
        normed=normed,
        weights=weights,
        density=density,
    )
    return counts, xbins * x.units, ybins * y.units


@implements(np.histogramdd)
def histogramdd(
    sample,
    bins=10,
    range=None,
    normed=None,
    weights=None,
    density=None,
):
    units = [_.units for _ in sample]
    range = _sanitize_range(range, units=units)
    counts, bins = np.histogramdd._implementation(
        [_.view(np.ndarray) for _ in sample],
        bins=bins,
        range=range,
        normed=normed,
        weights=weights,
        density=density,
    )
    return counts, tuple(_bin * u for _bin, u in zip(bins, units))


def _validate_units_consistency(arrs):
    # NOTE: we cannot validate that all arrays are unyt_arrays
    # by using this as a guard clause in unyt_array.__array_function__
    # because it's already a necessary condition for numpy to use our
    # custom implementations
    ref_units = getattr(arrs[0], "units", NULL_UNIT)
    if any(getattr(a, "units", NULL_UNIT) != ref_units for a in arrs[1:]):
        raise UnitInconsistencyError(*(a.units for a in arrs))


@implements(np.concatenate)
def concatenate(arrs, /, axis=0, out=None, dtype=None, casting="same_kind"):
    _validate_units_consistency(arrs)
    ret_units = arrs[0].units
    if out is None:
        if NUMPY_VERSION >= Version("1.20"):
            res = np.concatenate._implementation(
                [_.view(np.ndarray) for _ in arrs],
                axis=axis,
                dtype=dtype,
                casting=casting,
            )
        else:
            res = np.concatenate._implementation(
                [_.view(np.ndarray) for _ in arrs],
                axis=axis,
            )
    else:
        if NUMPY_VERSION >= Version("1.20"):
            res = np.concatenate._implementation(
                [_.view(np.ndarray) for _ in arrs],
                axis=axis,
                out=out.view(np.ndarray),
                dtype=dtype,
                casting=casting,
            )
        else:
            res = np.concatenate._implementation(
                [_.view(np.ndarray) for _ in arrs],
                axis=axis,
                out=out.view(np.ndarray),
            )
        if getattr(out, "units", None) is not None:
            out.units = ret_units
    return unyt_array(res, ret_units, bypass_validation=True)


@implements(np.cross)
def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    prod_units = getattr(a, "units", NULL_UNIT) * getattr(b, "units", NULL_UNIT)
    return (
        np.cross._implementation(
            a.view(np.ndarray),
            b.view(np.ndarray),
            axisa=axisa,
            axisb=axisb,
            axisc=axisc,
            axis=axis,
        )
        * prod_units
    )


@implements(np.intersect1d)
def intersect1d(arr1, arr2, /, assume_unique=False, return_indices=False):
    _validate_units_consistency((arr1, arr2))
    retv = np.intersect1d._implementation(
        arr1.view(np.ndarray),
        arr2.view(np.ndarray),
        assume_unique=assume_unique,
        return_indices=return_indices,
    )
    if return_indices:
        return retv
    else:
        return retv * arr1.units


@implements(np.union1d)
def union1d(arr1, arr2, /):
    _validate_units_consistency((arr1, arr2))
    return (
        np.union1d._implementation(arr1.view(np.ndarray), arr2.view(np.ndarray))
        * arr1.units
    )


@implements(np.linalg.norm)
def norm(x, /, ord=None, axis=None, keepdims=False):
    return (
        np.linalg.norm._implementation(
            x.view(np.ndarray), ord=ord, axis=axis, keepdims=keepdims
        )
        * x.units
    )


@implements(np.vstack)
def vstack(tup, /):
    _validate_units_consistency(tup)
    return np.vstack._implementation([_.view(np.ndarray) for _ in tup]) * tup[0].units


@implements(np.hstack)
def hstack(tup, /):
    _validate_units_consistency(tup)
    return np.vstack._implementation([_.view(np.ndarray) for _ in tup]) * tup[0].units


@implements(np.stack)
def stack(arrays, /, axis=0, out=None):
    _validate_units_consistency(arrays)
    ret_units = arrays[0].units
    if out is None:
        return (
            np.stack._implementation([_.view(np.ndarray) for _ in arrays], axis=axis)
            * ret_units
        )
    res = np.stack._implementation(
        [_.view(np.ndarray) for _ in arrays], axis=axis, out=out.view(np.ndarray)
    )
    if getattr(out, "units", None) is not None:
        out.units = ret_units
    return unyt_array(res, ret_units, bypass_validation=True)


@implements(np.around)
def around(a, decimals=0, out=None):
    ret_units = a.units
    if out is None:
        return (
            np.around._implementation(a.view(np.ndarray), decimals=decimals) * ret_units
        )
    res = np.around._implementation(
        a.view(np.ndarray), decimals=decimals, out=out.view(np.ndarray)
    )
    if getattr(out, "units", None) is not None:
        out.units = ret_units
    return unyt_array(res, ret_units, bypass_validation=True)
