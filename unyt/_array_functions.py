import warnings
from importlib.metadata import version
from numbers import Number

import numpy as np
from packaging.version import Version

from unyt import delta_degC
from unyt.array import NULL_UNIT, unyt_array, unyt_quantity
from unyt.dimensions import temperature
from unyt.exceptions import (
    InvalidUnitOperation,
    UnitInconsistencyError,
    UnytError,
)

NUMPY_VERSION = Version(version("numpy"))

# Functions for which passing units doesn't make sense
# bail out with NotImplemented (escalated to TypeError by numpy)
_UNSUPPORTED_FUNCTIONS = {
    # Polynomials
    np.poly,
    np.polyadd,
    np.polyder,
    np.polydiv,
    np.polyfit,
    np.polyint,
    np.polymul,
    np.polysub,
    np.polyval,
    np.roots,
    np.vander,
    # datetime64 is not a sensible dtype for unyt_array
    np.datetime_as_string,
    np.busday_count,
    np.busday_offset,
    np.is_busday,
    # not clear how to approach
    np.piecewise,  # astropy.units doens't have a simple implementation either
    np.packbits,
    np.unpackbits,
    np.ix_,
}

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
        return func._implementation(np.asarray(a), np.asarray(b)) * prod_units
    res = func._implementation(np.asarray(a), np.asarray(b), out=np.asarray(out))
    if getattr(out, "units", None) is not None:
        out.units = prod_units
    return unyt_array(res, prod_units, bypass_validation=True)


@implements(np.dot)
def dot(a, b, out=None):
    return product_helper(a, b, out, np.dot)


@implements(np.vdot)
def vdot(a, b):
    return np.vdot._implementation(np.asarray(a), np.asarray(b)) * (
        getattr(a, "units", NULL_UNIT) * getattr(b, "units", NULL_UNIT)
    )


@implements(np.inner)
def inner(a, b):
    return np.inner._implementation(np.asarray(a), np.asarray(b)) * (
        getattr(a, "units", NULL_UNIT) * getattr(b, "units", NULL_UNIT)
    )


@implements(np.outer)
def outer(a, b, out=None):
    return product_helper(a, b, out, np.outer)


@implements(np.kron)
def kron(a, b):
    return np.kron._implementation(np.asarray(a), np.asarray(b)) * (
        getattr(a, "units", NULL_UNIT) * getattr(b, "units", NULL_UNIT)
    )


@implements(np.linalg.inv)
def linalg_inv(a, *args, **kwargs):
    return np.linalg.inv._implementation(np.asarray(a), *args, **kwargs) / a.units


@implements(np.linalg.tensorinv)
def linalg_tensorinv(a, *args, **kwargs):
    return np.linalg.tensorinv._implementation(a, *args, **kwargs) / a.units


@implements(np.linalg.pinv)
def linalg_pinv(a, *args, **kwargs):
    return np.linalg.pinv._implementation(a, *args, **kwargs).view(np.ndarray) / a.units


@implements(np.linalg.svd)
def linalg_svd(a, full_matrices=True, compute_uv=True, *args, **kwargs):
    ret_units = a.units
    retv = np.linalg.svd._implementation(
        np.asarray(a), full_matrices, compute_uv, *args, **kwargs
    )
    if compute_uv:
        u, s, vh = retv
        return (u, s * ret_units, vh)
    else:
        return retv * ret_units


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
            if len(units) == 1:
                # allow range to be pure numerical scalars
                # for backward compatibility with unyt 2.9.5
                # see https://github.com/yt-project/unyt/issues/465
                imin *= units[0]
                imax *= units[0]
            else:
                raise TypeError(
                    f"Elements of range must both have a 'units' attribute. Got {_range}"
                )
        new_range[i] = imin.to_value(units[i]), imax.to_value(units[i])
    return new_range.squeeze()


@implements(np.histogram)
def histogram(
    a,
    bins=10,
    range=None,
    *args,
    **kwargs,
):
    range = _sanitize_range(range, units=[a.units])
    counts, bins = np.histogram._implementation(
        np.asarray(a), bins, range, *args, **kwargs
    )
    return counts, bins * a.units


@implements(np.histogram2d)
def histogram2d(x, y, bins=10, range=None, *args, **kwargs):
    range = _sanitize_range(range, units=[x.units, y.units])
    counts, xbins, ybins = np.histogram2d._implementation(
        np.asarray(x), np.asarray(y), bins, range, *args, **kwargs
    )
    return counts, xbins * x.units, ybins * y.units


@implements(np.histogramdd)
def histogramdd(sample, bins=10, range=None, *args, **kwargs):
    units = [_.units for _ in sample]
    range = _sanitize_range(range, units=units)
    counts, bins = np.histogramdd._implementation(
        [np.asarray(_) for _ in sample], bins, range, *args, **kwargs
    )
    return counts, tuple(_bin * u for _bin, u in zip(bins, units))


@implements(np.histogram_bin_edges)
def histogram_bin_edges(a, *args, **kwargs):
    return (
        np.histogram_bin_edges._implementation(np.asarray(a), *args, **kwargs) * a.units
    )


def get_units(objs):
    units = []
    for sub in objs:
        if isinstance(sub, np.ndarray):
            units.append(getattr(sub, "units", NULL_UNIT))
        elif isinstance(sub, Number):
            units.append(NULL_UNIT)
        else:
            units.extend(get_units(sub))
    return units


def _validate_units_consistency(objs):
    """
    Return unique units or raise UnitInconsistencyError if units are mixed.
    """
    # NOTE: we cannot validate that all arrays are unyt_arrays
    # by using this as a guard clause in unyt_array.__array_function__
    # because it's already a necessary condition for numpy to use our
    # custom implementations
    units = get_units(objs)

    # filtering unique units using equality comparison implicitly allows for units
    # which have a dimensionless**(n) component to be treated as one and the same
    unique_units = [units[0], *(u for u in units if u != units[0])]

    if len(unique_units) == 1:
        return units[0]
    else:
        raise UnitInconsistencyError(*units)


def _validate_units_consistency_v2(ref_units, *args) -> None:
    """
    raise UnitInconsistencyError if units are mixed
    if all args are pure numbers, they are treated as having ref_units,
    otherwise they are treated as dimensionless
    """
    if all(isinstance(_, Number) for _ in args):
        return
    else:
        _validate_units_consistency((1 * ref_units, *args))


@implements(np.concatenate)
def concatenate(arrs, /, axis=0, out=None, *args, **kwargs):
    ret_units = _validate_units_consistency(arrs)

    if out is not None:
        out_view = np.asarray(out)
    else:
        out_view = out

    res = np.concatenate._implementation(
        [np.asarray(_) for _ in arrs], axis, out_view, *args, **kwargs
    )

    if getattr(out, "units", None) is not None:
        out.units = ret_units

    return unyt_array(res, ret_units, bypass_validation=True)


@implements(np.cross)
def cross(a, b, *args, **kwargs):
    prod_units = getattr(a, "units", NULL_UNIT) * getattr(b, "units", NULL_UNIT)
    return (
        np.cross._implementation(np.asarray(a), np.asarray(b), *args, **kwargs)
        * prod_units
    )


@implements(np.intersect1d)
def intersect1d(arr1, arr2, /, assume_unique=False, return_indices=False):
    _validate_units_consistency((arr1, arr2))
    retv = np.intersect1d._implementation(
        np.asarray(arr1),
        np.asarray(arr2),
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
    return np.union1d._implementation(np.asarray(arr1), np.asarray(arr2)) * arr1.units


@implements(np.linalg.norm)
def norm(x, /, *args, **kwargs):
    return np.linalg.norm._implementation(np.asarray(x), *args, **kwargs) * x.units


@implements(np.vstack)
def vstack(tup, /):
    ret_units = _validate_units_consistency(tup)
    return np.vstack._implementation([np.asarray(_) for _ in tup]) * ret_units


@implements(np.hstack)
def hstack(tup, /):
    ret_units = _validate_units_consistency(tup)
    return np.vstack._implementation([np.asarray(_) for _ in tup]) * ret_units


@implements(np.dstack)
def dstack(tup, /):
    ret_units = _validate_units_consistency(tup)
    return np.dstack._implementation([np.asarray(_) for _ in tup]) * ret_units


@implements(np.column_stack)
def column_stack(tup, /):
    ret_units = _validate_units_consistency(tup)
    return np.column_stack._implementation([np.asarray(_) for _ in tup]) * ret_units


@implements(np.stack)
def stack(arrays, /, axis=0, out=None):
    ret_units = _validate_units_consistency(arrays)
    if out is None:
        return (
            np.stack._implementation([np.asarray(_) for _ in arrays], axis=axis)
            * ret_units
        )
    res = np.stack._implementation(
        [np.asarray(_) for _ in arrays], axis=axis, out=np.asarray(out)
    )
    if getattr(out, "units", None) is not None:
        out.units = ret_units
    return unyt_array(res, ret_units, bypass_validation=True)


@implements(np.around)
def around(a, decimals=0, out=None):
    ret_units = a.units
    if out is None:
        return np.around._implementation(np.asarray(a), decimals=decimals) * ret_units
    res = np.around._implementation(
        np.asarray(a), decimals=decimals, out=np.asarray(out)
    )
    if getattr(out, "units", None) is not None:
        out.units = ret_units
    return unyt_array(res, ret_units, bypass_validation=True)


@implements(np.block)
def block(arrays):
    ret_units = _validate_units_consistency(arrays)
    return np.block._implementation(arrays) * ret_units


@implements(np.fft.fft)
def ftt_fft(a, *args, **kwargs):
    return np.fft.fft._implementation(np.asarray(a), *args, **kwargs) / a.units


@implements(np.fft.fft2)
def ftt_fft2(a, *args, **kwargs):
    return np.fft.fft2._implementation(np.asarray(a), *args, **kwargs) / a.units


@implements(np.fft.fftn)
def ftt_fftn(a, *args, **kwargs):
    return np.fft.fftn._implementation(np.asarray(a), *args, **kwargs) / a.units


@implements(np.fft.hfft)
def ftt_hfft(a, *args, **kwargs):
    return np.fft.hfft._implementation(np.asarray(a), *args, **kwargs) / a.units


@implements(np.fft.rfft)
def ftt_rfft(a, *args, **kwargs):
    return np.fft.rfft._implementation(np.asarray(a), *args, **kwargs) / a.units


@implements(np.fft.rfft2)
def ftt_rfft2(a, *args, **kwargs):
    return np.fft.rfft2._implementation(np.asarray(a), *args, **kwargs) / a.units


@implements(np.fft.rfftn)
def ftt_rfftn(a, *args, **kwargs):
    return np.fft.rfftn._implementation(np.asarray(a), *args, **kwargs) / a.units


@implements(np.fft.ifft)
def ftt_ifft(a, *args, **kwargs):
    return np.fft.ifft._implementation(np.asarray(a), *args, **kwargs) / a.units


@implements(np.fft.ifft2)
def ftt_ifft2(a, *args, **kwargs):
    return np.fft.ifft2._implementation(np.asarray(a), *args, **kwargs) / a.units


@implements(np.fft.ifftn)
def ftt_ifftn(a, *args, **kwargs):
    return np.fft.ifftn._implementation(np.asarray(a), *args, **kwargs) / a.units


@implements(np.fft.ihfft)
def ftt_ihfft(a, *args, **kwargs):
    return np.fft.ihfft._implementation(np.asarray(a), *args, **kwargs) / a.units


@implements(np.fft.irfft)
def ftt_irfft(a, *args, **kwargs):
    return np.fft.irfft._implementation(np.asarray(a), *args, **kwargs) / a.units


@implements(np.fft.irfft2)
def ftt_irfft2(a, *args, **kwargs):
    return np.fft.irfft2._implementation(np.asarray(a), *args, **kwargs) / a.units


@implements(np.fft.irfftn)
def ftt_irfftn(a, *args, **kwargs):
    return np.fft.irfftn._implementation(np.asarray(a), *args, **kwargs) / a.units


@implements(np.fft.fftshift)
def fft_fftshift(x, *args, **kwargs):
    return np.fft.fftshift._implementation(np.asarray(x), *args, **kwargs) * x.units


@implements(np.fft.ifftshift)
def fft_ifftshift(x, *args, **kwargs):
    return np.fft.ifftshift._implementation(np.asarray(x), *args, **kwargs) * x.units


@implements(np.sort_complex)
def sort_complex(a):
    return np.sort_complex._implementation(np.asarray(a)) * a.units


def _array_comp_helper(a, b):
    au = getattr(a, "units", NULL_UNIT)
    bu = getattr(b, "units", NULL_UNIT)
    if bu != au and au != NULL_UNIT and bu != NULL_UNIT:
        b = b.in_units(au)
    elif bu == NULL_UNIT:
        b = np.array(b) * au
    elif au == NULL_UNIT:
        a = np.array(a) * bu

    return a, b


@implements(np.isclose)
def isclose(a, b, *args, **kwargs):
    a, b = _array_comp_helper(a, b)
    return np.isclose._implementation(np.asarray(a), np.asarray(b), *args, **kwargs)


@implements(np.allclose)
def allclose(a, b, *args, **kwargs):
    a, b = _array_comp_helper(a, b)
    return np.allclose._implementation(np.asarray(a), np.asarray(b), *args, **kwargs)


@implements(np.array_equal)
def array_equal(a1, a2, *args, **kwargs) -> bool:
    u1 = getattr(a1, "units", NULL_UNIT)
    u2 = getattr(a2, "units", NULL_UNIT)
    if u2 != u1:
        return False

    return np.array_equal._implementation(
        np.asarray(a1), np.asarray(a2), *args, **kwargs
    )


@implements(np.array_equiv)
def array_equiv(a1, a2, *args, **kwargs) -> bool:
    u1 = getattr(a1, "units", NULL_UNIT)
    u2 = getattr(a2, "units", NULL_UNIT)
    if u2 != u1:
        return False

    return np.array_equiv._implementation(
        np.asarray(a1), np.asarray(a2), *args, **kwargs
    )


@implements(np.linspace)
def linspace(start, stop, *args, **kwargs):
    _validate_units_consistency((start, stop))
    return (
        np.linspace._implementation(
            np.asarray(start), np.asarray(stop), *args, **kwargs
        )
        * start.units
    )


@implements(np.logspace)
def logspace(start, stop, *args, **kwargs):
    _validate_units_consistency((start, stop))
    return (
        np.logspace._implementation(
            np.asarray(start), np.asarray(stop), *args, **kwargs
        )
        * start.units
    )


@implements(np.geomspace)
def geomspace(start, stop, *args, **kwargs):
    _validate_units_consistency((start, stop))
    return (
        np.geomspace._implementation(
            np.asarray(start), np.asarray(stop), *args, **kwargs
        )
        * start.units
    )


@implements(np.copyto)
def copyto(dst, src, *args, **kwargs):
    # note that np.copyto is heavily used internally
    # in numpy, and it may be used with fundamental datatypes,
    # so we don't attempt to pass ndarray views to keep generality
    np.copyto._implementation(dst, src, *args, **kwargs)
    if getattr(dst, "units", None) is not None:
        dst.units = getattr(src, "units", dst.units)


@implements(np.prod)
def prod(a, *args, **kwargs):
    return np.prod._implementation(np.asarray(a), *args, **kwargs) * a.units**a.size


@implements(np.var)
def var(a, *args, **kwargs):
    return np.var._implementation(np.asarray(a), *args, **kwargs) * a.units**2


@implements(np.trace)
def trace(a, *args, **kwargs):
    return np.trace._implementation(np.asarray(a), *args, **kwargs) * a.units


@implements(np.percentile)
def percentile(a, *args, **kwargs):
    return np.percentile._implementation(np.asarray(a), *args, **kwargs) * a.units


@implements(np.quantile)
def quantile(a, *args, **kwargs):
    return np.quantile._implementation(np.asarray(a), *args, **kwargs) * a.units


@implements(np.nanpercentile)
def nanpercentile(a, *args, **kwargs):
    return np.nanpercentile._implementation(np.asarray(a), *args, **kwargs) * a.units


@implements(np.nanquantile)
def nanquantile(a, *args, **kwargs):
    return np.nanquantile._implementation(np.asarray(a), *args, **kwargs) * a.units


@implements(np.linalg.det)
def linalg_det(a, *args, **kwargs):
    return np.linalg.det._implementation(np.asarray(a), *args, **kwargs) * a.units ** (
        a.shape[0]
    )


@implements(np.linalg.lstsq)
def linalg_lstsq(a, b, *args, **kwargs):
    x, residuals, rank, s = np.linalg.lstsq._implementation(
        np.asarray(a), np.asarray(b), *args, **kwargs
    )
    au = getattr(a, "units", NULL_UNIT)
    bu = getattr(b, "units", NULL_UNIT)
    return (x * bu / au, residuals * bu / au, rank, s * au)


@implements(np.linalg.solve)
def linalg_solve(a, b, *args, **kwargs):
    au = getattr(a, "units", NULL_UNIT)
    bu = getattr(b, "units", NULL_UNIT)
    return (
        np.linalg.solve._implementation(np.asarray(a), np.asarray(b), *args, **kwargs)
        * bu
        / au
    )


@implements(np.linalg.tensorsolve)
def linalg_tensorsolve(a, b, *args, **kwargs):
    au = getattr(a, "units", NULL_UNIT)
    bu = getattr(b, "units", NULL_UNIT)
    return (
        np.linalg.tensorsolve._implementation(
            np.asarray(a), np.asarray(b), *args, **kwargs
        )
        * bu
        / au
    )


@implements(np.linalg.eig)
def linalg_eig(a, *args, **kwargs):
    ret_units = a.units
    w, v = np.linalg.eig._implementation(np.asarray(a), *args, **kwargs)
    return w * ret_units, v


@implements(np.linalg.eigh)
def linalg_eigh(a, *args, **kwargs):
    ret_units = a.units
    w, v = np.linalg.eigh._implementation(np.asarray(a), *args, **kwargs)
    return w * ret_units, v


@implements(np.linalg.eigvals)
def linalg_eigvals(a, *args, **kwargs):
    return np.linalg.eigvals._implementation(np.asarray(a), *args, **kwargs) * a.units


@implements(np.linalg.eigvalsh)
def linalg_eigvalsh(a, *args, **kwargs):
    return np.linalg.eigvalsh._implementation(np.asarray(a), *args, **kwargs) * a.units


@implements(np.savetxt)
def savetxt(fname, X, *args, **kwargs):
    warnings.warn(
        "numpy.savetxt does not preserve units, "
        "and will only save the raw numerical data from the unyt_array object.\n"
        "If this is the intended behaviour, call `numpy.savetxt(file, arr.d)` "
        "to silence this warning.\n"
        "If you want to preserve units, use `unyt.savetxt` "
        "(and `unyt.loadtxt`) instead.",
        stacklevel=4,
    )
    return np.savetxt._implementation(fname, np.asarray(X), *args, **kwargs)


@implements(np.apply_over_axes)
def apply_over_axes(func, a, axes):
    res = func(np.asarray(a), axes[0]) * a.units
    if len(axes) > 1:
        # this function is recursive by nature,
        # here we intentionally do not call the base _implementation
        return np.apply_over_axes(func, res, axes[1:])
    else:
        return res


def diff_helper(func, arr, *args, **kwargs):
    u = getattr(arr, "units", NULL_UNIT)
    if u.dimensions is temperature:
        if u.base_offset:
            raise InvalidUnitOperation(
                "Quantities with units of Fahrenheit or Celsius "
                "cannot be multiplied, divided, subtracted or added."
            )
        ret_units = delta_degC
    else:
        ret_units = u
    return func._implementation(np.asarray(arr), *args, **kwargs) * ret_units


@implements(np.diff)
def diff(a, *args, **kwargs):
    return diff_helper(np.diff, a, *args, **kwargs)


@implements(np.ediff1d)
def ediff1d(a, *args, **kwargs):
    return diff_helper(np.ediff1d, a, *args, **kwargs)


@implements(np.ptp)
def ptp(a, *args, **kwargs):
    return diff_helper(np.ptp, a, *args, **kwargs)


@implements(np.cumprod)
def cumprod(a, *args, **kwargs):
    raise UnytError(
        "numpy.cumprod (and other cumulative product function) cannot be used "
        "with a unyt_array as all return elements should (but cannot) "
        "have different units."
    )


@implements(np.pad)
def pad(array, *args, **kwargs):
    return np.pad._implementation(np.asarray(array), *args, **kwargs) * array.units


@implements(np.choose)
def choose(a, choices, out=None, *args, **kwargs):
    if (au := getattr(a, "units", NULL_UNIT)) != NULL_UNIT:
        raise TypeError(
            f"The first argument to numpy.choose must be dimensionless, got units={au}"
        )
    retu = _validate_units_consistency(choices)

    if out is None:
        return (
            np.choose._implementation(
                a, [np.asarray(c) for c in choices], *args, **kwargs
            )
            * retu
        )

    res = np.choose._implementation(
        a,
        [np.asarray(c) for c in choices],
        *args,
        out=np.asarray(out),
        **kwargs,
    )
    if getattr(out, "units", None) is not None:
        out.units = retu
    return unyt_array(res, retu, bypass_validation=True)


@implements(np.fill_diagonal)
def fill_diagonal(a, val, *args, **kwargs) -> None:
    _validate_units_consistency_v2(a.units, val)
    np.fill_diagonal._implementation(np.asarray(a), val, *args, **kwargs)


@implements(np.insert)
def insert(arr, obj, values, *args, **kwargs):
    _validate_units_consistency_v2(arr.units, values)
    return (
        np.insert._implementation(
            np.asarray(arr), obj, np.asarray(values), *args, **kwargs
        )
        * arr.units
    )


@implements(np.isin)
def isin(element, test_elements, *args, **kwargs):
    _validate_units_consistency((element, test_elements))
    return np.isin._implementation(
        np.asarray(element), np.asarray(test_elements), *args, **kwargs
    )


@implements(np.place)
def place(arr, mask, vals, *args, **kwargs) -> None:
    _validate_units_consistency_v2(arr.units, vals)
    np.place._implementation(np.asarray(arr), mask, np.asarray(vals), *args, **kwargs)


@implements(np.put)
def put(a, ind, v, *args, **kwargs) -> None:
    _validate_units_consistency_v2(a.units, v)
    np.put._implementation(np.asarray(a), ind, np.asarray(v))


@implements(np.put_along_axis)
def put_along_axis(arr, indices, values, axis, *args, **kwargs) -> None:
    _validate_units_consistency_v2(arr.units, values)
    np.put_along_axis._implementation(
        np.asarray(arr), indices, np.asarray(values), axis, *args, **kwargs
    )


@implements(np.putmask)
def putmask(a, mask, values, *args, **kwargs) -> None:
    _validate_units_consistency_v2(a.units, values)
    np.putmask._implementation(np.asarray(a), mask, np.asarray(values), *args, **kwargs)


@implements(np.searchsorted)
def searchsorted(a, v, *args, **kwargs):
    _validate_units_consistency_v2(a.units, v)
    return np.searchsorted._implementation(
        np.asarray(a), np.asarray(v), *args, **kwargs
    )


@implements(np.select)
def select(condlist, choicelist, default=0, *args, **kwargs):
    ref_units = choicelist[0].units
    _validate_units_consistency_v2(ref_units, choicelist, default)
    return (
        np.select._implementation(
            condlist, [np.asarray(c) for c in choicelist], default
        )
        * ref_units
    )


@implements(np.setdiff1d)
def setdiff1d(ar1, ar2, *args, **kwargs):
    retu = _validate_units_consistency((ar1, ar2))
    return (
        np.setdiff1d._implementation(np.asarray(ar1), np.asarray(ar2), *args, **kwargs)
        * retu
    )


@implements(np.sinc)
def sinc(x, *args, **kwargs):
    # this implementation becomes necessary after implementing where
    # we *want* this one to ignore units
    return np.sinc._implementation(np.asarray(x), *args, **kwargs)


@implements(np.clip)
def clip(a, a_min, a_max, out=None, *args, **kwargs):
    _validate_units_consistency_v2(a.units, a_min, a_max)
    if out is None:
        return (
            np.clip._implementation(
                np.asarray(a), np.asarray(a_min), np.asarray(a_max), *args, **kwargs
            )
            * a.units
        )

    res = (
        np.clip._implementation(
            np.asarray(a),
            np.asarray(a_min),
            np.asarray(a_max),
            *args,
            out=np.asarray(out),
            **kwargs,
        )
        * a.units
    )
    if getattr(out, "units", None) is not None:
        out.units = a.units
    return unyt_array(res, a.units, bypass_validation=True)


@implements(np.where)
def where(condition, *args, **kwargs):
    if len(args) == 0:
        return np.where._implementation(np.asarray(condition), **kwargs)

    elif len(args) < 2:
        # error message borrowed from numpy 1.24.1
        raise ValueError("either both or neither of x and y should be given")

    x, y, *args = args

    retu = _validate_units_consistency((x, y))
    return (
        np.where._implementation(
            condition, np.asarray(x), np.asarray(y), *args, **kwargs
        )
        * retu
    )


@implements(np.triu)
def triu(m, *args, **kwargs):
    return np.triu._implementation(np.asarray(m), *args, **kwargs) * m.units


@implements(np.tril)
def tril(m, *args, **kwargs):
    return np.tril._implementation(np.asarray(m), *args, **kwargs) * m.units


@implements(np.einsum)
def einsum(subscripts, *operands, out=None, **kwargs):
    ret_units = _validate_units_consistency(operands)

    if out is not None:
        out_view = np.asarray(out)
    else:
        out_view = out

    res = np.einsum._implementation(subscripts, *operands, out=out_view)

    if getattr(out, "units", None) is not None:
        out.units = ret_units

    if res.ndim == 0:
        cls = unyt_quantity
    else:
        cls = unyt_array

    return cls(res, ret_units, bypass_validation=True)


@implements(np.convolve)
def convolve(a, v, *args, **kwargs):
    ret_units = np.prod(get_units((a, v)))
    return (
        np.convolve._implementation(np.asarray(a), np.asarray(v), *args, **kwargs)
        * ret_units
    )


@implements(np.correlate)
def correlate(a, v, *args, **kwargs):
    ret_units = np.prod(get_units((a, v)))
    return (
        np.correlate._implementation(np.asarray(a), np.asarray(v), *args, **kwargs)
        * ret_units
    )


@implements(np.tensordot)
def tensordot(a, b, *args, **kwargs):
    ret_units = np.prod(get_units((a, b)))
    return (
        np.tensordot._implementation(np.asarray(a), np.asarray(b), *args, **kwargs)
        * ret_units
    )


@implements(np.unwrap)
def unwrap(p, *args, **kwargs):
    ret_units = p.units
    return np.unwrap._implementation(np.asarray(p), *args, **kwargs) * ret_units


@implements(np.interp)
def interp(x, xp, fp, *args, **kwargs):
    _validate_units_consistency((x, xp))

    # return array type should match fp's
    # so, the fallback multiplier is 1 instead of NULL_UNITS
    # This avoid leaking a dimensionless unyt_array if reference data
    # is a pure np.ndarray
    ret_units = getattr(fp, "units", 1)
    return (
        np.interp(np.asarray(x), np.asarray(xp), np.asarray(fp), *args, **kwargs)
        * ret_units
    )


@implements(np.array_repr)
def array_repr(arr, *args, **kwargs):
    rep = np.array_repr._implementation(np.asarray(arr), *args, **kwargs)
    rep = rep.replace("array", arr.__class__.__name__)
    units_repr = arr.units.__repr__()
    if "=" in rep:
        return rep[:-1] + ", units='" + units_repr + "')"
    else:
        return rep[:-1] + ", '" + units_repr + "')"


if NUMPY_VERSION < Version("2.0.0dev0"):
    # functions that are removed in numpy 2.0.0
    @implements(np.asfarray)  # noqa: NPY201
    def asfarray(a, dtype=np.double):
        ret_units = a.units
        arr = np.asfarray._implementation(np.asarray(a), dtype=dtype)  # noqa: NPY201
        return arr * ret_units

    _trapezoid_func = np.trapz

elif NUMPY_VERSION >= Version("2.0.0dev0"):
    # functions that were added in numpy 2.0.0
    @implements(np.linalg.outer)
    def linalg_outer(x1, x2, /):
        return product_helper(x1, x2, out=None, func=np.linalg.outer)

    _trapezoid_func = np.trapezoid


@implements(_trapezoid_func)
def trapezoid(y, x=None, dx=1.0, *args, **kwargs):
    ret_units = y.units
    if x is None:
        ret_units = ret_units * getattr(dx, "units", NULL_UNIT)
    else:
        ret_units = ret_units * getattr(x, "units", NULL_UNIT)
    if isinstance(x, np.ndarray):
        x = np.asarray(x)
    if isinstance(dx, np.ndarray):
        dx = np.asarray(dx)
    return (
        _trapezoid_func._implementation(np.asarray(y), x, dx, *args, **kwargs)
        * ret_units
    )


if hasattr(np, "in1d"):

    @implements(np.in1d)
    def in1d(ar1, ar2, *args, **kwargs):
        _validate_units_consistency((ar1, ar2))
        return np.isin._implementation(
            np.asarray(ar1), np.asarray(ar2), *args, **kwargs
        )
