import os
import warnings
from copy import copy, deepcopy

import numpy as np
import pytest

import unyt as u

from .sample_subclasses import ExtraAttributeError, subclass_uarray, subclass_uquantity

savetxt_file = "saved_array.txt"


def getfunc(fname):
    """
    Helper for our tests: get the function handle from a name (possibly with attribute
    access).
    """
    func = np
    for attr in fname.split("."):
        func = getattr(func, attr)
    return func


def sub_arr(x, unit=u.Mpc):
    """
    Helper for our tests: turn an array into a subclass_uarray.
    """
    return subclass_uarray(x, unit, extra_attr=True)


def sub_q(x, unit=u.Mpc):
    """
    Helper for our tests: turn a scalar into a subclass_uquantity.
    """
    return subclass_uquantity(x, unit, extra_attr=True)


def arg_to_ua(arg):
    """
    Helper for our tests: recursively convert subclass_* in an argument (possibly an
    iterable) to their unyt_* equivalents.
    """
    if type(arg) in (list, tuple):
        return type(arg)([arg_to_ua(a) for a in arg])
    else:
        return to_ua(arg)


def to_ua(x):
    """
    Helper for our tests: turn a subclass_* object into its unyt_* equivalent.
    """
    return u.unyt_array(x) if hasattr(x, "extra_attr") else x


def check_result(x_s, x_u, ignore_values=False):
    """
    Helper for our tests: check that a result with subclass input matches what we
    expected based on the result with unyt input.

    We check:
     - that the type of the result makes sense, recursing if needed.
     - that the value of the result matches (unless ignore_values=False).
     - that the units match.
    """
    if x_u is None:
        assert x_s is None
        return
    if isinstance(x_u, str):
        assert isinstance(x_s, str)
        return
    if isinstance(x_u, type) or isinstance(x_u, np.dtype):
        assert x_u == x_s
        return
    if type(x_u) in (list, tuple):
        assert type(x_u) is type(x_s)
        assert len(x_u) == len(x_s)
        for x_c_i, x_u_i in zip(x_s, x_u):
            check_result(x_c_i, x_u_i)
            return
    # careful, unyt_quantity is a subclass of unyt_array:
    if isinstance(x_u, u.unyt_quantity):
        assert isinstance(x_s, subclass_uquantity)
    elif isinstance(x_u, u.unyt_array):
        assert isinstance(x_s, subclass_uarray) and not isinstance(
            x_s, subclass_uquantity
        )
    else:
        assert not isinstance(x_s, subclass_uarray)
        if not ignore_values:
            assert np.allclose(x_s, x_u)
        return
    assert x_s.units == x_u.units
    if not ignore_values:
        assert np.allclose(x_s.to_value(x_s.units), x_u.to_value(x_u.units))
    if isinstance(x_s, subclass_uarray):  # includes subclass_uquantity
        assert x_s.extra_attr is True
    return


class TestSubclassArrayInit:
    """
    Test different ways of initializing a subclass_uarray.
    """

    def test_init_from_ndarray(self):
        """
        Check initializing from a bare numpy array.
        """
        arr = subclass_uarray(np.ones(5), units=u.Mpc, extra_attr=True)
        assert hasattr(arr, "extra_attr")
        assert isinstance(arr, subclass_uarray)

    def test_init_from_list(self):
        """
        Check initializing from a list of values.
        """
        arr = subclass_uarray(
            [1, 1, 1, 1, 1],
            units=u.Mpc,
            extra_attr=True,
        )
        assert hasattr(arr, "extra_attr")
        assert isinstance(arr, subclass_uarray)

    def test_init_from_unyt_array(self):
        """
        Check initializing from a unyt_array.
        """
        arr = subclass_uarray(
            u.unyt_array(np.ones(5), units=u.Mpc),
            extra_attr=False,
        )
        assert hasattr(arr, "extra_attr")
        assert isinstance(arr, subclass_uarray)

    def test_init_from_list_of_unyt_arrays(self):
        """
        Check initializing from a list of unyt_array's.

        Note that unyt won't recurse deeper than one level on inputs, so we don't test
        deeper than one level of lists.
        """
        arr = subclass_uarray(
            [u.unyt_array(1, units=u.Mpc) for _ in range(5)],
            extra_attr=True,
        )
        assert hasattr(arr, "extra_attr")
        assert isinstance(arr, subclass_uarray)

    def test_init_from_list_of_subclass_arrays(self):
        """
        Check initializing from a list of subclass_uarray's.

        Note that unyt won't recurse deeper than one level on inputs, so we don't test
        deeper than one level of lists.
        """
        arr = subclass_uarray(
            [
                subclass_uarray(
                    [1],
                    units=u.Mpc,
                    extra_attr=True,
                )
                for _ in range(5)
            ]
        )
        assert isinstance(arr, subclass_uarray)
        assert hasattr(arr, "extra_attr") and arr.extra_attr is True

    def test_expected_init_failures(self):
        # we refuse mixed extra_attr:
        with pytest.raises(ExtraAttributeError):
            subclass_uarray(
                [
                    subclass_uquantity(1, u.Mpc, extra_attr=True),
                    subclass_uquantity(1, u.Mpc, extra_attr=False),
                ]
            )
        for cls, inp in ((subclass_uarray, [1]), (subclass_uquantity, 1)):
            # we refuse overriding an input subclass_uarray's information:
            with pytest.raises(ValueError):
                cls(
                    cls(inp, units=u.Mpc, extra_attr=False),
                    units=u.Mpc,
                    extra_attr=True,
                )
            # unless it matches, that's fine:
            cls(
                cls(
                    inp,
                    units=u.Mpc,
                    extra_attr=True,
                ),
                units=u.Mpc,
                extra_attr=True,
            )


class TestNumpyFunctions:
    """
    Check that numpy functions recognize our subclasses as input and handle them
    correctly.
    """

    def test_explicitly_handled_funcs(self):
        """
        Make sure we at least handle everything that unyt does, and anything that
        'just worked' for unyt but that we need to handle by hand.

        We don't try to be exhaustive here, but at give some basic input to every function
        that we expect to be able to take subclass input. We then use our helpers defined
        above to convert the inputs to unyt equivalents and call the numpy function on
        both subclass and unyt input. Then we use our helpers to check the results for
        consistency. For instance, if with unyt input we got back a unyt_array, we
        should expect a subclass_uarray.

        We are not currently explicitly testing that the results of any specific function
        are numerically what we expected them to be (seems like overkill), nor that the
        extra_attr's make sense given the input. The latter would be a useful addition,
        but I can't think of a sensible way to implement this besides writing in the
        expectation for every output value of every function by hand.

        As long as no functions outright crash, the test will report the list of functions
        that we should have covered that we didn't cover in tests, and/or the list of
        functions whose output values were not what we expected based on running them with
        unyt input. Otherwise we just get a stack trace of the first function that
        crashed.
        """
        from unyt._array_functions import _HANDLED_FUNCTIONS
        from unyt.tests.test_array_functions import NOOP_FUNCTIONS

        functions_to_check = {
            # FUNCTIONS UNYT HANDLES EXPLICITLY:
            "array2string": (sub_arr(np.arange(3)),),
            "dot": (sub_arr(np.arange(3)), sub_arr(np.arange(3))),
            "vdot": (sub_arr(np.arange(3)), sub_arr(np.arange(3))),
            "inner": (sub_arr(np.arange(3)), sub_arr(np.arange(3))),
            "outer": (sub_arr(np.arange(3)), sub_arr(np.arange(3))),
            "kron": (sub_arr(np.arange(3)), sub_arr(np.arange(3))),
            "histogram_bin_edges": (sub_arr(np.arange(3)),),
            "linalg.inv": (sub_arr(np.eye(3)),),
            "linalg.tensorinv": (sub_arr(np.eye(9).reshape((3, 3, 3, 3))),),
            "linalg.pinv": (sub_arr(np.eye(3)),),
            "linalg.svd": (sub_arr(np.eye(3)),),
            "histogram": (sub_arr(np.arange(3)),),
            "histogram2d": (
                sub_arr(np.arange(3)),
                sub_arr(np.arange(3)),
            ),
            "histogramdd": (sub_arr(np.arange(3)).reshape((1, 3)),),
            "concatenate": (sub_arr(np.eye(3)),),
            "cross": (sub_arr(np.arange(3)), sub_arr(np.arange(3))),
            "intersect1d": (
                sub_arr(np.arange(3)),
                sub_arr(np.arange(3)),
            ),
            "union1d": (sub_arr(np.arange(3)), sub_arr(np.arange(3))),
            "linalg.norm": (sub_arr(np.arange(3)),),
            "vstack": (sub_arr(np.arange(3)),),
            "hstack": (sub_arr(np.arange(3)),),
            "dstack": (sub_arr(np.arange(3)),),
            "column_stack": (sub_arr(np.arange(3)),),
            "stack": (sub_arr(np.arange(3)),),
            "around": (sub_arr(np.arange(3)),),
            "block": ([[sub_arr(np.arange(3))], [sub_arr(np.arange(3))]],),
            "fft.fft": (sub_arr(np.arange(3)),),
            "fft.fft2": (sub_arr(np.eye(3)),),
            "fft.fftn": (sub_arr(np.arange(3)),),
            "fft.hfft": (sub_arr(np.arange(3)),),
            "fft.rfft": (sub_arr(np.arange(3)),),
            "fft.rfft2": (sub_arr(np.eye(3)),),
            "fft.rfftn": (sub_arr(np.arange(3)),),
            "fft.ifft": (sub_arr(np.arange(3)),),
            "fft.ifft2": (sub_arr(np.eye(3)),),
            "fft.ifftn": (sub_arr(np.arange(3)),),
            "fft.ihfft": (sub_arr(np.arange(3)),),
            "fft.irfft": (sub_arr(np.arange(3)),),
            "fft.irfft2": (sub_arr(np.eye(3)),),
            "fft.irfftn": (sub_arr(np.arange(3)),),
            "fft.fftshift": (sub_arr(np.arange(3)),),
            "fft.ifftshift": (sub_arr(np.arange(3)),),
            "sort_complex": (sub_arr(np.arange(3)),),
            "isclose": (sub_arr(np.arange(3)), sub_arr(np.arange(3))),
            "allclose": (sub_arr(np.arange(3)), sub_arr(np.arange(3))),
            "array_equal": (
                sub_arr(np.arange(3)),
                sub_arr(np.arange(3)),
            ),
            "array_equiv": (
                sub_arr(np.arange(3)),
                sub_arr(np.arange(3)),
            ),
            "linspace": (sub_q(1), sub_q(2)),
            "logspace": (
                sub_q(1, unit=u.dimensionless),
                sub_q(2, unit=u.dimensionless),
            ),
            "geomspace": (sub_q(1), sub_q(1)),
            "copyto": (sub_arr(np.arange(3)), sub_arr(np.arange(3))),
            "prod": (sub_arr(np.arange(3)),),
            "var": (sub_arr(np.arange(3)),),
            "trace": (sub_arr(np.eye(3)),),
            "percentile": (sub_arr(np.arange(3)), 30),
            "quantile": (sub_arr(np.arange(3)), 0.3),
            "nanpercentile": (sub_arr(np.arange(3)), 30),
            "nanquantile": (sub_arr(np.arange(3)), 0.3),
            "linalg.det": (sub_arr(np.eye(3)),),
            "diff": (sub_arr(np.arange(3)),),
            "ediff1d": (sub_arr(np.arange(3)),),
            "ptp": (sub_arr(np.arange(3)),),
            "cumprod": (sub_arr(np.arange(3)),),
            "pad": (sub_arr(np.arange(3)), 3),
            "choose": (np.arange(3), sub_arr(np.eye(3))),
            "insert": (sub_arr(np.arange(3)), 1, sub_q(1)),
            "linalg.lstsq": (sub_arr(np.eye(3)), sub_arr(np.eye(3))),
            "linalg.solve": (sub_arr(np.eye(3)), sub_arr(np.eye(3))),
            "linalg.tensorsolve": (
                sub_arr(np.eye(24).reshape((6, 4, 2, 3, 4))),
                sub_arr(np.ones((6, 4))),
            ),
            "linalg.eig": (sub_arr(np.eye(3)),),
            "linalg.eigh": (sub_arr(np.eye(3)),),
            "linalg.eigvals": (sub_arr(np.eye(3)),),
            "linalg.eigvalsh": (sub_arr(np.eye(3)),),
            "savetxt": (savetxt_file, sub_arr(np.arange(3))),
            "fill_diagonal": (
                sub_arr(np.eye(3)),
                sub_arr(np.arange(3)),
            ),
            "apply_over_axes": (lambda x, axis: x, sub_arr(np.eye(3)), (0, 1)),
            "isin": (sub_arr(np.arange(3)), sub_arr(np.arange(3))),
            "place": (
                sub_arr(np.arange(3)),
                np.arange(3) > 0,
                sub_arr(np.arange(3)),
            ),
            "put": (
                sub_arr(np.arange(3)),
                np.arange(3),
                sub_arr(np.arange(3)),
            ),
            "put_along_axis": (
                sub_arr(np.arange(3)),
                np.arange(3),
                sub_arr(np.arange(3)),
                0,
            ),
            "putmask": (
                sub_arr(np.arange(3)),
                np.arange(3),
                sub_arr(np.arange(3)),
            ),
            "searchsorted": (
                sub_arr(np.arange(3)),
                sub_arr(np.arange(3)),
            ),
            "select": (
                [np.arange(3) < 1, np.arange(3) > 1],
                [sub_arr(np.arange(3)), sub_arr(np.arange(3))],
                sub_q(1),
            ),
            "setdiff1d": (
                sub_arr(np.arange(3)),
                sub_arr(np.arange(3, 6)),
            ),
            "sinc": (sub_arr(np.arange(3)),),
            "clip": (sub_arr(np.arange(3)), sub_q(1), sub_q(2)),
            "where": (
                sub_arr(np.arange(3)),
                sub_arr(np.arange(3)),
                sub_arr(np.arange(3)),
            ),
            "triu": (sub_arr(np.ones((3, 3))),),
            "tril": (sub_arr(np.ones((3, 3))),),
            "einsum": ("ii->i", sub_arr(np.eye(3))),
            "convolve": (sub_arr(np.arange(3)), sub_arr(np.arange(3))),
            "correlate": (sub_arr(np.arange(3)), sub_arr(np.arange(3))),
            "tensordot": (sub_arr(np.eye(3)), sub_arr(np.eye(3))),
            "unwrap": (sub_arr(np.arange(3)),),
            "interp": (
                sub_arr(np.arange(3)),
                sub_arr(np.arange(3)),
                sub_arr(np.arange(3)),
            ),
            "array_repr": (sub_arr(np.arange(3)),),
            "linalg.outer": (
                sub_arr(np.arange(3)),
                sub_arr(np.arange(3)),
            ),
            "trapezoid": (sub_arr(np.arange(3)),),
            "in1d": (
                sub_arr(np.arange(3)),
                sub_arr(np.arange(3)),
            ),  # np deprecated
            "take": (sub_arr(np.arange(3)), np.arange(3)),
            # FUNCTIONS THAT UNYT DOESN'T HANDLE EXPLICITLY (THEY "JUST WORK"):
            "all": (sub_arr(np.arange(3)),),
            "amax": (sub_arr(np.arange(3)),),  # implemented via max
            "amin": (sub_arr(np.arange(3)),),  # implemented via min
            "angle": (sub_q(complex(1, 1)),),
            "any": (sub_arr(np.arange(3)),),
            "append": (sub_arr(np.arange(3)), sub_q(1)),
            "apply_along_axis": (lambda x: x, 0, sub_arr(np.eye(3))),
            "argmax": (sub_arr(np.arange(3)),),  # implemented via max
            "argmin": (sub_arr(np.arange(3)),),  # implemented via min
            "argpartition": (
                sub_arr(np.arange(3)),
                1,
            ),  # implemented via partition
            "argsort": (sub_arr(np.arange(3)),),  # implemented via sort
            "argwhere": (sub_arr(np.arange(3)),),
            "array_str": (sub_arr(np.arange(3)),),
            "atleast_1d": (sub_arr(np.arange(3)),),
            "atleast_2d": (sub_arr(np.arange(3)),),
            "atleast_3d": (sub_arr(np.arange(3)),),
            "average": (sub_arr(np.arange(3)),),
            "can_cast": (sub_arr(np.arange(3)), np.float64),
            "common_type": (
                sub_arr(np.arange(3)),
                sub_arr(np.arange(3)),
            ),
            "result_type": (sub_arr(np.ones(3)), sub_arr(np.ones(3))),
            "iscomplex": (sub_arr(np.arange(3)),),
            "iscomplexobj": (sub_arr(np.arange(3)),),
            "isreal": (sub_arr(np.arange(3)),),
            "isrealobj": (sub_arr(np.arange(3)),),
            "nan_to_num": (sub_arr(np.arange(3)),),
            "nanargmax": (sub_arr(np.arange(3)),),  # implemented via max
            "nanargmin": (sub_arr(np.arange(3)),),  # implemented via min
            "nanmax": (sub_arr(np.arange(3)),),  # implemented via max
            "nanmean": (sub_arr(np.arange(3)),),  # implemented via mean
            "nanmedian": (sub_arr(np.arange(3)),),  # implemented via median
            "nanmin": (sub_arr(np.arange(3)),),  # implemented via min
            "trim_zeros": (sub_arr(np.arange(3)),),
            "max": (sub_arr(np.arange(3)),),
            "mean": (sub_arr(np.arange(3)),),
            "median": (sub_arr(np.arange(3)),),
            "min": (sub_arr(np.arange(3)),),
            "ndim": (sub_arr(np.arange(3)),),
            "shape": (sub_arr(np.arange(3)),),
            "size": (sub_arr(np.arange(3)),),
            "sort": (sub_arr(np.arange(3)),),
            "sum": (sub_arr(np.arange(3)),),
            "repeat": (sub_arr(np.arange(3)), 2),
            "tile": (sub_arr(np.arange(3)), 2),
            "shares_memory": (
                sub_arr(np.arange(3)),
                sub_arr(np.arange(3)),
            ),
            "nonzero": (sub_arr(np.arange(3)),),
            "count_nonzero": (sub_arr(np.arange(3)),),
            "flatnonzero": (sub_arr(np.arange(3)),),
            "isneginf": (sub_arr(np.arange(3)),),
            "isposinf": (sub_arr(np.arange(3)),),
            "empty_like": (sub_arr(np.arange(3)),),
            "full_like": (sub_arr(np.arange(3)), sub_q(1)),
            "ones_like": (sub_arr(np.arange(3)),),
            "zeros_like": (sub_arr(np.arange(3)),),
            "copy": (sub_arr(np.arange(3)),),
            "meshgrid": (sub_arr(np.arange(3)), sub_arr(np.arange(3))),
            "transpose": (sub_arr(np.eye(3)),),
            "reshape": (sub_arr(np.arange(3)), (3,)),
            "resize": (sub_arr(np.arange(3)), 6),
            "roll": (sub_arr(np.arange(3)), 1),
            "rollaxis": (sub_arr(np.arange(3)), 0),
            "rot90": (sub_arr(np.eye(3)),),
            "expand_dims": (sub_arr(np.arange(3)), 0),
            "squeeze": (sub_arr(np.arange(3)),),
            "flip": (sub_arr(np.eye(3)),),
            "fliplr": (sub_arr(np.eye(3)),),
            "flipud": (sub_arr(np.eye(3)),),
            "delete": (sub_arr(np.arange(3)), 0),
            "partition": (sub_arr(np.arange(3)), 1),
            "broadcast_to": (sub_arr(np.arange(3)), 3),
            "broadcast_arrays": (sub_arr(np.arange(3)),),
            "split": (sub_arr(np.arange(3)), 1),
            "array_split": (sub_arr(np.arange(3)), 1),
            "dsplit": (sub_arr(np.arange(27)).reshape(3, 3, 3), 1),
            "hsplit": (sub_arr(np.arange(3)), 1),
            "vsplit": (sub_arr(np.eye(3)), 1),
            "swapaxes": (sub_arr(np.eye(3)), 0, 1),
            "moveaxis": (sub_arr(np.eye(3)), 0, 1),
            "nansum": (sub_arr(np.arange(3)),),  # implemented via sum
            "std": (sub_arr(np.arange(3)),),
            "nanstd": (sub_arr(np.arange(3)),),
            "nanvar": (sub_arr(np.arange(3)),),
            "nanprod": (sub_arr(np.arange(3)),),
            "diag": (sub_arr(np.eye(3)),),
            "diag_indices_from": (sub_arr(np.eye(3)),),
            "diagflat": (sub_arr(np.eye(3)),),
            "diagonal": (sub_arr(np.eye(3)),),
            "ravel": (sub_arr(np.arange(3)),),
            "ravel_multi_index": (np.eye(2, dtype=int), (2, 2)),
            "unravel_index": (np.arange(3), (3,)),
            "fix": (sub_arr(np.arange(3)),),
            "round": (sub_arr(np.arange(3)),),  # implemented via around
            "may_share_memory": (
                sub_arr(np.arange(3)),
                sub_arr(np.arange(3)),
            ),
            "linalg.matrix_power": (sub_arr(np.eye(3)), 2),
            "linalg.cholesky": (sub_arr(np.eye(3)),),
            "linalg.multi_dot": ((sub_arr(np.eye(3)), sub_arr(np.eye(3))),),
            "linalg.matrix_rank": (sub_arr(np.eye(3)),),
            "linalg.qr": (sub_arr(np.eye(3)),),
            "linalg.slogdet": (sub_arr(np.eye(3)),),
            "linalg.cond": (sub_arr(np.eye(3)),),
            "gradient": (sub_arr(np.arange(3)),),
            "cumsum": (sub_arr(np.arange(3)),),
            "nancumsum": (sub_arr(np.arange(3)),),
            "nancumprod": (sub_arr(np.arange(3)),),
            "bincount": (sub_arr(np.arange(3)),),
            "unique": (sub_arr(np.arange(3)),),
            "min_scalar_type": (sub_arr(np.arange(3)),),
            "extract": (0, sub_arr(np.arange(3))),
            "setxor1d": (sub_arr(np.arange(3)), sub_arr(np.arange(3))),
            "lexsort": (sub_arr(np.arange(3)),),
            "digitize": (sub_arr(np.arange(3)), sub_arr(np.arange(3))),
            "tril_indices_from": (sub_arr(np.eye(3)),),
            "triu_indices_from": (sub_arr(np.eye(3)),),
            "imag": (sub_arr(np.arange(3)),),
            "real": (sub_arr(np.arange(3)),),
            "real_if_close": (sub_arr(np.arange(3)),),
            "einsum_path": (
                "ij,jk->ik",
                sub_arr(np.eye(3)),
                sub_arr(np.eye(3)),
            ),
            "cov": (sub_arr(np.arange(3)),),
            "corrcoef": (sub_arr(np.arange(3)),),
            "compress": (np.zeros(3), sub_arr(np.arange(3))),
            "take_along_axis": (
                sub_arr(np.arange(3)),
                np.ones(3, dtype=int),
                0,
            ),
            "linalg.cross": (
                sub_arr(np.arange(3)),
                sub_arr(np.arange(3)),
            ),
            "linalg.diagonal": (sub_arr(np.eye(3)),),
            "linalg.matmul": (sub_arr(np.eye(3)), sub_arr(np.eye(3))),
            "linalg.matrix_norm": (sub_arr(np.eye(3)),),
            "linalg.matrix_transpose": (sub_arr(np.eye(3)),),
            "linalg.svdvals": (sub_arr(np.eye(3)),),
            "linalg.tensordot": (
                sub_arr(np.eye(3)),
                sub_arr(np.eye(3)),
            ),
            "linalg.trace": (sub_arr(np.eye(3)),),
            "linalg.vecdot": (
                sub_arr(np.arange(3)),
                sub_arr(np.arange(3)),
            ),
            "linalg.vector_norm": (sub_arr(np.arange(3)),),
            "astype": (sub_arr(np.arange(3)), float),
            "matrix_transpose": (sub_arr(np.eye(3)),),
            "unique_all": (sub_arr(np.arange(3)),),
            "unique_counts": (sub_arr(np.arange(3)),),
            "unique_inverse": (sub_arr(np.arange(3)),),
            "unique_values": (sub_arr(np.arange(3)),),
            "cumulative_sum": (sub_arr(np.arange(3)),),
            "cumulative_prod": (sub_arr(np.arange(3)),),
            "unstack": (sub_arr(np.arange(3)),),
        }
        functions_checked = []
        bad_funcs = {}
        for fname, args in functions_to_check.items():
            ua_args = []
            for arg in args:
                ua_args.append(arg_to_ua(arg))
            func = getfunc(fname)
            try:
                with warnings.catch_warnings():
                    if "savetxt" in fname:
                        warnings.filterwarnings(
                            action="ignore",
                            category=UserWarning,
                            message="numpy.savetxt does not preserve units",
                        )
                    try:
                        ua_result = func(*ua_args)
                    except:
                        print(f"Crashed in {fname} with unyt input.")
                        raise
            except u.exceptions.UnytError:
                raises_unyt_error = True
            else:
                raises_unyt_error = False
            if "savetxt" in fname and os.path.isfile(savetxt_file):
                os.remove(savetxt_file)
            functions_checked.append(func)
            if raises_unyt_error:
                with pytest.raises(u.exceptions.UnytError):
                    result = func(*args)
                continue
            with warnings.catch_warnings():
                if "savetxt" in fname:
                    warnings.filterwarnings(
                        action="ignore",
                        category=UserWarning,
                        message="numpy.savetxt does not preserve units or extra_attr",
                    )
                try:
                    result = func(*args)
                except:
                    print(f"Crashed in {fname} with subclass input.")
                    raise
            if fname.split(".")[-1] in (
                "fill_diagonal",
                "copyto",
                "place",
                "put",
                "put_along_axis",
                "putmask",
            ):
                # treat inplace modified values for relevant functions as result
                result = args[0]
                ua_result = ua_args[0]
            if "savetxt" in fname and os.path.isfile(savetxt_file):
                os.remove(savetxt_file)
            ignore_values = fname in {"empty_like"}  # empty_like has arbitrary data
            try:
                check_result(result, ua_result, ignore_values=ignore_values)
            except AssertionError:
                bad_funcs["np." + fname] = result, ua_result
        if len(bad_funcs) > 0:
            raise AssertionError(
                "Some functions did not return expected types "
                "(obtained, obtained with unyt input): " + str(bad_funcs)
            )
        unchecked_functions = [
            f
            for f in set(_HANDLED_FUNCTIONS) | NOOP_FUNCTIONS
            if f not in functions_checked
        ]
        try:
            assert len(unchecked_functions) == 0
        except AssertionError:
            raise AssertionError(
                "Did not check functions",
                [
                    (".".join((f.__module__, f.__name__)).replace("numpy", "np"))
                    for f in unchecked_functions
                ],
            )

    @pytest.mark.parametrize(
        "func_args",
        (
            (
                np.histogram,
                (
                    subclass_uarray(
                        [1, 2, 3],
                        u.m,
                        extra_attr=True,
                    ),
                ),
            ),
            (
                np.histogram2d,
                (
                    subclass_uarray(
                        [1, 2, 3],
                        u.m,
                        extra_attr=True,
                    ),
                    subclass_uarray(
                        [1, 2, 3],
                        u.K,
                        extra_attr=True,
                    ),
                ),
            ),
            (
                np.histogramdd,
                (
                    [
                        subclass_uarray(
                            [1, 2, 3],
                            u.m,
                            extra_attr=True,
                        ),
                        subclass_uarray(
                            [1, 2, 3],
                            u.K,
                            extra_attr=True,
                        ),
                        subclass_uarray(
                            [1, 2, 3],
                            u.kg,
                            extra_attr=True,
                        ),
                    ],
                ),
            ),
        ),
    )
    @pytest.mark.parametrize(
        "weights",
        (
            None,
            subclass_uarray([1, 2, 3], u.s, extra_attr=True),
            np.array([1, 2, 3]),
        ),
    )
    @pytest.mark.parametrize("bins_type", ("int", "np", "sub_arr"))
    @pytest.mark.parametrize("density", (None, True))
    def test_histograms(self, func_args, weights, bins_type, density):
        """
        Test that histograms give sensible output.

        Histograms are tricky with possible density and weights arguments, and the way
        that attributes need validation and propagation between the bins and values.
        They are also commonly used. They therefore need a bespoke test.
        """
        func, args = func_args
        bins = {
            "int": 10,
            "np": [np.linspace(0, 5, 11)] * 3,
            "sub_arr": [
                subclass_uarray(
                    np.linspace(0, 5, 11),
                    u.kpc,
                    extra_attr=True,
                ),
                subclass_uarray(
                    np.linspace(0, 5, 11),
                    u.K,
                    extra_attr=True,
                ),
                subclass_uarray(
                    np.linspace(0, 5, 11),
                    u.Msun,
                    extra_attr=True,
                ),
            ],
        }[bins_type]
        bins = (
            bins[
                {
                    np.histogram: np.s_[0],
                    np.histogram2d: np.s_[:2],
                    np.histogramdd: np.s_[:],
                }[func]
            ]
            if bins_type in ("np", "sub_arr")
            else bins
        )
        result = func(*args, bins=bins, density=density, weights=weights)
        ua_args = tuple(
            (
                to_ua(arg)
                if not isinstance(arg, tuple)
                else tuple(to_ua(item) for item in arg)
            )
            for arg in args
        )
        ua_bins = (
            to_ua(bins)
            if not isinstance(bins, tuple)
            else tuple(to_ua(item) for item in bins)
        )
        ua_result = func(
            *ua_args, bins=ua_bins, density=density, weights=to_ua(weights)
        )
        if isinstance(ua_result, tuple):
            assert isinstance(result, tuple)
            assert len(result) == len(ua_result)
            for r, ua_r in zip(result, ua_result):
                check_result(r, ua_r)
        else:
            check_result(result, ua_result)
        if not density and not isinstance(weights, subclass_uarray):
            assert not isinstance(result[0], subclass_uarray)
        else:
            assert result[0].extra_attr is True
        if density and not isinstance(weights, subclass_uarray):
            pass
        elif density and isinstance(weights, subclass_uarray):
            assert result[0].extra_attr is True
        elif not density and isinstance(weights, subclass_uarray):
            assert result[0].extra_attr is True
        ret_bins = {
            np.histogram: [result[1]],
            np.histogram2d: result[1:],
            np.histogramdd: result[1],
        }[func]
        for b in ret_bins:
            assert b.extra_attr is True

    def test_getitem(self):
        """
        Make sure that we don't degrade to an ndarray on slicing.
        """
        assert isinstance(sub_arr(np.arange(3))[0], subclass_uquantity)

    def test_reshape_to_scalar(self):
        """
        Make sure that we convert to a subclass_uquantity when we reshape to a scalar.
        """
        assert isinstance(sub_arr(np.ones(1)).reshape(()), subclass_uquantity)

    def test_iter(self):
        """
        Make sure that we get subclass_uquantity's when iterating over a subclass_uarray.
        """
        for sub_q in sub_arr(np.arange(3)):
            assert isinstance(sub_q, subclass_uquantity)

    def test_dot(self):
        """
        Make sure that we get a subclass_uarray when we use array attribute dot.
        """
        res = sub_arr(np.arange(3)).dot(sub_arr(np.arange(3)))
        assert isinstance(res, subclass_uquantity)
        assert res.extra_attr is True


class TestSubclassQuantity:
    """
    Test that the subclass_uquantity class works as desired, mostly around issues
    converting back and forth with subclass_uarray.
    """

    @pytest.mark.parametrize(
        "func, args",
        [
            ("astype", (float,)),
            ("in_units", (u.m,)),
            ("byteswap", ()),
            ("compress", ([True],)),
            ("flatten", ()),
            ("ravel", ()),
            ("repeat", (1,)),
            ("reshape", (1,)),
            ("take", ([0],)),
            ("transpose", ()),
            ("view", ()),
        ],
    )
    def test_propagation_func(self, func, args):
        """
        Test that functions that are supposed to propagate our attribute do so.
        """
        sub_q = subclass_uquantity(
            1,
            u.m,
            extra_attr=True,
        )
        res = getattr(sub_q, func)(*args)
        assert res.extra_attr is True

    def test_round(self):
        """
        Test that attributes propagate through the round builtin.
        """
        sub_q = subclass_uquantity(
            1.03,
            u.m,
            extra_attr=True,
        )
        res = round(sub_q)
        assert res.value == 1.0
        assert res.extra_attr is True

    def test_scalar_return_func(self):
        """
        Make sure that default-wrapped functions that take a subclass_uarray and return a
        scalar convert to a subclass_uquantity.
        """
        sub_arr = subclass_uarray(
            np.arange(3),
            u.m,
            extra_attr=True,
        )
        res = np.min(sub_arr)
        assert isinstance(res, subclass_uquantity)

    @pytest.mark.parametrize("prop", ["T", "ua", "unit_array"])
    def test_propagation_props(self, prop):
        """
        Test that properties propagate our attributes as intended.
        """
        sub_q = subclass_uquantity(
            1,
            u.m,
            extra_attr=True,
        )
        res = getattr(sub_q, prop)
        assert res.extra_attr is True

    def test_multiply_quantities(self):
        """
        Test multiplying two quantities.
        """
        sub_q = subclass_uquantity(
            2,
            u.m,
            extra_attr=True,
        )
        multiplied = sub_q * sub_q
        assert type(multiplied) is subclass_uquantity
        assert multiplied.extra_attr is True
        assert multiplied.to_value(u.m**2) == 4


class TestCosmoArrayCopy:
    """
    Tests of explicit (deep)copying of subclass_uarray.
    """

    def test_copy(self):
        """
        Check that when we copy a subclass_uarray it preserves its values and attributes.
        """
        units = u.Mpc
        arr = subclass_uarray(
            u.unyt_array(np.ones(5), units=units),
            extra_attr=True,
        )
        copy_arr = copy(arr)
        assert np.allclose(arr.to_value(units), copy_arr.to_value(units))
        assert arr.units == copy_arr.units
        assert arr.extra_attr == copy_arr.extra_attr

    def test_deepcopy(self):
        """
        Check that when we deepcopy a subclass_uarray it preserves its values and
        attribute
        """
        units = u.Mpc
        arr = subclass_uarray(
            u.unyt_array(np.ones(5), units=units),
            extra_attr=True,
        )
        copy_arr = deepcopy(arr)
        assert np.allclose(arr.to_value(units), copy_arr.to_value(units))
        assert arr.units == copy_arr.units
        assert arr.extra_attr == copy_arr.extra_attr

    def test_to_cgs(self):
        """
        Check that using to_cgs properly preserves attributes.
        """
        units = u.Mpc
        arr = subclass_uarray(
            u.unyt_array(np.ones(5), units=units),
            extra_attr=True,
        )
        cgs_arr = arr.in_cgs()
        assert np.allclose(arr.to_value(u.cm), cgs_arr.to_value(u.cm))
        assert cgs_arr.units == u.cm
        assert cgs_arr.extra_attr == arr.extra_attr


class TestMultiplicationByUnyt:
    @pytest.mark.parametrize(
        "sub_arr",
        [
            subclass_uarray(
                np.ones(3),
                u.Mpc,
                extra_attr=True,
            ),
            subclass_uquantity(
                np.ones(1),
                u.Mpc,
                extra_attr=True,
            ),
        ],
    )
    def test_multiplication_by_unyt(self, sub_arr):
        """
        We desire consistent behaviour for example for
        `subclass_uarray(...) * (1 * u.Mpc)` as for
        `subclass_uarray(...) * u.Mpc`.
        """

        lmultiplied_by_quantity = sub_arr * (
            1 * u.Mpc
        )  # parentheses very important here
        lmultiplied_by_unyt = sub_arr * u.Mpc
        assert isinstance(lmultiplied_by_quantity, subclass_uarray)
        assert isinstance(lmultiplied_by_unyt, subclass_uarray)
        assert lmultiplied_by_unyt.extra_attr == lmultiplied_by_quantity.extra_attr
        assert np.allclose(
            lmultiplied_by_unyt.to_value(lmultiplied_by_quantity.units),
            lmultiplied_by_quantity.to_value(lmultiplied_by_quantity.units),
        )

        ldivided_by_quantity = sub_arr / (1 * u.Mpc)  # parentheses very important here
        ldivided_by_unyt = sub_arr / u.Mpc
        assert isinstance(ldivided_by_quantity, subclass_uarray)
        assert isinstance(ldivided_by_unyt, subclass_uarray)
        assert ldivided_by_unyt.extra_attr == ldivided_by_quantity.extra_attr
        assert np.allclose(
            ldivided_by_unyt.to_value(ldivided_by_quantity.units),
            ldivided_by_quantity.to_value(ldivided_by_quantity.units),
        )

        rmultiplied_by_quantity = (
            1 * u.Mpc
        ) * sub_arr  # parentheses very important here
        assert rmultiplied_by_quantity.extra_attr
        rmultiplied_by_unyt = u.Mpc * sub_arr
        assert isinstance(rmultiplied_by_quantity, subclass_uarray)
        assert isinstance(rmultiplied_by_unyt, subclass_uarray)
        assert rmultiplied_by_unyt.extra_attr == rmultiplied_by_quantity.extra_attr
        assert np.allclose(
            rmultiplied_by_unyt.to_value(rmultiplied_by_quantity.units),
            rmultiplied_by_quantity.to_value(rmultiplied_by_quantity.units),
        )

        rdivided_by_quantity = (1 * u.Mpc) / sub_arr  # parentheses very important here
        rdivided_by_unyt = u.Mpc / sub_arr
        assert isinstance(rdivided_by_quantity, subclass_uarray)
        assert isinstance(rdivided_by_unyt, subclass_uarray)
        assert rdivided_by_unyt.extra_attr == rdivided_by_quantity.extra_attr
        assert np.allclose(
            rdivided_by_unyt.to_value(rdivided_by_quantity.units),
            rdivided_by_quantity.to_value(rdivided_by_quantity.units),
        )
