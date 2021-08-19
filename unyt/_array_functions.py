import numpy as np

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
    return np.array2string._implementation(a, *args, **kwargs) + f" {a.units}"


@implements(np.linalg.inv)
def linalg_inv(a, *args, **kwargs):
    return np.linalg.inv._implementation(a, *args, **kwargs).ndview / a.units
