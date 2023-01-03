"""
A set of convenient on-demand imports
"""


from functools import wraps
from importlib.util import find_spec
from typing import Type


class NotAModule:
    """
    A class to implement an informative error message that will be outputted if
    someone tries to use an on-demand import without having the requisite
    package installed.
    """

    def __init__(self, pkg_name):
        self.pkg_name = pkg_name
        self.error = ImportError(
            f"This functionality requires the {self.pkg_name} package to be installed."
        )

    def __getattr__(self, item):
        raise self.error

    def __call__(self, *args, **kwargs):
        raise self.error

    def __repr__(self) -> str:
        return f"NotAModule({self.pkg_name!r})"


class OnDemand:
    _default_factory: Type[NotAModule] = NotAModule

    def __init_subclass__(cls):
        if not cls.__name__.endswith("_imports"):
            raise TypeError(f"class {cls}'s name needs to be suffixed '_imports'")

    def __new__(cls):
        if cls is OnDemand:
            raise TypeError("The OnDemand base class cannot be instantiated.")
        else:
            return object.__new__(cls)

    @property
    def _name(self) -> str:
        _name, _, _suffix = self.__class__.__name__.rpartition("_")
        return _name

    @property
    def __is_available__(self) -> bool:

        return find_spec(self._name) is not None


def safe_import(func):
    @property
    @wraps(func)
    def inner(self):
        try:
            return func(self)
        except ImportError:
            return self._default_factory(self._name)

    return inner


class astropy_imports(OnDemand):
    @safe_import
    def log(self):
        from astropy import log

        if log.exception_logging_enabled():
            log.disable_exception_logging()

        return log

    @safe_import
    def units(self):
        from astropy import units

        self.log
        return units

    @safe_import
    def __version__(self):
        from astropy import __version__

        return __version__


_astropy = astropy_imports()


class h5py_imports(OnDemand):
    @safe_import
    def File(self):
        from h5py import File

        return File

    @safe_import
    def __version__(self):
        from h5py import __version__

        return __version__


_h5py = h5py_imports()


class pint_imports(OnDemand):
    @safe_import
    def UnitRegistry(self):
        from pint import UnitRegistry

        return UnitRegistry


_pint = pint_imports()


class matplotlib_imports(OnDemand):
    @safe_import
    def __version__(self):
        from matplotlib import __version__

        return __version__

    @safe_import
    def pyplot(self):
        from matplotlib import pyplot

        return pyplot

    @safe_import
    def units(self):
        from matplotlib import units

        return units

    @safe_import
    def use(self):
        from matplotlib import use

        return use


_matplotlib = matplotlib_imports()


class dask_imports(OnDemand):
    @safe_import
    def array(self):
        from dask import array

        return array

    @safe_import
    def __version__(self):
        from dask import __version__

        return __version__


_dask = dask_imports()
