"""
Equivalencies between different kinds of units

"""

# -----------------------------------------------------------------------------
# Copyright (c) 2018, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------

from collections import OrderedDict

from unyt.dimensions import (
    temperature,
    mass,
    energy,
    length,
    rate,
    velocity,
    dimensionless,
    density,
    number_density,
    flux,
    current_cgs,
    current_mks,
    charge_cgs,
    charge_mks,
    magnetic_field_cgs,
    magnetic_field_mks,
    electric_potential_cgs,
    electric_potential_mks,
    electric_field_cgs,
    electric_field_mks,
    resistance_cgs,
    resistance_mks
)

from unyt._physical_ratios import speed_of_light_cm_per_s
from six import add_metaclass
import numpy as np

equivalence_registry = OrderedDict()


class _RegisteredEquivalence(type):
    def __init__(cls, name, b, d):
        type.__init__(cls, name, b, d)
        if hasattr(cls, "type_name"):
            equivalence_registry[cls.type_name] = cls
        if hasattr(cls, "alternate_names"):
            for name in cls.alternate_names:
                equivalence_registry[name] = cls


@add_metaclass(_RegisteredEquivalence)
class Equivalence(object):
    one_way = False


class NumberDensityEquivalence(Equivalence):
    """Equivalence between mass and number density, given a mean molecular weight.

    Given a number density :math:`n`, the mass density :math:`\\rho` is:

    .. math::

      \\rho = \\mu m_{\\rm H} n

    And similarly

    .. math::

      n = \\rho (\\mu m_{\\rm H})^{-1}

    Parameters
    ----------
    mu : float
      The mean molecular weight. Defaults to 0.6 whcih is valid for fully
      ionized gas with primordial composition.

    Example
    -------
    >>> print(NumberDensityEquivalence())
    number density: density <-> number density
    >>> from unyt import Msun, pc
    >>> rho = Msun/pc**3
    >>> rho.to_equivalent('cm**-3', 'number_density', mu=1.4)
    unyt_quantity(28.88289965, 'cm**(-3)')
    """
    type_name = "number_density"
    _dims = (density, number_density,)

    def _convert(self, x, new_dims, mu=0.6):
        from unyt import physical_constants as pc
        if new_dims == number_density:
            return x/(mu*pc.mh)
        elif new_dims == density:
            return x*mu*pc.mh

    def __str__(self):
        return "number density: density <-> number density"


class ThermalEquivalence(Equivalence):
    """Equivalence between temperature and energy via the Boltzmann constant

    Given a temperature :math:`T` in an absolute scale (e.g. Kelvin or
    Rankine), the equivalent thermal energy :math:`E` for that temperature is
    given by:

    .. math::

      E = k_B T

    And

    .. math::

      T = E/k_B

    Where :math:`k_B` is Boltzmann's constant.

    Example
    -------
    >>> print(ThermalEquivalence())
    thermal: temperature <-> energy
    >>> from unyt import Kelvin
    >>> temp = 1e6*Kelvin
    >>> temp.to_equivalent('keV', 'thermal')
    unyt_quantity(0.08617332, 'keV')
    """
    type_name = "thermal"
    _dims = (temperature, energy,)

    def _convert(self, x, new_dims):
        from unyt import physical_constants as pc
        if new_dims == energy:
            return pc.kboltz*x
        elif new_dims == temperature:
            return x/pc.kboltz

    def __str__(self):
        return "thermal: temperature <-> energy"


class MassEnergyEquivalence(Equivalence):
    """Equivalence between mass and energy in special relativity

    Given a body with mass :math:`m`, the self-energy :math:`E` of that mass is
    given by

    .. math::

      E = m c^2

    where :math:`c` is the speed of light.

    Example
    -------
    >>> print(MassEnergyEquivalence())
    mass_energy: mass <-> energy
    >>> from unyt import g
    >>> g.to_equivalent('J', 'mass_energy')
    unyt_quantity(8.98755179e+13, 'J')

    """
    type_name = "mass_energy"
    _dims = (mass, energy,)

    def _convert(self, x, new_dims):
        from unyt import physical_constants as pc
        if new_dims == energy:
            return x*pc.clight*pc.clight
        elif new_dims == mass:
            return x/(pc.clight*pc.clight)

    def __str__(self):
        return "mass_energy: mass <-> energy"


class SpectralEquivalence(Equivalence):
    """Equivalence between wavelength, frequency, and energy of a photon.

    Given a photon with wavelength :math:`\\lambda`, frequency :math:`\\nu`
    and Energy :math:`E`, these quantities are related by the following
    forumlae:

    .. math::

      E = h \\nu = h c / \\lambda

    where :math:`h` is Planck's constant and :math:`c` is the speed of light.

    Example
    ------
    >>> print(SpectralEquivalence())
    spectral: length <-> frequency <-> energy
    >>> from unyt import angstrom, km
    >>> angstrom.to_equivalent('keV', 'spectral')
    unyt_quantity(12.39841932, 'keV')
    >>> km.to_equivalent('MHz', 'spectral')
    unyt_quantity(0.29979246, 'MHz')
    """
    type_name = "spectral"
    _dims = (length, rate, energy,)

    def _convert(self, x, new_dims):
        from unyt import physical_constants as pc
        if new_dims == energy:
            if x.units.dimensions == length:
                nu = pc.clight/x
            elif x.units.dimensions == rate:
                nu = x
            return pc.hcgs*nu
        elif new_dims == length:
            if x.units.dimensions == rate:
                return pc.clight/x
            elif x.units.dimensions == energy:
                return pc.hcgs*pc.clight/x
        elif new_dims == rate:
            if x.units.dimensions == length:
                return pc.clight/x
            elif x.units.dimensions == energy:
                return x/pc.hcgs

    def __str__(self):
        return "spectral: length <-> frequency <-> energy"


class SoundSpeedEquivalence(Equivalence):
    type_name = "sound_speed"
    _dims = (velocity, temperature, energy,)

    def _convert(self, x, new_dims, mu=0.6, gamma=5./3.):
        from unyt import physical_constants as pc
        if new_dims == velocity:
            if x.units.dimensions == temperature:
                kT = pc.kboltz*x
            elif x.units.dimensions == energy:
                kT = x
            return np.sqrt(gamma*kT/(mu*pc.mh))
        else:
            kT = x*x*mu*pc.mh/gamma
            if new_dims == temperature:
                return kT/pc.kboltz
            else:
                return kT

    def __str__(self):
        return "sound_speed (ideal gas): velocity <-> temperature <-> energy"


class LorentzEquivalence(Equivalence):
    type_name = "lorentz"
    _dims = (dimensionless, velocity,)

    def _convert(self, x, new_dims):
        from unyt import physical_constants as pc
        if new_dims == dimensionless:
            beta = x.in_cgs()/pc.clight
            return 1./np.sqrt(1.-beta**2)
        elif new_dims == velocity:
            return pc.clight*np.sqrt(1.-1./(x*x))

    def __str__(self):
        return "lorentz: velocity <-> dimensionless"


class SchwarzschildEquivalence(Equivalence):
    """Equivalence between the mass and radius of a Schwarzschild black hole

    A Schwarzschild black hole of mass :math:`M` has radius :math:`R`

    .. math::

      R = \\frac{2 G M}{c^2}

    and similarly

    .. math::

      M = \\frac{R c^2}{2 G}

    where :math:`G` is Newton's gravitational constant and :math:`c` is the
    speed of light.

    Example
    -------
    >>> from unyt import Msun
    >>> Msun.to_equivalent('km', 'schwarzschild')
    unyt_quantity(2.95305543, 'km')
    """
    type_name = "schwarzschild"
    _dims = (mass, length,)

    def _convert(self, x, new_dims):
        from unyt import physical_constants as pc
        if new_dims == length:
            return 2.*pc.G*x/(pc.clight*pc.clight)
        elif new_dims == mass:
            return 0.5*x*pc.clight*pc.clight/pc.G

    def __str__(self):
        return "schwarzschild: mass <-> length"


class ComptonEquivalence(Equivalence):
    type_name = "compton"
    _dims = (mass, length,)

    def _convert(self, x, new_dims):
        from unyt import physical_constants as pc
        return pc.hcgs/(x*pc.clight)

    def __str__(self):
        return "compton: mass <-> length"


class EffectiveTemperature(Equivalence):
    type_name = "effective_temperature"
    _dims = (flux, temperature,)

    def _convert(self, x, new_dims):
        from unyt import physical_constants as pc
        if new_dims == flux:
            return pc.stefan_boltzmann_constant_cgs*x**4
        elif new_dims == temperature:
            return (x/pc.stefan_boltzmann_constant_cgs)**0.25

    def __str__(self):
        return "effective_temperature: flux <-> temperature"


em_conversions = {
    charge_mks: ("esu", 0.1*speed_of_light_cm_per_s),
    magnetic_field_mks: ("gauss", 1.0e4),
    current_mks: ("statA", 0.1*speed_of_light_cm_per_s),
    electric_potential_mks: ("statV", 1.0e-8*speed_of_light_cm_per_s),
    resistance_mks: ("statohm", 1.0e9/(speed_of_light_cm_per_s**2)),
    charge_cgs: ("C", 10.0/speed_of_light_cm_per_s),
    magnetic_field_cgs: ("T", 1.0e-4),
    current_cgs: ("A", 10.0/speed_of_light_cm_per_s),
    electric_potential_cgs: ("V", 1.0e8/speed_of_light_cm_per_s),
    resistance_cgs: ("ohm", speed_of_light_cm_per_s**2*1.0e-9),
}


class ElectromagneticSI(Equivalence):
    type_name = "SI"
    alternate_names = ["si", "MKS", "mks"]
    one_way = True
    _dims = (current_cgs, charge_cgs, magnetic_field_cgs,
             electric_field_cgs, electric_potential_cgs,
             resistance_cgs)

    def _convert(self, x, new_dims):
        old_dims = x.units.dimensions
        new_units, convert_factor = em_conversions[old_dims]
        return x.in_cgs().v*convert_factor, new_units

    def __str__(self):
        return "SI: EM CGS unit -> EM SI unit"


class ElectromagneticCGS(Equivalence):
    type_name = "CGS"
    alternate_names = ["cgs"]
    one_way = True
    _dims = (current_mks, charge_mks, magnetic_field_mks,
             electric_field_mks, electric_potential_mks,
             resistance_mks)

    def _convert(self, x, new_dims):
        old_dims = x.units.dimensions
        new_units, convert_factor = em_conversions[old_dims]
        return x.in_mks().v*convert_factor, new_units

    def __str__(self):
        return "CGS: EM SI unit -> EM CGS unit"
