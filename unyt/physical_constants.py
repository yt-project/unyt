"""
Predefined useful physical constants

Note that all of these names can be imported from the top-level unyt namespace.
For example::

    >>> from unyt import c, G, kb

"""

# -----------------------------------------------------------------------------
# Copyright (c) 2018, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------


from math import pi as _pi

from unyt._physical_ratios import (
    mass_electron_kg as _mass_electron_kg,
    amu_kg as _amu_kg,
    mass_hydrogen_kg as _mass_hydrogen_kg,
    speed_of_light_m_per_s as _speed_of_light_m_per_s,
    boltzmann_constant_J_per_K as _boltzmann_constant_J_per_K,
    mass_sun_kg as _mass_sun_kg,
    mass_jupiter_kg as _mass_jupiter_kg,
    mass_mercury_kg as _mass_mercury_kg,
    mass_venus_kg as _mass_venus_kg,
    mass_earth_kg as _mass_earth_kg,
    mass_mars_kg as _mass_mars_kg,
    mass_saturn_kg as _mass_saturn_kg,
    mass_uranus_kg as _mass_uranus_kg,
    mass_neptune_kg as _mass_neptune_kg,
    planck_mass_kg as _planck_mass_kg,
    planck_length_m as _planck_length_m,
    planck_time_s as _planck_time_s,
    planck_energy_J as _planck_energy_J,
    planck_charge_C as _planck_charge_C,
    planck_temperature_K as _planck_temperature_K,
    standard_gravity_m_per_s2 as _standard_gravity_m_per_s2,
    newton_mks as _newton_mks,
    planck_mks as _planck_mks,
    eps_0 as _eps_0,
    mu_0 as _mu_0
)
from unyt.array import unyt_quantity

#: mass of the electron
mass_electron_mks = mass_electron = me = \
    unyt_quantity(_mass_electron_kg, 'kg')

#: atomic mass unit
amu_mks = amu = unyt_quantity(_amu_kg, 'kg')

#: Avogadro's number
Na = avogadros_number = unyt_quantity(6.02214085774*10**23, 'mol**-1')

#: Mass of hydrogen
mp = mh = mass_hydrogen = mass_hydrogen_mks = \
    unyt_quantity(_mass_hydrogen_kg, 'kg')

# Velocities
#: speed of light
c = clight = speed_of_light = speed_of_light_mks = \
    unyt_quantity(_speed_of_light_m_per_s, 'm/s')

# Cross Sections
#: Thompson cross section (8*pi/3 (alpha*hbar*c/(2*pi))**2)
sigma_thompson = thompson_cross_section = cross_section_thompson = \
    cross_section_thompson_mks = unyt_quantity(6.65245854533e-29, 'm**2')

# Charge
#: Charge of the proton
qp = elementary_charge = proton_charge = charge_proton = charge_proton_mks = \
    unyt_quantity(1.6021766208e-19, 'C')
#: Charge of the electron
electron_charge = charge_electron = charge_electron_mks = -qp

qp_cgs = elementary_charge_cgs = proton_charge_cgs = charge_proton_cgs = \
    unyt_quantity(4.8032056e-10, 'esu')
electron_charge_cgs = charge_electron_cgs = -charge_proton_cgs

# Physical Constants
#: Boltzmann constant
kb = kboltz = boltzmann_constant = boltzmann_constant_mks = unyt_quantity(
    _boltzmann_constant_J_per_K, 'J/K')

#: Gravitational constant
G = newtons_constant = gravitational_constant = gravitational_constant_mks = \
    unyt_quantity(_newton_mks, 'm**3/kg/s**2')

#: Planck constant
hmks = planck_constant = planck_constant_mks = \
    unyt_quantity(_planck_mks, 'J*s')

#: Reduced Planck constant
hbar = reduced_planck_constant = 0.5*hmks/_pi

#: Stefan-Boltzmann constant
stefan_boltzmann_constant = stefan_boltzmann_constant_mks = unyt_quantity(
    5.670373e-8, 'W/m**2/K**4')

#: Current CMB temperature
CMD_temperature = Tcmb = unyt_quantity(2.726, 'K')

# Solar System
#: Mass of the sun
msun = solar_mass = mass_sun = mass_sun_mks = unyt_quantity(
    _mass_sun_kg, 'kg')

#: Mass of Jupiter
Mjup = mjup = jupiter_mass = mass_jupiter = mass_jupiter_mks = unyt_quantity(
    _mass_jupiter_kg, 'kg')

#: Mass of Mercury
mercury_mass = mass_mercury = mass_mercury_mks = unyt_quantity(
    _mass_mercury_kg, 'kg')

#: Mass of Venus
venus_mass = mass_venus = mass_venus_mks = unyt_quantity(
    _mass_venus_kg, 'kg')

#: Mass of Earth
Mearth = mearth = earth_mass = mass_earth = mass_earth_mks = unyt_quantity(
    _mass_earth_kg, 'kg')

#: Mass of Mars
mars_mass = mass_mars = mass_mars_mks = unyt_quantity(_mass_mars_kg, 'kg')

#: Mass of Saturn
saturn_mass = mass_saturn = mass_saturn_mks = unyt_quantity(
    _mass_saturn_kg, 'kg')

#: Mass of Uranus
uranus_mass = mass_uranus = mass_uranus_mks = unyt_quantity(
    _mass_uranus_kg, 'kg')

#: Mass of Neptune
neptune_mass = mass_neptune = mass_neptune_mks = unyt_quantity(
    _mass_neptune_kg, 'kg')

# Planck units
#: Planck mass
m_pl = planck_mass = unyt_quantity(_planck_mass_kg, "g")
#: Planck length
l_pl = planck_length = unyt_quantity(_planck_length_m, "cm")
#: Planck time
t_pl = planck_time = unyt_quantity(_planck_time_s, "s")
#: Planck energy
E_pl = planck_energy = unyt_quantity(_planck_energy_J, "erg")
#: Planck charge
q_pl = planck_charge = unyt_quantity(_planck_charge_C, "C")
#: Planck tempearture
T_pl = planck_temperature = unyt_quantity(_planck_temperature_K, "K")

# MKS E&M constants
#: Permeability of Free Space
mu_0 = unyt_quantity(_mu_0, "N/A**2")
#: Permittivity of Free Space
eps_0 = unyt_quantity(_eps_0, "C**2/N/m**2")

# Misc
#: Standard gravitational acceleration
standard_gravity = standard_gravity_mks = unyt_quantity(
    _standard_gravity_m_per_s2, "m/s**2")
