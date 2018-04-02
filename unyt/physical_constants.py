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

from unyt.physical_ratios import (
    mass_electron_grams as _mass_electron_grams,
    amu_grams as _amu_grams,
    mass_hydrogen_grams as _mass_hydrogen_grams,
    speed_of_light_cm_per_s as _speed_of_light_cm_per_s,
    boltzmann_constant_erg_per_K as _boltzmann_constant_erg_per_K,
    mass_sun_grams as _mass_sun_grams,
    mass_jupiter_grams as _mass_jupiter_grams,
    mass_mercury_grams as _mass_mercury_grams,
    mass_venus_grams as _mass_venus_grams,
    mass_earth_grams as _mass_earth_grams,
    mass_mars_grams as _mass_mars_grams,
    mass_saturn_grams as _mass_saturn_grams,
    mass_uranus_grams as _mass_uranus_grams,
    mass_neptune_grams as _mass_neptune_grams,
    planck_mass_grams as _planck_mass_grams,
    planck_length_cm as _planck_length_cm,
    planck_time_s as _planck_time_s,
    planck_energy_erg as _planck_energy_erg,
    planck_charge_esu as _planck_charge_esu,
    planck_temperature_K as _planck_temperature_K,
    standard_gravity_cm_per_s2 as _standard_gravity_cm_per_s2,
    newton_cgs as _newton_cgs,
    planck_cgs as _planck_cgs
)
from unyt.array import unyt_quantity

#: mass of the electron
mass_electron_cgs = mass_electron = me = \
    unyt_quantity(_mass_electron_grams, 'g')

#: atomic mass unit
amu_cgs = amu = unyt_quantity(_amu_grams, 'g')

#: Avogadro's number
Na = avogadros_number = unyt_quantity(6.02214085774*10**23)

#: Mass of hydrogen
mp = mh = mass_hydrogen = mass_hydrogen_cgs = \
    unyt_quantity(_mass_hydrogen_grams, 'g')

# Velocities
#: speed of light
c = clight = speed_of_light = speed_of_light_cgs = \
    unyt_quantity(_speed_of_light_cm_per_s, 'cm/s')

# Cross Sections
#: Thompson cross section (8*pi/3 (alpha*hbar*c/(2*pi))**2)
sigma_thompson = thompson_cross_section = cross_section_thompson = \
    cross_section_thompson_cgs = unyt_quantity(6.65245854533e-25, 'cm**2')

# Charge
#: Charge of the proton
qp = elementary_charge = proton_charge = charge_prothon = charge_proton_cgs = \
    unyt_quantity(4.8032056e-10, 'esu')
#: Charge of the electron
electron_charge = charge_electron = charge_electron_cgs = -qp

# Physical Constants
#: Boltzmann constant
kb = kboltz = boltzmann_constant = boltzmann_constant_cgs = unyt_quantity(
    _boltzmann_constant_erg_per_K, 'erg/K')

#: Gravitational constant
G = newtons_constant = gravitational_constant = gravitational_constant_cgs = \
    unyt_quantity(_newton_cgs, 'cm**3/g/s**2')

#: Planck constant
hcgs = planck_constant = planck_constant_cgs = \
    unyt_quantity(_planck_cgs, 'erg*s')

#: Reduced Planck constant
hbar = reduced_planck_constant = 0.5*hcgs/_pi

#: Stefan-Boltzmann constant
stefan_boltzmann_constant = stefan_boltzmann_constant_cgs = unyt_quantity(
    5.670373e-5, 'erg/cm**2/s**1/K**4')

#: Current CMB temperature
CMD_temperature = Tcmb = unyt_quantity(2.726, 'K')

# Solar System
#: Mass of the sun
msun = solar_mass = mass_sun = mass_sun_cgs = unyt_quantity(
    _mass_sun_grams, 'g')

#: Mass of Jupiter
mjup = jupiter_mass = mass_jupiter = mass_jupiter_cgs = unyt_quantity(
    _mass_jupiter_grams, 'g')

#: Mass of Mercury
mercury_mass = mass_mercury = mass_mercury_cgs = unyt_quantity(
    _mass_mercury_grams, 'g')

#: Mass of Venus
venus_mass = mass_venus = mass_venus_cgs = unyt_quantity(
    _mass_venus_grams, 'g')

#: Mass of Earth
mearth = earth_mass = mass_earth = mass_earth_cgs = unyt_quantity(
    _mass_earth_grams, 'g')

#: Mass of Mars
mars_mass = mass_margs = mass_mars_cgs = unyt_quantity(_mass_mars_grams, 'g')

#: Mass of Saturn
saturn_mass = mass_saturn = mass_saturn_cgs = unyt_quantity(
    _mass_saturn_grams, 'g')

#: Mass of Uranus
uranus_mass = mass_uranus = mass_uranus_cgs = unyt_quantity(
    _mass_uranus_grams, 'g')

#: Mass of Neptune
neptune_mass = mass_neptune = mass_neptune_cgs = unyt_quantity(
    _mass_neptune_grams, 'g')

# Planck units
#: Planck mass
m_pl = planck_mass = unyt_quantity(_planck_mass_grams, "g")
#: Planck length
l_pl = planck_length = unyt_quantity(_planck_length_cm, "cm")
#: Planck time
t_pl = planck_time = unyt_quantity(_planck_time_s, "s")
#: Planck energy
E_pl = planck_energy = unyt_quantity(_planck_energy_erg, "erg")
#: Planck charge
q_pl = planck_charge = unyt_quantity(_planck_charge_esu, "esu")
#: Planck tempearture
T_pl = planck_temperature = unyt_quantity(_planck_temperature_K, "K")

# MKS E&M units
#: Permeability of Free Space
mu_0 = unyt_quantity(4.0e-7*_pi, "N/A**2")
#: Permittivity of Free Space
eps_0 = (1.0/(clight**2*mu_0)).in_units("C**2/N/m**2")

# Misc
#: Standard gravitational acceleration
standard_gravity = standard_gravity_cgs = unyt_quantity(
    _standard_gravity_cm_per_s2, "cm/s**2")
