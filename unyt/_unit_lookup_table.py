"""
The default unit symbol lookup table.


"""

# -----------------------------------------------------------------------------
# Copyright (c) 2018, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------

from unyt import dimensions
from unyt._physical_ratios import (
    m_per_pc,
    m_per_ly,
    m_per_au,
    m_per_rsun,
    m_per_inch,
    m_per_ft,
    watt_per_horsepower,
    mass_sun_kg,
    sec_per_year,
    sec_per_day,
    sec_per_hr,
    sec_per_min,
    temp_sun_kelvin,
    luminosity_sun_watts,
    metallicity_sun,
    J_per_eV,
    amu_kg,
    amu_grams,
    mass_electron_kg,
    m_per_ang,
    jansky_mks,
    mass_jupiter_kg,
    mass_earth_kg,
    kelvin_per_rankine,
    speed_of_light_m_per_s,
    planck_length_m,
    planck_charge_C,
    planck_energy_J,
    planck_mass_kg,
    planck_temperature_K,
    planck_time_s,
    mass_hydrogen_kg,
    kg_per_pound,
    standard_gravity_m_per_s2,
    pascal_per_atm,
    newton_mks,
    m_per_rearth,
    m_per_rjup
)
import numpy as np

# Lookup a unit symbol with the symbol string, and provide a tuple with the
# conversion factor to cgs and dimensionality.

default_unit_symbol_lut = {
    # base
    "m": (1.0, dimensions.length, 0.0, r"\rm{m}", True),
    "g":  (1.0e-3, dimensions.mass, 0.0, r"\rm{g}", True),
    "s":  (1.0, dimensions.time, 0.0, r"\rm{s}", True),
    "K":  (1.0, dimensions.temperature, 0.0, r"\rm{K}", True),
    "radian": (1.0, dimensions.angle, 0.0, r"\rm{radian}", True),
    "A": (1.0, dimensions.current_mks, 0.0, r"\rm{A}", True),
    "cd": (1.0, dimensions.luminous_intensity, 0.0, r"\rm{cd}", True),
    "mol": (1.0 / amu_grams, dimensions.dimensionless, 0.0, r"\rm{mol}", True),

    # some cgs
    "dyne": (1.0e-5, dimensions.force, 0.0, r"\rm{dyn}", True),
    "erg":  (1.0e-7, dimensions.energy, 0.0, r"\rm{erg}", True),
    "Ba": (0.1, dimensions.pressure, 0.0, r"\rm{Ba}", True),
    "esu":  (1.0e-3**1.5, dimensions.charge_cgs, 0.0, r"\rm{esu}", True),
    "gauss": (0.1**0.5, dimensions.magnetic_field_cgs, 0.0, r"\rm{G}", True),
    "degC": (1.0, dimensions.temperature, -273.15, r"^\circ\rm{C}", True),
    "statA": (1.0e-3**1.5, dimensions.current_cgs, 0.0, r"\rm{statA}", True),
    "statV": (0.1*1.0e-3**0.5, dimensions.electric_potential_cgs,
              0.0, r"\rm{statV}", True),
    "statohm": (100.0, dimensions.resistance_cgs, 0.0, r"\rm{statohm}", True),
    "Mx": (1.0e-3**1.5, dimensions.magnetic_flux_cgs, 0.0, r"\rm{Mx}", True),

    # some SI
    "J": (1.0, dimensions.energy, 0.0, r"\rm{J}", True),
    "W": (1.0, dimensions.power, 0.0, r"\rm{W}", True),
    "Hz": (1.0, dimensions.rate, 0.0, r"\rm{Hz}", True),
    "N": (1.0, dimensions.force, 0.0, r"\rm{N}", True),
    "C": (1.0, dimensions.charge_mks, 0.0, r"\rm{C}", True),
    "T": (1.0, dimensions.magnetic_field_mks, 0.0, r"\rm{T}", True),
    "Pa": (1.0, dimensions.pressure, 0.0, r"\rm{Pa}", True),
    "V": (1.0, dimensions.electric_potential_mks, 0.0, r"\rm{V}", True),
    "ohm": (1.0, dimensions.resistance_mks, 0.0, r"\Omega", True),
    "Wb": (1.0, dimensions.magnetic_flux_mks, 0.0, r"\rm{Wb}", True),
    "lm": (1.0, dimensions.luminous_flux, 0.0, r"\rm{lm}", True),
    "lx": (1.0, dimensions.luminous_flux/dimensions.area, 0.0, r"\rm{lx}",
           True),

    # Imperial and other non-metric units
    "inch": (m_per_inch, dimensions.length, 0.0, r"\rm{in}", False),
    "ft": (m_per_ft, dimensions.length, 0.0, r"\rm{ft}", False),
    "yd": (0.9144, dimensions.length, 0.0, r"\rm{yd}", False),
    "mile": (1609.344, dimensions.length, 0.0, r"\rm{mile}", False),
    "fur": (m_per_ft*660.0, dimensions.length, 0.0, r"\rm{fur}", False),
    "degF": (kelvin_per_rankine, dimensions.temperature, -459.67,
             "^\circ\rm{F}", False),
    "R": (kelvin_per_rankine, dimensions.temperature, 0.0, r"^\circ\rm{R}",
          False),
    "lbf": (kg_per_pound*standard_gravity_m_per_s2,
            dimensions.force, 0.0, r"\rm{lbf}", False),
    "lb": (kg_per_pound, dimensions.mass, 0.0, r"\rm{lb}", False),
    "lbm": (kg_per_pound, dimensions.mass, 0.0, r"\rm{lbm}", False),
    "atm": (pascal_per_atm, dimensions.pressure, 0.0, r"\rm{atm}", False),
    "hp": (watt_per_horsepower, dimensions.power, 0.0, r"\rm{hp}", False),
    "oz": (kg_per_pound/16.0, dimensions.mass, 0.0, r"\rm{oz}", False),
    "ton": (kg_per_pound*2000.0, dimensions.mass, 0.0, r"\rm{ton}", False),
    "slug": (kg_per_pound*standard_gravity_m_per_s2/m_per_ft,
             dimensions.mass, 0.0, r"\rm{slug}", False),
    "cal": (4.184, dimensions.energy, 0.0, r"\rm{cal}", True),
    "BTU": (1055.0559, dimensions.energy, 0.0, r"\rm{BTU}", False),
    "psi": (kg_per_pound*standard_gravity_m_per_s2/m_per_inch**2,
            dimensions.pressure, 0.0, r"\rm{psi}", False),
    "smoot": (1.7018, dimensions.length, 0.0, r"\rm{smoot}", False),

    # dimensionless stuff
    "h": (1.0, dimensions.dimensionless, 0.0, r"h", False),
    "dimensionless": (1.0, dimensions.dimensionless, 0.0, r"", False),

    # times
    "min": (sec_per_min, dimensions.time, 0.0, r"\rm{min}", False),
    "hr":  (sec_per_hr, dimensions.time, 0.0, r"\rm{hr}", False),
    "day": (sec_per_day, dimensions.time, 0.0, r"\rm{d}", False),
    "d": (sec_per_day, dimensions.time, 0.0, r"\rm{d}", False),
    "yr":  (sec_per_year, dimensions.time, 0.0, r"\rm{yr}", True),

    # Velocities
    "c": (speed_of_light_m_per_s, dimensions.velocity, 0.0, r"\rm{c}", False),

    # Solar units
    "Msun": (mass_sun_kg, dimensions.mass, 0.0, r"M_\odot", False),
    "msun": (mass_sun_kg, dimensions.mass, 0.0, r"M_\odot", False),
    "Rsun": (m_per_rsun, dimensions.length, 0.0, r"R_\odot", False),
    "rsun": (m_per_rsun, dimensions.length, 0.0, r"R_\odot", False),
    "R_sun": (m_per_rsun, dimensions.length, 0.0, r"R_\odot", False),
    "r_sun": (m_per_rsun, dimensions.length, 0.0, r"R_\odot", False),
    "Lsun": (luminosity_sun_watts, dimensions.power, 0.0, r"L_\odot", False),
    "Tsun": (temp_sun_kelvin, dimensions.temperature, 0.0, r"T_\odot", False),
    "Zsun": (metallicity_sun, dimensions.dimensionless, 0.0, r"Z_\odot",
             False),
    "Mjup": (mass_jupiter_kg, dimensions.mass, 0.0, r"M_{\rm{Jup}}", False),
    "Mearth": (mass_earth_kg, dimensions.mass, 0.0, r"M_\oplus", False),
    "R_jup": (m_per_rjup, dimensions.length, 0.0, r"R_\mathrm{Jup}", False),
    "r_jup": (m_per_rjup, dimensions.length, 0.0, r"R_\mathrm{Jup}", False),
    "R_earth": (m_per_rearth, dimensions.length, 0.0, r"R_\oplus", False),
    "r_earth": (m_per_rearth, dimensions.length, 0.0, r"R_\oplus", False),

    # astro distances
    "AU": (m_per_au, dimensions.length, 0.0, r"\rm{AU}", False),
    "au": (m_per_au, dimensions.length, 0.0, r"\rm{AU}", False),
    "ly": (m_per_ly, dimensions.length, 0.0, r"\rm{ly}", False),
    "pc": (m_per_pc, dimensions.length, 0.0, r"\rm{pc}", True),

    # angles
    "degree": (np.pi/180., dimensions.angle, 0.0, r"\rm{deg}", False),
    "arcmin": (np.pi/10800., dimensions.angle, 0.0,
               r"\rm{arcmin}", False),  # arcminutes
    "arcsec": (np.pi/648000., dimensions.angle, 0.0,
               r"\rm{arcsec}", False),  # arcseconds
    "mas": (np.pi/648000000., dimensions.angle, 0.0,
            r"\rm{mas}", False),  # milliarcseconds
    "hourangle": (np.pi/12., dimensions.angle, 0.0, r"\rm{HA}", False),
    "steradian": (1.0, dimensions.solid_angle, 0.0, r"\rm{sr}", False),
    "lat": (-np.pi/180.0, dimensions.angle, 90.0, r"\rm{Latitude}", False),
    "lon": (np.pi/180.0, dimensions.angle, -180.0, r"\rm{Longitude}", False),

    # misc
    "eV": (J_per_eV, dimensions.energy, 0.0, r"\rm{eV}", True),
    "amu": (amu_kg, dimensions.mass, 0.0, r"\rm{amu}", False),
    "angstrom": (m_per_ang, dimensions.length, 0.0, r"\AA", False),
    "Jy": (jansky_mks, dimensions.specific_flux, 0.0, r"\rm{Jy}", True),
    "counts": (1.0, dimensions.dimensionless, 0.0, r"\rm{counts}", False),
    "photons": (1.0, dimensions.dimensionless, 0.0, r"\rm{photons}", False),
    "me": (mass_electron_kg, dimensions.mass, 0.0, r"m_e", False),
    "mp": (mass_hydrogen_kg, dimensions.mass, 0.0, r"m_p", False),
    'Sv': (1.0, dimensions.specific_energy, 0.0, r"\rm{Sv}", True),
    "rayleigh": (2.5e9/np.pi, dimensions.count_intensity, 0.0, r"\rm{R}",
                 False),
    "lambert": (1.0e4/np.pi, dimensions.luminance, 0.0, r"\rm{L}", False),
    "nt": (1.0, dimensions.luminance, 0.0, r"\rm{nt}", False),

    # for AstroPy compatibility
    "solMass": (mass_sun_kg, dimensions.mass, 0.0, r"M_\odot", False),
    "solRad": (m_per_rsun, dimensions.length, 0.0, r"R_\odot", False),
    "solLum": (luminosity_sun_watts, dimensions.power, 0.0, r"L_\odot", False),
    "dyn": (1.0e-5, dimensions.force, 0.0, r"\rm{dyn}", False),
    "sr": (1.0, dimensions.solid_angle, 0.0, r"\rm{sr}", False),
    "rad": (1.0, dimensions.angle, 0.0, r"\rm{rad}", False),
    "deg": (np.pi/180., dimensions.angle, 0.0, r"\rm{deg}", False),
    "Fr":  (1.0e-3**1.5, dimensions.charge_cgs, 0.0, r"\rm{Fr}", False),
    "G": (0.1**0.5, dimensions.magnetic_field_cgs, 0.0, r"\rm{G}", True),
    "Angstrom": (m_per_ang, dimensions.length, 0.0, r"\AA", False),
    "statC": (1.0e-3**1.5, dimensions.charge_cgs, 0.0, r"\rm{statC}", True),

    # Planck units
    "m_pl": (planck_mass_kg, dimensions.mass, 0.0, r"m_{\rm{P}}", False),
    "l_pl": (planck_length_m, dimensions.length, 0.0, r"\ell_\rm{P}", False),
    "t_pl": (planck_time_s, dimensions.time, 0.0, r"t_{\rm{P}}", False),
    "T_pl": (planck_temperature_K, dimensions.temperature, 0.0, r"T_{\rm{P}}",
             False),
    "q_pl": (planck_charge_C, dimensions.charge_mks, 0.0, r"q_{\rm{P}}",
             False),
    "E_pl": (planck_energy_J, dimensions.energy, 0.0, r"E_{\rm{P}}", False),

    # Geometrized units
    "m_geom": (mass_sun_kg, dimensions.mass, 0.0, r"M_\odot", False),
    "l_geom": (newton_mks*mass_sun_kg/speed_of_light_m_per_s**2,
               dimensions.length, 0.0, r"M_\odot", False),
    "t_geom": (newton_mks*mass_sun_kg/speed_of_light_m_per_s**3,
               dimensions.time, 0.0, r"M_\odot", False),
}

# This dictionary formatting from magnitude package, credit to Juan Reyero.
unit_prefixes = {
    'Y': 1e24,   # yotta
    'Z': 1e21,   # zetta
    'E': 1e18,   # exa
    'P': 1e15,   # peta
    'T': 1e12,   # tera
    'G': 1e9,    # giga
    'M': 1e6,    # mega
    'k': 1e3,    # kilo
    'h': 1e2,    # hecto
    'da': 1e1,   # deca
    'd': 1e-1,   # deci
    'c': 1e-2,   # centi
    'm': 1e-3,   # mili
    'u': 1e-6,   # micro
    'n': 1e-9,   # nano
    'p': 1e-12,  # pico
    'f': 1e-15,  # femto
    'a': 1e-18,  # atto
    'z': 1e-21,  # zepto
    'y': 1e-24,  # yocto
}

latex_prefixes = {
    "u": r"\mu",
    }

default_base_units = {
    dimensions.mass: 'kg',
    dimensions.length: 'm',
    dimensions.time: 's',
    dimensions.temperature: 'K',
    dimensions.angle: 'radian',
    dimensions.current_mks: 'A',
    dimensions.luminous_intensity: 'cd',
}
