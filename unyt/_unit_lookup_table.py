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
    "m": (1.0, dimensions.length, 0.0, r"\rm{m}"),
    "g":  (1.0e-3, dimensions.mass, 0.0, r"\rm{g}"),
    "s":  (1.0, dimensions.time, 0.0, r"\rm{s}"),
    "K":  (1.0, dimensions.temperature, 0.0, r"\rm{K}"),
    "radian": (1.0, dimensions.angle, 0.0, r"\rm{radian}"),
    "A": (1.0, dimensions.current_mks, 0.0, r"\rm{A}"),
    "cd": (1.0, dimensions.luminous_intensity, 0.0, r"\rm{cd}"),
    "mol": (1.0 / amu_grams, dimensions.dimensionless, 0.0, r"\rm{mol}"),

    # some cgs
    "dyne": (1.0e-5, dimensions.force, 0.0, r"\rm{dyn}"),
    "erg":  (1.0e-7, dimensions.energy, 0.0, r"\rm{erg}"),
    "Ba": (0.1, dimensions.pressure, 0.0, r"\rm{Ba}"),
    "esu":  (1.0e-3**1.5, dimensions.charge_cgs, 0.0, r"\rm{esu}"),
    "gauss": (0.1**0.5, dimensions.magnetic_field_cgs, 0.0, r"\rm{G}"),
    "degC": (1.0, dimensions.temperature, -273.15, r"^\circ\rm{C}"),
    "statA": (1.0e-3**1.5, dimensions.current_cgs, 0.0, r"\rm{statA}"),
    "statV": (0.1*1.0e-3**0.5, dimensions.electric_potential_cgs,
              0.0, r"\rm{statV}"),
    "statohm": (100.0, dimensions.resistance_cgs, 0.0, r"\rm{statohm}"),
    "Mx": (1.0e-3**1.5, dimensions.magnetic_flux_cgs, 0.0, r"\rm{Mx}"),

    # some SI
    "J": (1.0, dimensions.energy, 0.0, r"\rm{J}"),
    "W": (1.0, dimensions.power, 0.0, r"\rm{W}"),
    "Hz": (1.0, dimensions.rate, 0.0, r"\rm{Hz}"),
    "N": (1.0, dimensions.force, 0.0, r"\rm{N}"),
    "C": (1.0, dimensions.charge_mks, 0.0, r"\rm{C}"),
    "T": (1.0, dimensions.magnetic_field_mks, 0.0, r"\rm{T}"),
    "Pa": (1.0, dimensions.pressure, 0.0, r"\rm{Pa}"),
    "V": (1.0, dimensions.electric_potential_mks, 0.0, r"\rm{V}"),
    "ohm": (1.0, dimensions.resistance_mks, 0.0, r"\Omega"),
    "Wb": (1.0, dimensions.magnetic_flux_mks, 0.0, r"\rm{Wb}"),
    "lm": (1.0, dimensions.luminous_flux, 0.0, r"\rm{lm}"),
    "lx": (1.0, dimensions.luminous_flux/dimensions.area, 0.0, r"\rm{lx}"),

    # Imperial and other non-metric units
    "inch": (m_per_inch, dimensions.length, 0.0, r"\rm{in}"),
    "ft": (m_per_ft, dimensions.length, 0.0, r"\rm{ft}"),
    "yd": (0.9144, dimensions.length, 0.0, r"\rm{yd}"),
    "mile": (1609.344, dimensions.length, 0.0, r"\rm{mile}"),
    "fur": (m_per_ft*660.0, dimensions.length, 0.0, r"\rm{fur}"),
    "degF": (kelvin_per_rankine, dimensions.temperature, -459.67,
             "^\circ\rm{F}"),
    "R": (kelvin_per_rankine, dimensions.temperature, 0.0, r"^\circ\rm{R}"),
    "lbf": (kg_per_pound*standard_gravity_m_per_s2,
            dimensions.force, 0.0, r"\rm{lbf}"),
    "lb": (kg_per_pound, dimensions.mass, 0.0, r"\rm{lb}"),
    "lbm": (kg_per_pound, dimensions.mass, 0.0, r"\rm{lbm}"),
    "atm": (pascal_per_atm, dimensions.pressure, 0.0, r"\rm{atm}"),
    "hp": (watt_per_horsepower, dimensions.power, 0.0, r"\rm{hp}"),
    "oz": (kg_per_pound/16.0, dimensions.mass, 0.0, r"\rm{oz}"),
    "ton": (kg_per_pound*2000.0, dimensions.mass, 0.0, r"\rm{ton}"),
    "slug": (kg_per_pound*standard_gravity_m_per_s2/m_per_ft,
             dimensions.mass, 0.0, r"\rm{slug}"),
    "cal": (4.184, dimensions.energy, 0.0, r"\rm{cal}"),
    "BTU": (1055.0559, dimensions.energy, 0.0, r"\rm{BTU}"),
    "psi": (kg_per_pound*standard_gravity_m_per_s2/m_per_inch**2,
            dimensions.pressure, 0.0, r"\rm{psi}"),

    # dimensionless stuff
    "h": (1.0, dimensions.dimensionless, 0.0, r"h"),
    "dimensionless": (1.0, dimensions.dimensionless, 0.0, r""),

    # times
    "min": (sec_per_min, dimensions.time, 0.0, r"\rm{min}"),
    "hr":  (sec_per_hr, dimensions.time, 0.0, r"\rm{hr}"),
    "day": (sec_per_day, dimensions.time, 0.0, r"\rm{d}"),
    "d": (sec_per_day, dimensions.time, 0.0, r"\rm{d}"),
    "yr":  (sec_per_year, dimensions.time, 0.0, r"\rm{yr}"),

    # Velocities
    "c": (speed_of_light_m_per_s, dimensions.velocity, 0.0, r"\rm{c}"),

    # Solar units
    "Msun": (mass_sun_kg, dimensions.mass, 0.0, r"M_\odot"),
    "msun": (mass_sun_kg, dimensions.mass, 0.0, r"M_\odot"),
    "Rsun": (m_per_rsun, dimensions.length, 0.0, r"R_\odot"),
    "rsun": (m_per_rsun, dimensions.length, 0.0, r"R_\odot"),
    "R_sun": (m_per_rsun, dimensions.length, 0.0, r"R_\odot"),
    "r_sun": (m_per_rsun, dimensions.length, 0.0, r"R_\odot"),
    "Lsun": (luminosity_sun_watts, dimensions.power, 0.0, r"L_\odot"),
    "Tsun": (temp_sun_kelvin, dimensions.temperature, 0.0, r"T_\odot"),
    "Zsun": (metallicity_sun, dimensions.dimensionless, 0.0, r"Z_\odot"),
    "Mjup": (mass_jupiter_kg, dimensions.mass, 0.0, r"M_{\rm{Jup}}"),
    "Mearth": (mass_earth_kg, dimensions.mass, 0.0, r"M_\oplus"),

    # astro distances
    "AU": (m_per_au, dimensions.length, 0.0, r"\rm{AU}"),
    "au": (m_per_au, dimensions.length, 0.0, r"\rm{AU}"),
    "ly": (m_per_ly, dimensions.length, 0.0, r"\rm{ly}"),
    "pc": (m_per_pc, dimensions.length, 0.0, r"\rm{pc}"),

    # angles
    "degree": (np.pi/180., dimensions.angle, 0.0, r"\rm{deg}"),  # degrees
    "arcmin": (np.pi/10800., dimensions.angle, 0.0,
               r"\rm{arcmin}"),  # arcminutes
    "arcsec": (np.pi/648000., dimensions.angle, 0.0,
               r"\rm{arcsec}"),  # arcseconds
    "mas": (np.pi/648000000., dimensions.angle, 0.0,
            r"\rm{mas}"),  # milliarcseconds
    "hourangle": (np.pi/12., dimensions.angle, 0.0, r"\rm{HA}"),  # hour angle
    "steradian": (1.0, dimensions.solid_angle, 0.0, r"\rm{sr}"),
    "lat": (-np.pi/180.0, dimensions.angle, 90.0, r"\rm{Latitude}"),
    "lon": (np.pi/180.0, dimensions.angle, -180.0, r"\rm{Longitude}"),

    # misc
    "eV": (J_per_eV, dimensions.energy, 0.0, r"\rm{eV}"),
    "amu": (amu_kg, dimensions.mass, 0.0, r"\rm{amu}"),
    "angstrom": (m_per_ang, dimensions.length, 0.0, r"\AA"),
    "Jy": (jansky_mks, dimensions.specific_flux, 0.0, r"\rm{Jy}"),
    "counts": (1.0, dimensions.dimensionless, 0.0, r"\rm{counts}"),
    "photons": (1.0, dimensions.dimensionless, 0.0, r"\rm{photons}"),
    "me": (mass_electron_kg, dimensions.mass, 0.0, r"m_e"),
    "mp": (mass_hydrogen_kg, dimensions.mass, 0.0, r"m_p"),
    'Sv': (1.0, dimensions.specific_energy, 0.0, r"\rm{Sv}"),
    "rayleigh": (2.5e9/np.pi, dimensions.count_intensity, 0.0, r"\rm{R}"),
    "lambert": (1.0e4/np.pi, dimensions.luminance, 0.0, r"\rm{L}"),
    "nt": (1.0, dimensions.luminance, 0.0, r"\rm{nt}"),

    # for AstroPy compatibility
    "solMass": (mass_sun_kg, dimensions.mass, 0.0, r"M_\odot"),
    "solRad": (m_per_rsun, dimensions.length, 0.0, r"R_\odot"),
    "solLum": (luminosity_sun_watts, dimensions.power, 0.0, r"L_\odot"),
    "dyn": (1.0e-5, dimensions.force, 0.0, r"\rm{dyn}"),
    "sr": (1.0, dimensions.solid_angle, 0.0, r"\rm{sr}"),
    "rad": (1.0, dimensions.angle, 0.0, r"\rm{rad}"),
    "deg": (np.pi/180., dimensions.angle, 0.0, r"\rm{deg}"),
    "Fr":  (1.0e-3**1.5, dimensions.charge_cgs, 0.0, r"\rm{Fr}"),
    "G": (0.1**0.5, dimensions.magnetic_field_cgs, 0.0, r"\rm{G}"),
    "Angstrom": (m_per_ang, dimensions.length, 0.0, r"\AA"),
    "statC": (1.0e-3**1.5, dimensions.charge_cgs, 0.0, r"\rm{statC}"),

    # Planck units
    "m_pl": (planck_mass_kg, dimensions.mass, 0.0, r"m_{\rm{P}}"),
    "l_pl": (planck_length_m, dimensions.length, 0.0, r"\ell_\rm{P}"),
    "t_pl": (planck_time_s, dimensions.time, 0.0, r"t_{\rm{P}}"),
    "T_pl": (planck_temperature_K, dimensions.temperature, 0.0, r"T_{\rm{P}}"),
    "q_pl": (planck_charge_C, dimensions.charge_mks, 0.0, r"q_{\rm{P}}"),
    "E_pl": (planck_energy_J, dimensions.energy, 0.0, r"E_{\rm{P}}"),

    # Geometrized units
    "m_geom": (mass_sun_kg, dimensions.mass, 0.0, r"M_\odot"),
    "l_geom": (newton_mks*mass_sun_kg/speed_of_light_m_per_s**2,
               dimensions.length, 0.0, r"M_\odot"),
    "t_geom": (newton_mks*mass_sun_kg/speed_of_light_m_per_s**3,
               dimensions.time, 0.0, r"M_\odot"),

    # Some Solar System units
    "R_earth": (m_per_rearth, dimensions.length, 0.0, r"R_\oplus"),
    "r_earth": (m_per_rearth, dimensions.length, 0.0, r"R_\oplus"),
    "R_jup": (m_per_rjup, dimensions.length, 0.0, r"R_\mathrm{Jup}"),
    "r_jup": (m_per_rjup, dimensions.length, 0.0, r"R_\mathrm{Jup}"),
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

prefixable_units = [
    "m",
    "pc",
    "mcm",
    "pccm",
    "g",
    "eV",
    "s",
    "yr",
    "K",
    "dyne",
    "erg",
    "esu",
    "J",
    "Hz",
    "W",
    "gauss",
    "G",
    "Jy",
    "N",
    "T",
    "A",
    "C",
    "statA",
    "Pa",
    "V",
    "statV",
    "ohm",
    "statohm",
    "Sv",
    "mol",
    "cd",
    "lm",
    "lx",
    "Wb",
    "Mx",
]

default_base_units = {
    dimensions.mass: 'kg',
    dimensions.length: 'm',
    dimensions.time: 's',
    dimensions.temperature: 'K',
    dimensions.angle: 'radian',
    dimensions.current_mks: 'A',
    dimensions.luminous_intensity: 'cd',
}
