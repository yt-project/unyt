"""
Predefined useful aliases to physical units

Note that all of these names can be imported from the top-level unyt namespace.
For example::

    >>> from unyt import cm, g, s
    >>> data = [3, 4, 5]*g*cm/s
    >>> print(data)
    [3. 4. 5.] cm*g/s

"""

# -----------------------------------------------------------------------------
# Copyright (c) 2018, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------

from unyt.array import unyt_quantity as quan

#
# meter
#

#: femtometer
fm = femtometer = quan(1.0, "fm")
#: picometer
pm = picometer = quan(1.0, "pm")
#: nanometer
nm = nanometer = quan(1.0, "nm")
#: micrometer
um = micrometer = quan(1.0, "um")
#: millimeter
mm = millimeter = quan(1.0, "mm")
#: centimeter
cm = centimeter = quan(1.0, "cm")
#: meter
m = meter = quan(1.0, "m")
#: kilometer
km = kilometer = quan(1.0, "km")
#: Megameter
Mm = Megameter = megameter = quan(1.0, "Mm")

#
# parsec
#

#: parsec
pc = parsec = quan(1.0, "pc")
#: kiloparsec
kpc = kiloparsec = quan(1.0, "kpc")
#: megaparsec
Mpc = mpc = megaparsec = quan(1.0, "Mpc")
#: gigaparsec
Gpc = gpc = Gigaparsec = quan(1.0, "Gpc")

#
# gram
#

#: picogram
pg = picogram = quan(1.0, "pg")
#: nanogram
ng = nanogram = quan(1.0, "ng")
#: micogram
ug = microgram = quan(1.0, "ug")
#: milligram
mg = milligram = quan(1.0, "mg")
#: gram
g = gram = quan(1.0, "g")
#: kilogram
kg = kilogram = quan(1.0, "kg")
#: megagram
Mg = megagramme = tonne = metric_ton = quan(1.0, "kg")

#
# second
#

#: femtosecond
fs = femtoseconds = quan(1.0, "fs")
#: picosecond
ps = picosecond = quan(1.0, "ps")
#: nanosecond
ns = nanosecond = quan(1.0, "ns")
#: millisecond
ms = millisecond = quan(1.0, "ms")
#: second
s = second = quan(1.0, "s")
#: kilosecond
ks = kilosecond = quan(1.0, "ks")
#: megasecond
Ms = megasecond = quan(1.0, "Ms")
#: gigasecond
Gs = gigasecond = quan(1.0, "Gs")

#
# minute
#

#: minute
min = minute = quan(1.0, "min")

#
# hr
#

#: hour
hr = hour = quan(1.0, "hr")

#
# day
#

#: day
day = quan(1.0, "day")

#
# year
#

#: year
yr = year = quan(1.0, "yr")
#: kiloyear
kyr = kiloyear = quan(1.0, "kyr")
#: Megayear
Myr = Megayear = megayear = quan(1.0, "Myr")
#: Gigayear
Gyr = Gigayear = gigayear = quan(1.0, "Gyr")

#
# Temperatures
#

#: Degree kelvin
degree_kelvin = Kelvin = K = quan(1.0, "K")
#: Degree fahrenheit
degree_fahrenheit = degF = quan(1.0, "degF")
#: Degree Celsius
degree_celsius = degC = quan(1.0, "degC")
#:
degree_rankine = R = quan(1.0, "R")

#
# Misc CGS
#

#: dyne (CGS force)
dyne = dyn = quan(1.0, "dyne")
#: erg (CGS energy)
erg = ergs = quan(1.0, "erg")
#:
barye = Ba = quan(1.0, "Ba")

#
# Misc SI
#

#: Newton (SI force)
N = Newton = newton = quan(1.0, "N")
#: Joule (SI energy)
J = Joule = joule = quan(1.0, "J")
#: Watt (SI power)
W = Watt = watt = quan(1.0, "W")
#: Hertz
Hz = Hertz = hertz = quan(1.0, "Hz")
#: Pascal
Pa = pascal = quan(1.0, "Pa")
#: sievert
Sv = sievert = quan(1.0, "Sv")

#
# Imperial units
#

#: foot
ft = foot = quan(1.0, "ft")
#: mile
mile = quan(1.0, "mile")
#: furlong
furlong = quan(660, "ft")
#: yard
yard = quan(3, "ft")
#: pound
lb = pound = quan(1.0, "lb")
#: pound-foot
lfb = pound_foot = quan(1.0, "lbf")
#: atmosphere
atm = atmosphere = quan(1.0, "atm")

#
# Solar units
#

#: Mass of the sun
Msun = quan(1.0, "Msun")
#: Radius of the sun
Rsun = R_sun = solar_radius = quan(1.0, "Rsun")
#: Radius of the sun
rsun = r_sun = quan(1.0, "rsun")
#: Luminosity of the sun
Lsun = lsun = l_sun = solar_luminosity = quan(1.0, "Lsun")
#: Temperature of the sun
Tsun = T_sun = solar_temperature = quan(1.0, "Tsun")
#: Metallicity of the sun
Zsun = Z_sun = solar_metallicity = quan(1.0, "Zsun")

#
# Misc Astronomical units
#

#: Astronomical unit
AU = astronomical_unit = quan(1.0, "AU")
#: Astronomical unit
au = quan(1.0, "au")
#: Light year
ly = light_year = quan(1.0, "ly")
#: Radius of the Earth
Rearth = R_earth = earth_radius = quan(1.0, 'R_earth')
#: Radius of the Earth
rearth = r_earth = quan(1.0, 'r_earth')
#: Radius of Jupiter
Rjup = R_jup = jupiter_radius = quan(1.0, 'R_jup')
#: Radius of Jupiter
rjup = r_jup = quan(1.0, 'r_jup')
#: Jansky
Jy = jansky = quan(1.0, "Jy")

#
# Physical units
#

#: electronvolt
eV = electron_volt = electronvolt = quan(1.0, "eV")
#: kiloelectronvolt
keV = kilo_electron_volt = kiloelectronvolt = quan(1.0, "keV")
#: Megaelectronvolt
MeV = mega_electron_volt = Megaelectronvolt = megaelectronvolt = quan(
    1.0, "MeV")
#: Gigaelectronvolt
GeV = giga_electron_volt = Gigaelectronvolt = gigaelectronvolt = quan(
    1.0, "GeV")
#: Atomic mass unit
amu = atomic_mass_unit = quan(1.0, "amu")
mol = quan(1.0, "mol")
#: Angstrom
angstrom = quan(1.0, "angstrom")
#: Electron mass
me = electron_mass = quan(1.0, "me")

#
# Angle units
#

#: Degree (angle)
deg = degree = quan(1.0, "degree")
#: Radian
rad = radian = quan(1.0, "radian")
#: Arcsecond
arcsec = arcsecond = quan(1.0, "arcsec")
#: Arcminute
arcmin = arcminute = quan(1.0, "arcmin")
#: milliarcsecond
mas = milliarcsecond = quan(1.0, "mas")
#: steradian
sr = steradian = quan(1.0, "steradian")
#: hourangle
HA = hourangle = quan(1.0, "hourangle")


#
# CGS electromagnetic units
#

#: electrostatic unit
electrostatic_unit = esu = quan(1.0, "esu")
#: Gauss
gauss = G = quan(1.0, "gauss")
#: Statampere
statampere = statA = quan(1.0, "statA")
#: Statvolt
statvolt = statV = quan(1.0, "statV")
#: Statohm
statohm = quan(1.0, "statohm")

#
# SI electromagnetic units
#

#: Coulomb
C = coulomb = Coulomb = quan(1.0, "C")
#: Tesla
T = tesla = Tesla = quan(1.0, "T")
#: Ampere
A = ampere = Ampere = quan(1.0, "A")
#: Volt
V = volt = Volt = quan(1.0, "V")
#: Ohm
ohm = Ohm = quan(1.0, "ohm")

#
# Geographic units
#

#: Degree latitude
latitude = lat = quan(1.0, "lat")
#: Degree longitude
longitude = lon = quan(1.0, "lon")

#
# Misc dimensionless units
#

#: Count of something
counts = count = quan(1.0, 'counts')
#: Number of photons
photons = photon = quan(1.0, 'photons')
