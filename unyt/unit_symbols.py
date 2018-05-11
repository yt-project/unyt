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

from unyt.unit_object import Unit

#
# meter
#

#: femtometer
fm = femtometer = Unit("fm")
#: picometer
pm = picometer = Unit("pm")
#: nanometer
nm = nanometer = Unit("nm")
#: micrometer
um = micrometer = Unit("um")
#: millimeter
mm = millimeter = Unit("mm")
#: centimeter
cm = centimeter = Unit("cm")
#: meter
m = meter = Unit("m")
#: kilometer
km = kilometer = Unit("km")
#: Megameter
Mm = Megameter = megameter = Unit("Mm")

#
# parsec
#

#: parsec
pc = parsec = Unit("pc")
#: kiloparsec
kpc = kiloparsec = Unit("kpc")
#: megaparsec
Mpc = mpc = megaparsec = Unit("Mpc")
#: gigaparsec
Gpc = gpc = Gigaparsec = Unit("Gpc")

#
# gram
#

#: picogram
pg = picogram = Unit("pg")
#: nanogram
ng = nanogram = Unit("ng")
#: micogram
ug = microgram = Unit("ug")
#: milligram
mg = milligram = Unit("mg")
#: gram
g = gram = Unit("g")
#: kilogram
kg = kilogram = Unit("kg")
#: megagram
Mg = megagramme = tonne = metric_ton = Unit("kg")

#
# second
#

#: femtosecond
fs = femtoseconds = Unit("fs")
#: picosecond
ps = picosecond = Unit("ps")
#: nanosecond
ns = nanosecond = Unit("ns")
#: millisecond
ms = millisecond = Unit("ms")
#: second
s = second = Unit("s")
#: kilosecond
ks = kilosecond = Unit("ks")
#: megasecond
Ms = megasecond = Unit("Ms")
#: gigasecond
Gs = gigasecond = Unit("Gs")

#
# minute
#

#: minute
min = minute = Unit("min")

#
# hr
#

#: hour
hr = hour = Unit("hr")

#
# day
#

#: day
day = Unit("day")

#
# year
#

#: year
yr = year = Unit("yr")
#: kiloyear
kyr = kiloyear = Unit("kyr")
#: Megayear
Myr = Megayear = megayear = Unit("Myr")
#: Gigayear
Gyr = Gigayear = gigayear = Unit("Gyr")

#
# Temperatures
#

#: Degree kelvin
degree_kelvin = Kelvin = K = Unit("K")
#: Degree fahrenheit
degree_fahrenheit = degF = Unit("degF")
#: Degree Celsius
degree_celsius = degC = Unit("degC")
#:
degree_rankine = R = Unit("R")

#
# Misc CGS
#

#: dyne (CGS force)
dyne = dyn = Unit("dyne")
#: erg (CGS energy)
erg = ergs = Unit("erg")
#:
barye = Ba = Unit("Ba")

#
# Misc SI
#

#: Newton (SI force)
N = Newton = newton = Unit("N")
#: Joule (SI energy)
J = Joule = joule = Unit("J")
#: Watt (SI power)
W = Watt = watt = Unit("W")
#: Hertz
Hz = Hertz = hertz = Unit("Hz")
#: Kilohertz
kHz = khz = kilohertz = Unit("kHz")
#: Megahertz
MHz = mhz = megahertz = Unit("MHz")
#: Gigahertz
GHz = ghz = gigahertz = Unit("GHz")
#: Terahertz
THz = thz = terahertz = Unit("THz")
#: Pascal
Pa = pascal = Unit("Pa")
#: sievert
Sv = sievert = Unit("Sv")
#: candela
cd = Candela = candela = Unit("cd")
#: lumen
lm = Lumen = lumen = Unit("lm")
#: lux
lx = Lux = lux = Unit('lx')
#: lambert
lambert = Lambert = Unit('lambert')
#: nit
nt = nit = Nit = Unit("nt")

#
# Imperial units
#

#: inch
inch = Unit("inch")
#: foot
ft = foot = Unit("ft")
#: mile
mile = Unit("mile")
#: furlong
fur = furlong = Unit("fur")
#: yard
yard = yd = Unit("yd")
#: pound
lb = pound = Unit("lb")
#: slug
slug = Unit("slug")
#: ounce
oz = ounce = Unit("oz")
#: ton
ton = Unit("ton")
#: pound-force
lbf = pound_force = Unit("lbf")
#: atmosphere
atm = atmosphere = Unit("atm")
#: horsepower
hp = horsepower = Unit("hp")
#: BTU
BTU = Unit("BTU")
#: calorie
cal = calorie = Unit("cal")
#: psi
psi = Unit("psi")

#
# Solar units
#

#: Mass of the sun
Msun = Unit("Msun")
#: Radius of the sun
Rsun = R_sun = solar_radius = Unit("Rsun")
#: Radius of the sun
rsun = r_sun = Unit("rsun")
#: Luminosity of the sun
Lsun = lsun = l_sun = solar_luminosity = Unit("Lsun")
#: Temperature of the sun
Tsun = T_sun = solar_temperature = Unit("Tsun")
#: Metallicity of the sun
Zsun = Z_sun = solar_metallicity = Unit("Zsun")

#
# Misc Astronomical units
#

#: Astronomical unit
AU = astronomical_unit = Unit("AU")
#: Astronomical unit
au = Unit("au")
#: Light year
ly = light_year = Unit("ly")
#: Radius of the Earth
Rearth = R_earth = earth_radius = Unit('R_earth')
#: Radius of the Earth
rearth = r_earth = Unit('r_earth')
#: Radius of Jupiter
Rjup = R_jup = jupiter_radius = Unit('R_jup')
#: Radius of Jupiter
rjup = r_jup = Unit('r_jup')
#: Jansky
Jy = jansky = Unit("Jy")

#
# Physical units
#

#: electronvolt
eV = electron_volt = electronvolt = Unit("eV")
#: kiloelectronvolt
keV = kilo_electron_volt = kiloelectronvolt = Unit("keV")
#: Megaelectronvolt
MeV = mega_electron_volt = Megaelectronvolt = megaelectronvolt = Unit("MeV")
#: Gigaelectronvolt
GeV = giga_electron_volt = Gigaelectronvolt = gigaelectronvolt = Unit("GeV")
#: Atomic mass unit
amu = atomic_mass_unit = Unit("amu")
mol = Unit("mol")
#: Angstrom
angstrom = Unit("angstrom")
#: Electron mass
me = electron_mass = Unit("me")

#
# Angle units
#

#: Degree (angle)
deg = degree = Unit("degree")
#: Radian
rad = radian = Unit("radian")
#: Arcsecond
arcsec = arcsecond = Unit("arcsec")
#: Arcminute
arcmin = arcminute = Unit("arcmin")
#: milliarcsecond
mas = milliarcsecond = Unit("mas")
#: steradian
sr = steradian = Unit("steradian")
#: hourangle
HA = hourangle = Unit("hourangle")


#
# CGS electromagnetic units
#

#: electrostatic unit
electrostatic_unit = esu = Unit("esu")
#: Gauss
gauss = G = Unit("gauss")
#: Statcoulomb
statcoulomb = statC = Unit("statC")
#: Statampere
statampere = statA = Unit("statA")
#: Statvolt
statvolt = statV = Unit("statV")
#: Statohm
statohm = Unit("statohm")
#: Maxwell
Mx = maxwell = Maxwell = Unit("Mx")

#
# SI electromagnetic units
#

#: Coulomb
C = coulomb = Coulomb = Unit("C")
#: Tesla
T = tesla = Tesla = Unit("T")
#: Ampere
A = ampere = Ampere = Unit("A")
#: Volt
V = volt = Volt = Unit("V")
#: Ohm
ohm = Ohm = Unit("ohm")
#: Weber
Wb = weber = Weber = Unit("Wb")

#
# Geographic units
#

#: Degree latitude
latitude = lat = Unit("lat")
#: Degree longitude
longitude = lon = Unit("lon")

#
# Misc dimensionless units
#

#: Count of something
counts = count = Unit('counts')
#: Number of photons
photons = photon = Unit('photons')
#: dimensionless
_ = dimensionless = Unit('')
