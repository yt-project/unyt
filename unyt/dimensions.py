"""
Dimensions of physical quantities


"""

# -----------------------------------------------------------------------------
# Copyright (c) 2018, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------


from sympy import Symbol, sympify, Rational

#: mass
mass = Symbol("(mass)", positive=True)
#: length
length = Symbol("(length)", positive=True)
#: time
time = Symbol("(time)", positive=True)
#: temperature
temperature = Symbol("(temperature)", positive=True)
#: angle
angle = Symbol("(angle)", positive=True)
#: current_mks
current_mks = Symbol("(current_mks)", positive=True)
#: luminous_intensity
luminous_intensity = Symbol("(luminous_intensity)", positive=True)
#: dimensionless
dimensionless = sympify(1)

#: A list of all of the base dimensions
base_dimensions = [mass, length, time, temperature, angle, current_mks,
                   dimensionless, luminous_intensity]

#
# Derived dimensions
#

# rate
rate = 1 / time
# frequency (alias for rate)
frequency = rate

#: solid_angle
solid_angle = angle * angle
#: velocity
velocity = length / time
#: acceleration
acceleration = length / time**2
#: jerk
jerk = length / time**3
#: snap
snap = length / time**4
#: crackle
crackle = length / time**5
#: pop
pop = length / time**6

#: area
area = length * length
#: volume
volume = area * length
#: momentum
momentum = mass * velocity
#: force
force = mass * acceleration
#: pressure
pressure = force / area
#: energy
energy = force * length
#: power
power = energy / time
#: flux
flux = power / area
#: specific_flux
specific_flux = flux / rate
#: number_density
number_density = 1/(length*length*length)
#: density
density = mass * number_density
#: angular_momentum
angular_momentum = mass*length*velocity
#: specific_angular_momentum
specific_angular_momentum = angular_momentum / mass
#: specific_energy
specific_energy = energy / mass
#: count_flux
count_flux = 1 / (area*time)
#: count_intensity
count_intensity = count_flux / solid_angle
#: luminous_flux
luminous_flux = luminous_intensity * solid_angle
#: luminance
luminance = luminous_intensity / area

# Gaussian electromagnetic units
#: charge_cgs
charge_cgs = (energy * length)**Rational(1, 2)  # proper 1/2 power
#: current_cgs
current_cgs = charge_cgs / time
#: electric_field_cgs
electric_field_cgs = charge_cgs / length**2
#: magnetic_field_cgs
magnetic_field_cgs = electric_field_cgs
#: electric_potential_cgs
electric_potential_cgs = energy / charge_cgs
#: resistance_cgs
resistance_cgs = electric_potential_cgs / current_cgs
#: magnetic_flux_cgs
magnetic_flux_cgs = magnetic_field_cgs * area

# SI electromagnetic units
#: charge_mks
charge = charge_mks = current_mks * time
#: electric_field_mks
electric_field = electric_field_mks = force / charge_mks
#: magnetic_field_mks
magnetic_field = magnetic_field_mks = electric_field_mks / velocity
#: electric_potential_mks
electric_potential = electric_potential_mks = energy / charge_mks
#: resistance_mks
resistance = resistance_mks = electric_potential_mks / current_mks
#: magnetic_flux_mks
magnetic_flux = magnetic_flux_mks = magnetic_field_mks * area

#: a list containing all derived_dimensions
derived_dimensions = [
    rate, velocity, acceleration, jerk, snap, crackle, pop,
    momentum, force, energy, power, charge_cgs, electric_field_cgs,
    magnetic_field_cgs, solid_angle, flux, specific_flux, volume,
    luminous_flux, area, current_cgs, charge_mks, electric_field_mks,
    magnetic_field_mks, electric_potential_cgs, electric_potential_mks,
    resistance_cgs, resistance_mks, magnetic_flux_mks, magnetic_flux_cgs,
    luminance]


#: a list containing all dimensions
dimensions = base_dimensions + derived_dimensions

#: a dict containing a bidirectional mapping from
#: mks dimension to cgs dimension
em_dimensions = {magnetic_field_mks: magnetic_field_cgs,
                 magnetic_flux_mks: magnetic_flux_cgs,
                 charge_mks: charge_cgs,
                 current_mks: current_cgs,
                 electric_potential_mks: electric_potential_cgs,
                 resistance_mks: resistance_cgs}

for k, v in list(em_dimensions.items()):
    em_dimensions[v] = k
