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


from itertools import chain

from sympy import Symbol, sympify, Rational
from functools import wraps
from unyt.exceptions import UnitOperationError

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
base_dimensions = [
    mass,
    length,
    time,
    temperature,
    angle,
    current_mks,
    dimensionless,
    luminous_intensity,
]

#
# Derived dimensions
#

# rate
rate = 1 / time
# frequency (alias for rate)
frequency = rate

# spatial frequency
spatial_frequency = 1 / length

#: solid_angle
solid_angle = angle * angle
#: velocity
velocity = length / time
#: acceleration
acceleration = length / time ** 2
#: jerk
jerk = length / time ** 3
#: snap
snap = length / time ** 4
#: crackle
crackle = length / time ** 5
#: pop
pop = length / time ** 6

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
number_density = 1 / (length * length * length)
#: density
density = mass * number_density
#: angular_momentum
angular_momentum = mass * length * velocity
#: specific_angular_momentum
specific_angular_momentum = angular_momentum / mass
#: specific_energy
specific_energy = energy / mass
#: count_flux
count_flux = 1 / (area * time)
#: count_intensity
count_intensity = count_flux / solid_angle
#: luminous_flux
luminous_flux = luminous_intensity * solid_angle
#: luminance
luminance = luminous_intensity / area

# Gaussian electromagnetic units
#: charge_cgs
charge_cgs = (energy * length) ** Rational(1, 2)  # proper 1/2 power
#: current_cgs
current_cgs = charge_cgs / time
#: electric_field_cgs
electric_field_cgs = charge_cgs / length ** 2
#: magnetic_field_cgs
magnetic_field_cgs = electric_field_cgs
#: electric_potential_cgs
electric_potential_cgs = energy / charge_cgs
#: resistance_cgs
resistance_cgs = electric_potential_cgs / current_cgs
#: magnetic_flux_cgs
magnetic_flux_cgs = magnetic_field_cgs * area

# SI electromagnetic units
#: charge
charge = charge_mks = current_mks * time
#: electric_field
electric_field = electric_field_mks = force / charge_mks
#: magnetic_field
magnetic_field = magnetic_field_mks = electric_field_mks / velocity
#: electric_potential
electric_potential = electric_potential_mks = energy / charge_mks
#: resistance
resistance = resistance_mks = electric_potential_mks / current_mks
#: capacitance
capacitance = capacitance_mks = charge / electric_potential
#: magnetic_flux
magnetic_flux = magnetic_flux_mks = magnetic_field_mks * area

#: a list containing all derived_dimensions
derived_dimensions = [
    rate,
    velocity,
    acceleration,
    jerk,
    snap,
    crackle,
    pop,
    momentum,
    force,
    energy,
    power,
    charge_cgs,
    electric_field_cgs,
    magnetic_field_cgs,
    solid_angle,
    flux,
    specific_flux,
    volume,
    luminous_flux,
    area,
    current_cgs,
    charge_mks,
    electric_field_mks,
    magnetic_field_mks,
    electric_potential_cgs,
    electric_potential_mks,
    resistance_cgs,
    resistance_mks,
    magnetic_flux_mks,
    magnetic_flux_cgs,
    luminance,
    spatial_frequency,
]


#: a list containing all dimensions
dimensions = base_dimensions + derived_dimensions

#: a dict containing a bidirectional mapping from
#: mks dimension to cgs dimension
em_dimensions = {
    magnetic_field_mks: magnetic_field_cgs,
    magnetic_flux_mks: magnetic_flux_cgs,
    charge_mks: charge_cgs,
    current_mks: current_cgs,
    electric_potential_mks: electric_potential_cgs,
    resistance_mks: resistance_cgs,
}

for k, v in list(em_dimensions.items()):
    em_dimensions[v] = k


def check_dimensions(**arg_units):
    """Decorator for checking units of function arguments.

    Parameters
    ----------
    arg_units: dict
        Mapping of function arguments to dimensions, of the form {'arg1': dimension1, etc},
        where ``'arg1'`` etc are the function arguments and ``dimension1`` etc
        is an SI base unit (or combination of units), eg. length/time.

    Examples
    --------
    >>> import unyt as u
    >>> from unyt.dimensions import length, time
    >>> @check_dimensions(a=time, v=length/time)
    ... def f(a, v):
    ...     return a * v
    ...
    >>> res = f(a= 2 * u.s, v = 3 * u.m/u.s)
    >>> print(res)
    6 m
    """
    def check_nr_args(f):
        """Ensure correct number of arguments and decorate.

        Parameters
        ----------
        f : function
            Function being decorated.

        Returns
        -------
        new_f: function
            Decorated function.

        """
        number_of_args = f.__code__.co_argcount
        names_of_args = f.__code__.co_varnames

        assert (
            len(arg_units) == number_of_args
        ), f'decorator number of arguments not equal with function number of arguments in "{f.__name__}"'

        @wraps(f)
        def new_f(*args, **kwargs):
            """The new function being returned from the decorator.

            Check units of `args` and `kwargs`, then run original function.

            Raises
            ------
            UnitOperationError
                If the units do not match.

            """
            for arg_name, arg_value in chain(zip(names_of_args, args), kwargs.items()):
                dimension = arg_units[arg_name]
                if arg_name in arg_units and not _has_units(arg_value, dimension):
                    raise UnitOperationError(
                        f"arg '{arg_name}'={repr(arg_value)} does not match {dimension}"
                    )
            return f(*args, **kwargs)
        return new_f
    return check_nr_args


def _has_units(quant, dim):
    """Checks the argument has the right dimensionality.

    Parameters
    ----------
    quant : :py:class:`unyt.array.unyt_quantity`
        Quantity whose dimensionality we want to check.
    dim : :py:class:`sympy.core.symbol.Symbol`
        SI base unit (or combination of units), eg. length/time

    Returns
    -------
    bool
        True if check successful.

    Examples
    --------
    >>> import unyt as u
    >>> from unyt.dimensions import length, time
    >>> _has_units(3 * u.m/u.s, length/time)
    True
    """
    try:
        arg_dim = quant.units.dimensions
    except AttributeError:
        arg_dim = None
    return arg_dim == dim
