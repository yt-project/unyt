=====
Usage
=====

To use unyt in a project::

  >>> import unyt

The top-level :mod:`unyt` namespace defines both a number of useful functions as
well as a number of unit symbols you can use to attach units to numpy arrays or
python lists.

A Basic Example from High School Physics
----------------------------------------

To see how you might use these symbols to solve a problem where units might be a
headache, let's estimate the orbital periods of Jupiter's Galilean moons,
assuming they have circular orbits and their masses are negligible compared to
Jupiter. Under these assumptions, the orbital period is

.. math::

   T = 2\pi\left( \frac{r^3}{GM}\right)^{1/2}.

For this exercise let's calculate the orbital period in days. While it's
possible to do this using plain old floating point numbers (you probably had to
do something similar on a calculator in a high school physics class, looking up
and plugging in conversion factors by hand), it's much easier to do this sort of
thing symbolically and let ``unyt`` handle the unit conversions.

To do this we'll need to know the mass of jupiter (fortunately that is built
into ``unyt``) and the semimajor axis of the orbits of Jupiter's moons, which we
can look up from `Wikipedia
<https://en.wikipedia.org/wiki/Moons_of_Jupiter#List>`_ and enter by hand::

  >>> from unyt import Mjup, G, km
  >>> from math import pi
  ...
  >>> moons = ['Io', 'Europa', 'Ganymede', 'Callisto']
  >>> semimajor_axis = [421700, 671034, 1070412, 1882709]*km
  ...
  >>> period = 2*pi*(semimajor_axis**3/(G*Mjup))**0.5
  >>> period = period.to('d')
  ...
  >>> for moon, period in zip(moons, period):
  ...     print('{}: {:04.2f}'.format(moon, period))
  Io: 1.77 d
  Europa: 3.55 d
  Ganymede: 7.15 d
  Callisto: 16.69 d

Let's break up this example into a few components so you can see what's going
on. First, we import the unit symbols we need from the ``unyt`` namespace::

  >>> from unyt import Mjup, G, km

The ``unyt`` namespace has a large numbe of units and physical constants you
can import to apply units to data in your own code. You can see how that works
in the example::

  >>> semimajor_axis = [421700, 671034, 1070412, 1882709]*km
  >>> semimajor_axis
  unyt_array([ 421700.,  671034., 1070412., 1882709.], 'km')

By multiplying by ``km``, we converted the python list into a
:class:`unyt.array.unyt_array` instance. This is a class that's built
into ``unyt``, has units attached to it, and knows how to convert itself
into different dimensionally equivalent units::

  >>> semimajor_axis.value
  array([ 421700.,  671034., 1070412., 1882709.])
  >>> semimajor_axis.units
  km
  >>> print(semimajor_axis.to('AU'))
  [0.00281889 0.00448559 0.00715526 0.01258513] AU

Next, we calculated the orbital period by translating the orbital period
formula to Python and then converting the answer to the units we want in the
end, days::

  >>> period = 2*pi*(semimajor_axis**3/(G*Mjup))**0.5
  >>> period
  unyt_array([0.00483381, 0.00970288, 0.01954837, 0.04559936], 'km**(3/2)*s/cm**(3/2)')
  >>> period.to('d')
  unyt_array([ 1.76919479,  3.55129736,  7.1547869 , 16.68956617], 'd')

Note that we haven't added any conversion factors between different units,
that's all handled internally by ``unyt``. Also note how the intermediate result
ended up with complicated, ugly units, but the :meth:`unyt.array.unyt_array.to`
method was able to automagically handle the conversion to days.

Arithmetic and units
--------------------

The real power of working with ``unyt`` is its ability to add, subtract,
multiply, and divide quantities and arrays with units in mathematical formulas
while atuomatically handling unit conversions and detecting
when you have made a mistake in your units in a mathematical formula. To see
what I mean by that, let's take a look at the following examples::

  >>> from unyt import cm, m, ft, yard
  >>> print("{}, {}, {}, {}".format(cm, m, ft, yard))
  1.0 cm, 1.0 m, 1.0 ft, 1.0 yd
  >>> print(3*cm + 4*m - 5*ft + 6*yard)
  799.24 cm

Despite the fact that the four unit symbols used in the above example have four
different units, ``unyt`` is able to automatically convert the value of all
three units into a common unit and return the result in those units. Note
that for expressions where the return units are ambiguous, ``unyt`` always
returns data in the units of the leftmost object in an expression::

  >>> print(4*m + 3*cm - 5*ft + 6*yard)  # doctest: +FLOAT_CMP
  7.9924 m

One can also form more complex units out of atomic unit symbols. For example, here
is how we'd create an array with units of meters per second::

  >>> from unyt import m, s
  >>> velocities = [20, 22, 25]*m/s
  >>> print(velocities.to('mile/hr'))
  [44.73883704 49.21272074 55.9235463 ] mile/hr

Similarly one can multiply two units together to create new compound units::

  >>> from unyt import N, m
  >>> energy = 3*N * 4*m
  >>> print(energy)
  12.0 N*m
  >>> print(energy.to('erg'))
  120000000.0 erg

In general, one can multiple or divide by an arbitrary rational power of a unit symbol. Most commonly this shows up in mathematical formulas in terms of square roots. For example, let's calculate the gravitational free-fall time for a person
to fall from the surface of the Earth through to a hole dug all the way to the center of the Earth. It turns out that this time `is given by <https://en.wikipedia.org/wiki/Free-fall_time>`_::

.. math::

   t_{\rm ff} = \sqrt{\frac{3\pi}{32 G \rho}}

where :math:`\rho` is the average density of the Earth.

  >>> from unyt import G, Mearth, Rearth
  >>> from math import pi
  >>> import numpy as np
  ...
  >>> rho = Mearth / (4./3 * pi* Rearth**3)
  >>> print(rho.to('g/cm**3'))
  5.581225129861077 g/cm**3
  >>> tff = np.sqrt(3*pi/(32*G*rho))
  >>> print(tff.to('min'))
  14.8202885145703 min

If you make a mistake by adding two things that have different dimensions, ``unyt`` will raise an error to let you know that you have a bug in your code::

  >>> from unyt import kg, m
  >>> kg + m  # doctest: +IGNORE_EXCEPTION_DETAIL
  Traceback (most recent call last):
  ...
  unyt.exceptions.UnitOperationError: The <ufunc 'add'> operator for unyt_arrays with units (kg) and (m) is not well defined.

while this example is trivial when one writes more complicated formulae it can
be easy to accidentally write expressions that are not dimensionally sound.

Sometimes this can be annoying to deal with, particularly if one is mixing data
that has units attached with data from some outside source with no units. To
quickly patch over this lack of unit metadata (which could be applied by
explicitly attaching units at I/O time), one can use the ``unit_quantity``
attribute of the :class:`unyt.array.unyt_array` class to quickly apply units::

  >>> from unyt import cm, s
  >>> velocities = [10, 20, 30] * cm/s
  >>> velocities + 12  # doctest: +IGNORE_EXCEPTION_DETAIL
  Traceback (most recent call last):
  ...
  unyt.exceptions.UnitOperationError: The <ufunc 'add'> operator for unyt_arrays with units (cm/s) and (dimensionless) is not well defined.
  >>> velocities + 12*velocities.units
  unyt_array([22., 32., 42.], 'cm/s')

Transcendental functions
------------------------
