========================
Working with :mod:`unyt`
========================

Basic Usage
+++++++++++

To use unyt in a project::

  >>> import unyt

The top-level :mod:`unyt` namespace defines both a number of useful functions as
well as a number of unit symbols you can use to attach units to NumPy arrays or
python lists.

An Example from High School Physics
-----------------------------------

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
thing symbolically and let :mod:`unyt` handle the unit conversions.

To do this we'll need to know the mass of Jupiter (fortunately that is built
into :mod:`unyt`) and the semimajor axis of the orbits of Jupiter's moons, which
we can look up from `Wikipedia
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
on. First, we import the unit symbols we need from the :mod:`unyt` namespace::

  >>> from unyt import Mjup, G, km

The :mod:`unyt` namespace has a large number of units and physical constants you
can import to apply units to data in your own code. You can see how that works
in the example::

  >>> semimajor_axis = [421700, 671034, 1070412, 1882709]*km
  >>> semimajor_axis
  unyt_array([ 421700.,  671034., 1070412., 1882709.], 'km')

By multiplying by ``km``, we converted the python list into a
:class:`unyt.unyt_array <unyt.array.unyt_array>` instance. This is a class
that's built into :mod:`unyt`, has units attached to it, and knows how to
convert itself into different dimensionally equivalent units::

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
  unyt_array([ 4.83380797,  9.70288268, 19.54836529, 45.5993645 ], 'km**(3/2)*s/m**(3/2)')
  >>> period.to('d')
  unyt_array([ 1.76919479,  3.55129736,  7.1547869 , 16.68956617], 'd')

Note that we haven't added any conversion factors between different units,
that's all handled internally by :mod:`unyt`. Also note how the intermediate
result ended up with complicated, ugly units, but the :meth:`unyt_array.to
<unyt.array.unyt_array.to>` method was able to automagically handle the
conversion to days.

It's also worth emphasizing that :mod:`unyt` represents powers using standard
python syntax. This means you must use `**` and not `^`, even when writing a
unit as a string:

  >>> from unyt import kg, m
  >>> print((10*kg/m**3).to('g/cm**3'))
  0.01 g/cm**3

Arithmetic and units
--------------------

The real power of working with :mod:`unyt` is its ability to add, subtract,
multiply, and divide quantities and arrays with units in mathematical formulas
while automatically handling unit conversions and detecting when you have made a
mistake in your units in a mathematical formula. To see what I mean by that,
let's take a look at the following examples::

  >>> from unyt import cm, m, ft, yard
  >>> print(3*cm + 4*m - 5*ft + 6*yard)
  799.24 cm

Despite the fact that the four unit symbols used in the above example correspond
to four different units, :mod:`unyt` is able to automatically convert the value
of all three units into a common unit and return the result in those units. Note
that for expressions where the return units are ambiguous, :mod:`unyt` always
returns data in the units of the leftmost object in an expression::

  >>> print(4*m + 3*cm - 5*ft + 6*yard)  # doctest: +FLOAT_CMP
  7.9924 m

One can also form more complex units out of atomic unit symbols. For example, here is how we'd create an array with units of meters per second and print out the values in the array in miles per hour::

  >>> from unyt import m, s
  >>> velocities = [20, 22, 25]*m/s
  >>> print(velocities.to('mile/hr'))
  [44.73872584 49.21259843 55.9234073 ] mile/hr

Similarly one can multiply two units together to create new compound units::

  >>> from unyt import N, m
  >>> energy = 3*N * 4*m
  >>> print(energy)
  12.0 N*m
  >>> print(energy.to('erg'))
  120000000.0 erg

In general, one can multiple or divide by an arbitrary rational power of a unit symbol. Most commonly this shows up in mathematical formulas in terms of square roots. For example, let's calculate the gravitational free-fall time for a person
to fall from the surface of the Earth through to a hole dug all the way to the center of the Earth. It turns out that this time `is given by <https://en.wikipedia.org/wiki/Free-fall_time>`_:

.. math::

   t_{\rm ff} = \sqrt{\frac{3\pi}{32 G \rho}}

where :math:`\rho` is the average density of the Earth.

  >>> from unyt import G, Mearth, Rearth
  >>> from math import pi
  >>> import numpy as np
  ...
  >>> rho = Mearth / (4./3 * pi* Rearth**3)
  >>> print(rho.to('g/cm**3'))
  5.581225129861083 g/cm**3
  >>> tff = np.sqrt(3*pi/(32*G*rho))
  >>> print(tff.to('min'))
  14.820288514570295 min

If you make a mistake by adding two things that have different dimensions,
:mod:`unyt` will raise an error to let you know that you have a bug in your
code:

  >>> from unyt import kg, m
  >>> 3*kg + 5*m  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
  Traceback (most recent call last):
  ...
  unyt.exceptions.UnitOperationError: The <ufunc 'add'> operator for
  unyt_arrays with units "kg" (dimensions "(mass)") and
  "m" (dimensions "(length)") is not well defined.

while this example is trivial when one writes more complicated formulae it can
be easy to accidentally write expressions that are not dimensionally sound.

Sometimes this can be annoying to deal with, particularly if one is mixing data
that has units attached with data from some outside source with no units. To
quickly patch over this lack of unit metadata (which could be applied by
explicitly attaching units at I/O time), one can use the ``units`` attribute of
the :class:`unyt.unyt_array <unyt.array.unyt_array>` class to quickly apply units to a scalar, list, or array:

  >>> from unyt import cm, s
  >>> velocities = [10, 20, 30] * cm/s
  >>> velocities + 12  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
  Traceback (most recent call last):
  ...
  unyt.exceptions.UnitOperationError: The <ufunc 'add'> operator for
  unyt_arrays with units "cm/s" (dimensions "(length)/(time)") and
  "dimensionless" (dimensions "1") is not well defined.
  >>> velocities + 12*velocities.units
  unyt_array([22., 32., 42.], 'cm/s')

Logarithms, Exponentials, and Trigonometric Functions
-----------------------------------------------------

Formally it does not make sense to exponentiate, take the logarithm of, or apply
a transcendental function to a quantity with units. However, the :mod:`unyt`
library makes the practical affordance to allow this, simply ignoring the units
present and returning a result without units. This makes it easy to work with
data that has units both in linear space and in log space:

  >>> from unyt import g, cm
  >>> import numpy as np
  >>> print(np.log10(1e-23*g/cm**3))
  -23.0

The one exception to this rule is for trigonometric functions applied to data with angular units:

  >>> from unyt import degree, radian
  >>> import numpy as np
  >>> print(np.sin(np.pi/4*radian))
  0.7071067811865475
  >>> print(np.sin(45*degree))
  0.7071067811865475

Printing Units
--------------

The print formatting of :class:`unyt_array <unyt.array.unyt_array>` can be
controlled identically to numpy arrays, using ``numpy.setprintoptions``:

  >>> import numpy as np
  >>> import unyt as u
  ...
  >>> np.set_printoptions(precision=4)
  >>> print([1.123456789]*u.km)
  [1.1235] km
  >>> np.set_printoptions(precision=8)

Print a :math:`\rm{\LaTeX}` representation of a set of units using the :meth:`unyt.unit_object.Unit.latex_representation` function or :attr:`unyt.unit_object.Unit.latex_repr` attribute:

  >>> from unyt import g, cm
  >>> (g/cm**3).units.latex_representation()
  '\\frac{\\rm{g}}{\\rm{cm}^{3}}'
  >>> (g/cm**3).units.latex_repr
  '\\frac{\\rm{g}}{\\rm{cm}^{3}}'

Unit Conversions and Unit Systems
+++++++++++++++++++++++++++++++++

Converting Data to Arbitrary Units
----------------------------------

If you have some data that you want to convert to a different set of units and
you know which units you would like to convert it to, you can make use of the
:meth:`unyt_array.to <unyt.array.unyt_array.to>` function:

  >>> from unyt import mile
  >>> (1.0*mile).to('ft')
  unyt_quantity(5280., 'ft')

If you try to convert to a unit with different dimensions, :mod:`unyt` will
raise an error:

  >>> from unyt import mile
  >>> (1.0*mile).to('lb')  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
  Traceback (most recent call last):
  ...
  unyt.exceptions.UnitConversionError: Unit dimensionalities do not match.
  Tried to convert between mile (dim (length)) and lb (dim (mass)).

While we recommend using :meth:`unyt_array.to <unyt.array.unyt_array.to>` in
most cases to convert arrays or quantities to different units, if you would like
to explicitly emphasize that this operation has to do with units, we also
provide the more verbose name :meth:`unyt_array.in_units
<unyt.array.unyt_array.in_units>` which behaves identically to
:meth:`unyt_array.to <unyt.array.unyt_array.to>`.

Converting Units In-Place
-------------------------

The :meth:`unyt_array.to <unyt.array.unyt_array.to>` method makes a copy of the
array data. For most cases this is fine, but when dealing with big arrays, or
when performance is a concern, it sometimes is preferable to convert the data in
an array in-place, without copying the data to a new array. This can be
accomplished with the :meth:`unyt_array.convert_to_units
<unyt.array.unyt_array.convert_to_units>` function:

  >>> from unyt import mile
  >>> data = [1, 2, 3]*mile
  >>> data
  unyt_array([1., 2., 3.], 'mile')
  >>> data.convert_to_units('km')
  >>> data
  unyt_array([1.609344, 3.218688, 4.828032], 'km')

Converting to MKS and CGS Base Units
------------------------------------

If you don't necessarily know the units you want to convert data to ahead of
time, it's often convenient to specify a unit system to convert to. The
:class:`unyt_array <unyt.array.unyt_array>` has built-in conversion methods for
the two most popular unit systems, MKS (meter kilogram second) and CGS
(centimeter gram second). For CGS these are :meth:`unyt_array.in_cgs
<unyt.array.unyt_array.in_cgs>` and :meth:`unyt_array.convert_to_cgs
<unyt.array.unyt_array.convert_to_cgs>`. These functions create a new copy of an
array in CGS units and convert an array in-place to CGS. respectively. For MKS,
there are the :meth:`unyt_array.in_mks <unyt.array.unyt_array.in_mks>`
and :meth:`unyt_array.convert_to_mks <unyt.array.unyt_array.convert_to_mks>` methods, which play analogous roles.

See below for details on CGS and MKS electromagnetic units.

Other Unit Systems
------------------

The :mod:`unyt` library currently has built-in support for a number of unit
systems, as detailed in the table below. Note that all unit systems currently
use "radian" as the base angle unit.

If a unit system in the table below has "Other Units" specified, this is a
mapping from dimension to a unit name. These units override the unit system's
default unit for that dimension. If no unit is explicitly specified of a
dimension then the base unit for that dimension is calculated at runtime by
combining the base units for the unit system into the appropriate dimension.

+--------------+--------------------+--------------------------+
| Unit system  | Base Units         | Other Units              |
+==============+====================+==========================+
| cgs          | cm, g, s           | * Energy: erg            |
|              |                    | * Specific Energy: erg/g |
|              |                    | * Pressure: dyne/cm**2   |
|              |                    | * Force: dyne            |
|              |                    | * Power: erg/s           |
|              |                    | * Magnetic Field: G      |
|              |                    | * Charge: esu            |
|              |                    | * Current: statA         |
+--------------+--------------------+--------------------------+
| mks          | m, kg, s           | * Energy: J              |
|              |                    | * Specific Energy: J/kg  |
|              |                    | * Pressure: Pa           |
|              |                    | * Force: N               |
|              |                    | * Power: W               |
|              |                    | * Magnetic Field: T      |
|              |                    | * Charge: C              |
+--------------+--------------------+--------------------------+
| imperial     | ft, lb, s          | * Energy: ft*lbf         |
|              |                    | * Temperature: R         |
|              |                    | * Pressure: lbf/ft**2    |
|              |                    | * Force: lbf             |
|              |                    | * Power: hp              |
+--------------+--------------------+--------------------------+
| galactic     | kpc, Msun, kyr     | * Energy: kev            |
|              |                    | * Magnetic Field: uG     |
+--------------+--------------------+--------------------------+
| solar        | AU, Mearth, yr     |                          |
+--------------+--------------------+--------------------------+

Note that in MKS units the current unit, Ampere, is a base unit in the unit
system. In CGS units the electromagnetic units like Gauss and statA are
decomposable in terms of the base mass, length, and time units in the unit
system. For this reason quantities defined in E&M units in CGS units are not
readily convertible to MKS units and vice verse since the units are not
dimensionally equivalent. To resolve this, :mod:`unyt` provides a unit
equivalency system, discussed below, to convert data between semantically
equivalent but not dimensionally equal units.

The names ``"SI"``, ``"si"``, and ``"MKS"`` are accepted as alternative names
by :mod:`unyt` for the MKS unit system. Similarly, ``"CGS"`` is acceptable as
a name for the CGS unit system.

You can convert data to a unit system :mod:`unyt` knows about using the
:meth:`unyt_array.in_base <unyt.array.unyt_array.in_base>` and
:meth:`unyt_array.convert_to_base <unyt.array.unyt_array.convert_to_base>`
methods:

  >>> from unyt import g, cm, horsepower
  >>> (1e-9*g/cm**2).in_base('galactic')
  unyt_quantity(4.78843804, 'Msun/kpc**2')
  >>> data = [100, 500, 700]*horsepower
  >>> data
  unyt_array([100., 500., 700.], 'hp')
  >>> data.convert_to_base('mks')
  >>> data
  unyt_array([ 74569.98715823, 372849.93579114, 521989.91010759], 'W')

Defining and Using New Unit Systems
***********************************

To define a new custom unit system, one need only create a new instance of the
:class:`unyt.UnitSystem <unyt.unit_systems.UnitSystem>` class. The class
initializer accepts a set of base units to define the unit system. If you would
like to additionally customize any derived units in the unit system, you can do
this using item setting.

As an example, let's define an atomic unit system based on typical scales for
atoms and molecules:

   >>> from unyt import UnitSystem
   >>> atomic_unit_system = UnitSystem('atomic', 'nm', 'mp', 'fs', 'nK', 'rad')
   >>> atomic_unit_system['energy'] = 'eV'
   >>> atomic_unit_system
   atomic Unit System
    Base Units:
     length: nm
     mass: mp
     time: fs
     temperature: nK
     angle: rad
     current_mks: A
     luminous_intensity: cd
    Other Units:
     energy: eV

   >>> atomic_unit_system['number_density']
   nm**(-3)
   >>> atomic_unit_system['angular_momentum']
   mp*nm**2/fs

Once you have defined a new unit system that will register the new system with a
global registry of unit systems known to the :mod:`unyt` library. That means you
will immediately be able to use it just like the built-in unit systems:

  >>> from unyt import W
  >>> (1.0*W).in_base('atomic')
  unyt_quantity(0.59746607, 'mp*nm**2/fs**3')

If you would like your unit system to include an MKS current unit
(e.g. something that is directly convertible to the MKS Ampere unit), then
specify a ``current_mks_unit`` in the :class:`UnitSystem
<unyt.unit_systems.UnitSystem>` initializer.

Equivalencies
+++++++++++++

An equivalency is a way to define a mapping to convert from one unit to another
even if the two units are not dimensionally equivalent. This usually involves
some sort of shorthand or heuristic understanding of the problem under
consideration. Only use one of these equivalencies if it makes sense to use it
for the problem you are working on.

The :mod:`unyt` library implements the following equivalencies:

* "thermal": conversions between temperature and energy (:math:`E = k_BT`)
* "spectral": conversions between wavelength, frequency, and energy for photons
  (:math:`E = h\nu = hc/\lambda`, :math:`c = \lambda\nu`)
* "mass_energy": conversions between mass and energy (:math:`E = mc^2`)
* "lorentz": conversions between velocity and Lorentz factor
  (:math:`\gamma = 1/\sqrt{1-(v/c)^2}`)
* "schwarzschild": conversions between mass and Schwarzschild radius
  (:math:`R_S = 2GM/c^2`)
* "compton": conversions between mass and Compton wavelength
  (:math:`\lambda = h/mc`)

You can convert data to a specific set of units via an equivalency appropriate
for the units of the data. To see the equivalencies that are available for an
array, use the :meth:`unit_array.list_equivalencies
<unyt.array.unyt_array.list_equivalencies>` method:

  >>> from unyt import gram, km
  >>> gram.list_equivalencies()
  mass_energy: mass <-> energy
  schwarzschild: mass <-> length
  compton: mass <-> length
  >>> km.list_equivalencies()
  spectral: length <-> frequency <-> energy
  schwarzschild: mass <-> length
  compton: mass <-> length

All of the unit conversion methods described above have an ``equivalence``
keyword argument that allows one to optionally specify an equivalence to use for
the unit conversion operation. For example, let's use the ``schwarzschild``
equivalence to calculate the mass of a black hole with a radius of one AU:

  >>> from unyt import AU
  >>> (1.0*AU).to('Msun', equivalence='schwarzschild')
  unyt_quantity(50658673.46804734, 'Msun')

Both the methods that convert data in-place and the ones that return a copy
support optionally specifying equivalence. In addition to the methods described
above, :mod:`unyt` also supplies two more conversion methods that *require* an
equivalence to be specified: :meth:`unyt_array.to_equivalent
<unyt.array.unyt_array.to_equivalent>` and
:meth:`unyt_array.convert_to_equivalent
<unyt.array.unyt_array.convert_to_equivalent>`. These are identical to their
counterparts described above, except they equivalence is a required positional
argument to the function rather than an optional keyword argument. Use these
functions when you want to emphasize that an equivalence is being used.

If the equivalence has optional keyword arguments, these can be passed to the
unit conversion function. For example, here's an example where we specify a
custom mean molecular weight (``mu``) for the ``number_density`` equivalence:

  >>> from unyt import g, cm
  >>> rho = 1e-23 * g/cm**3
  >>> rho.to('cm**-3', equivalence='number_density', mu=1.4)
  unyt_quantity(4.26761476, 'cm**(-3)')

For full API documentation and an autogenerated listing of the built-in
equivalencies in :mod:`unyt` as well as a short usage example for each, see the
:mod:`unyt.equivalencies` API listing.

Dealing with code that doesn't use :mod:`unyt`
++++++++++++++++++++++++++++++++++++++++++++++

Optimally, a function will work the same irrespective of whether the data passed in has units attached or not:

    >>> from unyt import cm
    >>> def square(x):
    ...     return x**2
    >>> print(square(3.))
    9.0
    >>> print(square(3.*cm))
    9.0 cm**2

However in the real world that is not always the case. In this section we describe strategies for dealing with that situation.

Stripping units off of data
---------------------------

The :mod:`unyt` library provides a number of ways to convert
:class:`unyt_quantity <unyt.array.unyt_quantity>` instances into floats and
:class:`unyt_array <unyt.array.unyt_array>` instances into numpy arrays. These
methods either return a copy of the data as a numpy array or return a view
onto the underlying array data owned by a :class:`unyt_array
<unyt.array.unyt_array>` instance.

To obtain a new array containing a copy of the original data, use either the
:meth:`unyt_array.to_value <unyt.array.unyt_array.to_value>` function or the
:attr:`unyt_array.value <unyt.array.unyt_array.value>` or :attr:`unyt_array.v
<unyt.array.unyt_array.v>` properties. All of these are equivalent to passing a
:class:`unyt_array <unyt.array.unyt_array>` to the ``numpy.array()`` function:

  >>> from unyt import g
  >>> import numpy as np
  >>> data = [1, 2, 3]*g
  >>> data
  unyt_array([1., 2., 3.], 'g')
  >>> np.array(data)
  array([1., 2., 3.])
  >>> data.to_value('kg')
  array([0.001, 0.002, 0.003])
  >>> data.value
  array([1., 2., 3.])
  >>> data.v
  array([1., 2., 3.])

Similarly, to obtain a ndarray containing a view of the data in the original
array, use either the :attr:`unyt_array.ndview <unyt.array.unyt_array.ndview>`
or the :attr:`unyt_array.d <unyt.array.unyt_array.d>` properties:

  >>> data.view(np.ndarray)
  array([1., 2., 3.])
  >>> data.ndview
  array([1., 2., 3.])
  >>> data.d
  array([1., 2., 3.])

Applying units to data
----------------------

.. note::

   A numpy array that shares memory with another numpy array points to the array
   that owns the data with the ``base`` attribute. If ``arr1.base is arr2`` is
   ``True`` then ``arr1`` is a view onto ``arr2`` and ``arr2.base`` will be
   ``None``.

When you create a :class:`unyt_array <unyt.array.unyt_array>` instance from a
numpy array, :mod:`unyt` will create a copy of the original array:

  >>> from unyt import g
  >>> data = np.random.random((100, 100))
  >>> data_with_units = data*g
  >>> data_with_units.base is data
  False

If you would like to create a view rather than a copy, you can apply units like this:

  >>> from unyt import unyt_array
  >>> data_with_units = unyt_array(data, g)
  >>> data_with_units.base is data
  True

Any set of units can be used for either of these operations. For example, if
you already have an existing array, you could do this to create a new array
with the same units:

  >>> more_data = [4, 5, 6]*data_with_units.units
  >>> more_data
  unyt_array([4., 5., 6.], 'g')

Working with code that uses ``astropy.units``
---------------------------------------------

The :mod:`unyt` library can convert data contained inside of an Astropy
``Quantity`` instance. It can also produce a ``Quantity`` from an existing
:class:`unyt_array <unyt.array.unyt_array>` instance. To convert data from
``astropy.units`` to :mod:`unyt` use the :func:`unyt_array.from_astropy
<unyt.array.unyt_array.from_astropy>` function:

  >>> from astropy.units import km
  >>> from unyt import unyt_quantity
  >>> unyt_quantity.from_astropy(km)
  unyt_quantity(1., 'km')
  >>> a = [1, 2, 3]*km
  >>> a
  <Quantity [1., 2., 3.] km>
  >>> unyt_array.from_astropy(a)
  unyt_array([1., 2., 3.], 'km')

To convert data *to* ``astropy.units`` use the :meth:`unyt_array.to_astropy <unyt.array.unyt_array.to_astropy>` method:

  >>> from unyt import g, cm
  >>> data = [3, 4, 5]*g/cm**3
  >>> data.to_astropy()
  <Quantity [3., 4., 5.] g / cm3>
  >>> (4*cm).to_astropy()
  <Quantity 4. cm>


Working with code that uses ``Pint``
------------------------------------

The :mod:`unyt` library can also convert data contained in ``Pint`` ``Quantity``
instances. To convert data from ``Pint`` to :mod:`unyt`, use the :func:`unyt_array.from_pint <unyt.array.unyt_array.from_pint>` function:

  >>> from pint import UnitRegistry
  >>> import numpy as np
  >>> ureg = UnitRegistry()
  >>> a = np.arange(4)
  >>> b = ureg.Quantity(a, "erg/cm**3")
  >>> b
  <Quantity([0 1 2 3], 'erg / centimeter ** 3')>
  >>> c = unyt_array.from_pint(b)
  >>> c
  unyt_array([0., 1., 2., 3.], 'erg/cm**3')

And to convert data contained in a :class:`unyt_array <unyt.array.unyt_array>`
instance, use the :meth:`unyt_array.to_pint <unyt.array.unyt_array.to_pint>`
method:

  >>> from unyt import cm, s
  >>> a = 4*cm**2/s
  >>> print(a)
  4.0 cm**2/s
  >>> a.to_pint()
  <Quantity(4.0, 'centimeter ** 2 / second')>
  >>> b = [1, 2, 3]*cm
  >>> b.to_pint()
  <Quantity([1. 2. 3.], 'centimeter')>


Integrating :mod:`unyt` Into a Python Library
+++++++++++++++++++++++++++++++++++++++++++++

The :mod:`unyt` library began life as the unit system for the ``yt`` data
analysis and visualization package, in the form of ``yt.units``. In this role,
:mod:`unyt` was deeply integrated into a larger python library. Due to these
origins, it is straightforward to build applications that ensure unit
consistency by making use of :mod:`unyt`. Below we discuss a few topics that
most often come up when integrating :mod:`unyt` into a new or existing Python library.

User-Defined Units
------------------

Often it is convenient to define new custom units. This can happen when you need
to make use of a unit that the :mod:`unyt` library does not have a definition
for already. It can also happen when dealing with data that uses a custom unit
system or when writing software that needs to deal with such data in a flexible
way, particularly when the units might change from dataset to dataset. This
comes up often when modeling a physical system since it is often convenient to
rescale data from a physical unit system to an internal "code" unit system in
which the values of the variables under consideration are close to unity. This
approach can help minimize floating point round-off error but is often done for
convenience or to non-dimensionalize the problem under consideration.

The :mod:`unyt` library provides two approaches for dealing with this
problem. For more toy one-off use-cases, we suggest using
:func:`unyt.define_unit <unyt.unit_object.define_unit>` which allows defining a
new unit name in the global, default unit system that :mod:`unyt` ships with by
default. For more complex uses cases that need more flexibility, it is possible
to use a custom unit system by ensuring that the data you are working with makes
use of a :class:`UnitRegistry <unyt.unit_registry.UnitRegistry>` customized for
your use case.

Using :func:`unyt.define_unit <unyt.unit_object.define_unit>`
*************************************************************

This function makes it possible to easily define a new unit that is unknown to
the :mod:`unyt` library:

  >>> import unyt as u
  >>> two_weeks = 14.0*u.day
  >>> one_day = 1.0*u.day
  >>> u.define_unit("fortnight", two_weeks)
  >>> print((3*u.fortnight)/one_day)
  42.0 dimensionless

This is primarily useful for one-off definitions of units that the :mod:`unyt` library does not already have predefined.

Unit registries
***************

In these cases it becomes important to understand how ``unyt`` stores unit metadata in an internal database, how to add custom entries to the database, how to modify them, and how to persist custom units.

A common example would be adding a ``code_length`` unit that corresponds to the scaling to from physical lengths to an internal unit system. In practice, this value is arbitrary, but will be fixed for a given problem.

Writing Data with Units to Disk
-------------------------------

Pickles
*******

HDF5 Files
**********

Text Files
**********

Performance Considerations
--------------------------
