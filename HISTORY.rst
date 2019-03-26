=======
History
=======

2.1.0 (2019-03-26)
------------------

This release includes a few minor new features and bugfixes for the 2.0.0 release.

* Added support for the matmul ``@`` operator. See `PR #80
  <https://github.com/yt-project/unyt/pull/80>`_.
* Allow defining unit systems using ``Unit`` instances instead of string unit
  names. See `PR #71 <https://github.com/yt-project/unyt/pull/71>`_.
* Fix incorrect behavior when ``uhstack`` is called with the ``axis``
  argument. See `PR #73 <https://github.com/yt-project/unyt/pull/73>`_.
* Add ``"rsun"``, ``"lsun"``, and ``"au"`` as alternate spellings for the
  ``"Rsun"``, ``"Lsun"``, and ``"AU"`` units. See `PR #77
  <https://github.com/yt-project/unyt/pull/77>`_.
* Improvements for working with code unit systems. See `PR #78
  <https://github.com/yt-project/unyt/pull/78>`_.
* Reduce impact of floating point round-off noise on unit comparisons. See `PR
  #79 <https://github.com/yt-project/unyt/pull/79>`_.

2.0.0 (2019-03-08)
------------------

``unyt`` 2.0.0 includes a number of exciting new features as well as some
bugfixes. There are some small backwards incompatible changes in this release
related to automatic unit simplification and handling of dtypes. Please see the
release notes below for more details. If you are upgrading from ``unyt 1.x`` we
suggest testing to make sure these changes do not siginificantly impact you. If
you run into issues please let us know by `opening an issue on GitHub
<https://github.com/yt-project/unyt/issues/new>`_.

* Dropped support for Python 2.7 and Python 3.4. Added support for Python 3.7.
* Added ``Unit.simplify()``, which cancels pairs of terms in a unit expression
  that have inverse dimensions and made it so the results of ``unyt_array``
  multiplication and division will automatically simplify units. This means
  operations that combine distinct dimensionally equivalent units will cancel in
  many situations. For example

  .. code-block:: python

     >>> from unyt import kg, g
     >>> print((12*kg)/(4*g))
     3000.0 dimensionless

  older versions of ``unyt`` would have returned ``4.0 kg/g``. See `PR #58
  <https://github.com/yt-project/unyt/pull/58>`_ for more details. This change
  may cause the units of operations to have different, equivalent simplified
  units than they did with older versions of ``unyt``.
* Added the ability to resolve non-canonical unit names to the equivalent
  canonical unit names. This means it is now possible to refer to a unit name
  using an alternative non-canonical unit name when importing the unit from the
  ``unyt`` namespace as well as when a unit name is passed as a string to
  ``unyt``. For example:

  .. code-block:: python

     >>> from unyt import meter, second
     >>> data = 1000.*meter/second
     >>> data.to('kilometer/second')
     unyt_quantity(1., 'km/s')
     >>> data.to('metre/s')
     unyt_quantity(1000., 'm/s')

  The documentation now has a table of units recognized by ``unyt`` along with
  known alternative spellings for each unit.
* Added support for unicode unit names, including ``μm`` for micrometer and ``Ω``
  for ohm. See `PR #59 <https://github.com/yt-project/unyt/pull/59>`_.
* Substantially improved support for data that does not have a ``float64``
  dtype. Rather than coercing all data to ``float64`` ``unyt`` will now preserve
  the dtype of data. Data that is not already a numpy array will be coerced to a
  dtype by calling ``np.array`` internally. Converting integer data to a new
  unit will convert the data to floats, if this causes a loss of precision then
  a warning message will be printed. See `PR #55
  <https://github.com/yt-project/unyt/pull/55>`_ for details. This change may
  cause data to be loaded into ``unyt`` with a different dtype. On Windows the
  default integer dtype is ``int32``, so data may begin to be recognized as
  ``int32`` or converted to ``float32`` where before it was interpreted as
  ``float64`` by default.
* Unit registries are now associated with a unit system. This means that it's
  possible to create a unit registry that is associated with a non-MKS unit
  system so that conversions to "base" units will end up in that non-MKS
  system. For example:

  .. code-block:: python

     >>> from unyt import UnitRegistry, unyt_quantity
     >>> ureg = UnitRegistry(unit_system='cgs')
     >>> data = unyt_quantity(12, 'N', registry=ureg)
     >>> data.in_base()
     unyt_quantity(1200000., 'dyn')

  See `PR #62 <https://github.com/yt-project/unyt/pull/62>`_ for details.
* Added two new utility functions, ``unyt.unit_systems.add_constants`` and
  ``unyt.unit_systems.add_symbols`` that can populate a namespace with a set of
  unit symbols in the same way that the top-level ``unyt`` namespace is
  populated. For example, the author of a library making use of ``unyt`` could
  create an object that users can use to access unit data like this:

  .. code-block:: python

      >>> from unyt.unit_systems import add_symbols
      >>> from unyt.unit_registry import UnitRegistry
      >>> class UnitContainer(object):
      ...    def __init__(self):
      ...        add_symbols(vars(self), registry=UnitRegistry())
      >>> units = UnitContainer()
      >>> units.kilometer
      km
      >>> units.microsecond
      µs

  See `PR #68 <https://github.com/yt-project/unyt/pull/68>`_.
* The ``unyt`` codebase is now automatically formatted by `black
  <https://github.com/ambv/black>`_. See `PR #57
  <https://github.com/yt-project/unyt/pull/57>`_.
* Add missing "microsecond" name from top-level ``unyt`` namespace. See `PR
  #48 <https://github.com/yt-project/unyt/pull/48>`_.
* Add support for ``numpy.argsort`` by defining ``unyt_array.argsort``. See `PR
  #52 <https://github.com/yt-project/unyt/pull/52>`_.
* Add Farad unit and fix issues with conversions between MKS and CGS
  electromagnetic units. See `PR #54
  <https://github.com/yt-project/unyt/pull/54>`_.
* Fixed incorrect conversions between inverse velocities and ``statohm``. See
  `PR #61 <https://github.com/yt-project/unyt/pull/61>`_.
* Fixed issues with installing ``unyt`` from source with newer versions of
  ``pip``. See `PR #63 <https://github.com/yt-project/unyt/pull/62>`_.
* Fixed bug when using `define_unit` that caused crashes when using a custom
  unit registry. Thank you to Bili Dong (@qobilidob on GitHub) for the pull
  request. See `PR #64 <https://github.com/yt-project/unyt/pull/64>`_.

We would also like to thank Daniel Gomez (@dangom), Britton Smith
(@brittonsmith), Lee Johnston (@l-johnston), Meagan Lang (@langmm), Eric Chen
(@ericchen), Justin Gilmer (@justinGilmer), and Andy Perez (@sharkweek) for
reporting issues.

1.0.7 (2018-08-13)
------------------

Trigger zenodo archiving.

1.0.6 (2018-08-13)
------------------

Minor paper updates to finalize JOSS submission.

1.0.5 (2018-08-03)
------------------

``unyt`` 1.0.5 includes changes that reflect the peew review process for the
JOSS method paper. The peer reviewers were Stuart Mumfork (`@cadair
<https://github.com/cadair>`_), Trevor Bekolay (`@tbekolay
<https://github.com/tbekolay>`_), and Yan Grange (`@ygrange
<https://github.com/ygrange>`_). The editor was Kyle Niemeyer (`@kyleniemeyer
<https://github.com/kyleniemeyer>`_). The` `unyt`` development team thank our
reviewers and editor for their help getting the ``unyt`` paper out the door as
well as for the numerous comments and suggestions that improved the paper and
package as a whole.

In addition we'd like to thank Mike Zingale, Meagan Lang, Maksin Ratkin,
DougAJ4, Ma Jianjun, Paul Ivanov, and Stephan Hoyer for reporting issues.

* Added docstrings for the custom exception classes defined by ``unyt``. See `PR
  #44 <https://github.com/yt-project/unyt/pull/44>`_.
* Added improved documentation to the contributor guide on how to run the tests
  and what the PR review guidelines are. See `PR #43
  <https://github.com/yt-project/unyt/pull/43>`_.
* Updates to the text of the method paper in response to reviewer
  suggestions. See `PR #42 <https://github.com/yt-project/unyt/pull/42>`_.
* It is now possible to run the tests on an installed copy of ``unyt`` by
  executing ``unyt.test()``. See `PR #41
  <https://github.com/yt-project/unyt/pull/41>`_.
* Minor edit to LICENSE file so GitHub recognizes it. See `PR #40
  <https://github.com/yt-project/unyt/pull/35>`_. Thank you to Kyle Sunden
  (`@ksunden <https://github.com/ksunden>`_) for the contribution.
* Add spatial frequency as a dimension and added support in the ``spectral``
  equivalence for the spatial frequency dimension. See `PR #38
  <https://github.com/yt-project/unyt/pull/38>`_ Thank you to Kyle Sunden
  (`@ksunden <https://github.com/ksunden>`_) for the contribution.
* Add support for Python 3.7. See `PR #37
  <https://github.com/yt-project/unyt/pull/35>`_.
* Importing ``unyt`` will now fail if ``numpy`` and ``sympy`` are not
  installed. See `PR #35 <https://github.com/yt-project/unyt/pull/35>`_
* Testing whether a unit name is contained in a unit registry using the Python
  ``in`` keyword will now work correctly for all unit names. See `PR #31
  <https://github.com/yt-project/unyt/pull/31>`_.
* The aliases for megagram in the top-level unyt namespace were incorrectly set
  to reference kilogram and now have the correct value. See `PR #29
  <https://github.com/yt-project/unyt/pull/29>`_.
* Make it possible to take scalars to dimensionless array powers with a properly
  broadcasted result without raising an error about units. See `PR #23
  <https://github.com/yt-project/unyt/pull/23>`_.
* Whether or not a unit is allowed to be SI-prefixable (for example, meter is
  SI-prefixable to form centimeter, kilometer, and many other units) is now
  stored as metadata in the unit registry rather than as global state inside
  ``unyt``. See `PR #21 <https://github.com/yt-project/unyt/pull/21>`_.
* Made adjustments to the rules for converting between CGS and MKS E&M units so
  that errors are only raised when going between unit systems and not merely
  when doing a complicated unit conversion invoving E&M units. See `PR #20
  <https://github.com/yt-project/unyt/pull/20>`_.
* ``round(q)`` where ``q`` is a ``unyt_quantity`` instance will no
  longer raise an error and will now return the nearest rounded float.
  See `PR #19 <https://github.com/yt-project/unyt/pull/19>`_.
* Fixed a typo in the readme. Thank you to Paul Ivanov (`@ivanov
  <https://github.com/ivanov>`_) for `the fix
  <https://github.com/yt-project/unyt/pull/16>`_.
* Added smoot as a unit. See `PR #14
  <https://github.com/yt-project/unyt/pull/14>`_.

1.0.4 (2018-06-08)
------------------

* Expand installation instructions
* Mention paper and arxiv submission in the readme.

1.0.3 (2018-06-06)
------------------

* Fix readme rendering on pypi

1.0.2 (2018-06-06)
------------------

* Added a paper to be submitted to the Journal of Open Source Software.
* Tweaks for the readme

1.0.1 (2018-05-24)
------------------

* Don't use setup_requires in setup.py

1.0.0 (2018-05-24)
------------------

* First release on PyPI.
* unyt began life as a submodule of yt named yt.units.
* It was separated from yt.units as its own package in 2018.
