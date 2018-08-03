=======
History
=======

unyt began life as a submodule of yt named yt.units.

It was separated from yt.units as its own package in 2018.

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
