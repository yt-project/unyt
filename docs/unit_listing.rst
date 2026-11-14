.. _unit-listing:

Listing of Units
================

.. show_all_units::

.. _unit-prefix-listing:

Listing of Unit Prefixes
========================

The following prefixes are supported by ``unyt`` for units that allow prefixes.
Prefix symbols are case-sensitive: for example, ``mm`` is a millimeter, while
``Mm`` is a megameter. The scale factor multiplies the unprefixed unit.

.. list-table:: Supported prefixes
   :header-rows: 1
   :widths: 25 30 45

   * - Prefix
     - Symbol
     - Scale factor
   * - yotta
     - ``Y``
     - :math:`10^{24}`
   * - zetta
     - ``Z``
     - :math:`10^{21}`
   * - exa
     - ``E``
     - :math:`10^{18}`
   * - peta
     - ``P``
     - :math:`10^{15}`
   * - tera
     - ``T``
     - :math:`10^{12}`
   * - giga
     - ``G``
     - :math:`10^{9}`
   * - mega
     - ``M``
     - :math:`10^{6}`
   * - kilo
     - ``k``
     - :math:`10^{3}`
   * - hecto
     - ``h``
     - :math:`10^{2}`
   * - deca
     - ``da``
     - :math:`10^{1}`
   * - deci
     - ``d``
     - :math:`10^{-1}`
   * - centi
     - ``c``
     - :math:`10^{-2}`
   * - milli
     - ``m``
     - :math:`10^{-3}`
   * - micro
     - ``u``, ``µ``, ``μ``
     - :math:`10^{-6}`
   * - nano
     - ``n``
     - :math:`10^{-9}`
   * - pico
     - ``p``
     - :math:`10^{-12}`
   * - femto
     - ``f``
     - :math:`10^{-15}`
   * - atto
     - ``a``
     - :math:`10^{-18}`
   * - zepto
     - ``z``
     - :math:`10^{-21}`
   * - yocto
     - ``y``
     - :math:`10^{-24}`

For micro, ASCII ``u`` (U+0075), the micro sign ``µ`` (U+00B5), and the Greek
small letter mu ``μ`` (U+03BC) are accepted. The two Unicode symbols look similar
but are different characters; ``um``, ``µm``, and ``μm`` all denote a micrometer.

.. _constant-listing:

Listing of Physical Constants
=============================

.. show_all_constants::
