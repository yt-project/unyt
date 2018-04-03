unyt
====

This is the documentation for ``unyt`` (pronounced like "unit"). This package
provides a python library for working with data that has physical units. It was
originally developed as part of `The yt Project <https://yt-project.org>`_ but
was split out as an independent project so that other Python projects can easily
make use of it.

The unyt library defines the :class:`unyt.array.unyt_array` and
:class:`unyt.array.unyt_quantity` classes for handling arrays and scalars with
units, respectively.

In addition, ``unyt`` provides a number of predefined units and physical constants
that can be directly imported from from the ``unyt`` namespace:

.. doctest::

  >>> from unyt import G, Mearth, Rearth
  >>> v_esc = (2*G*Mearth/Rearth)**(1/2)
  >>> v_esc.to('km/s')
  11.254342299356157 km/s


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   usage
   api
   contributing
   authors
   history

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
