====
unyt
====


.. image:: https://img.shields.io/pypi/v/unyt.svg
        :target: https://pypi.python.org/pypi/unyt

.. image:: https://img.shields.io/travis/yt-project/unyt.svg
        :target: https://travis-ci.org/yt-project/unyt

.. image:: https://readthedocs.org/projects/unyt/badge/?version=latest
        :target: https://unyt.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


A package for handling numpy arrays with units


Features
--------

Often writing code that deals with data that has units can be confusing. A function might return an array but at least with plain NumPy arrays, there is no way to easily tell what the units of the data are without somehow knowing a prioi.

The ``unyt`` package provides a subclass of NumPy's ``ndarray`` class that knows about units. For example, one could do::

  import unyt

  cars = ['toyota', 'volkswagen', 'honda']
  distance_traveled = [3.4, 5.8, 7.2] * unyt.kilometer

  print(distance_traveled)


License
-------

The unyt package is licensed under the BSD 3-clause license. If you make use
of unyt in a publication we would appreciate a mention in the text of the paper or in the acknowledgements.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
