====
unyt
====


.. image:: https://img.shields.io/pypi/v/unyt.svg
        :target: https://pypi.python.org/pypi/unyt

.. image:: https://img.shields.io/travis/yt-project/unyt.svg
        :target: https://travis-ci.org/yt-project/unyt

.. image:: https://ci.appveyor.com/api/projects/status/4j1nxunkj759pgo0?svg=true
        :target: https://ci.appveyor.com/project/ngoldbaum/unyt

.. image:: https://readthedocs.org/projects/unyt/badge/?version=latest
        :target: https://unyt.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://codecov.io/gh/yt-project/unyt/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/yt-project/unyt
        :alt: Test Coverage


A package for handling numpy arrays with units

This package only depends on `numpy`, `sympy`, `six` and a backport of
`lru_cache`.  Notably, it does *not* depend on `yt`.


Features
--------

Often writing code that deals with data that has units can be confusing. A
function might return an array but at least with plain NumPy arrays, there is no
way to easily tell what the units of the data are without somehow knowing a
prioi.

The ``unyt`` package provides a subclass of NumPy's ``ndarray`` class that knows
about units. For example, one could do:

    >>> import unyt
    ...
    >>> cars = ['toyota', 'volkswagen', 'honda']
    >>> distance_traveled = [3.4, 5.8, 7.2] * unyt.mile
    ...
    >>> print(distance_traveled.to('km'))
    [ 5.4717696  9.3341952 11.5872768] km

And a whole lot more! See `the documentation <http://unyt.readthedocs.io>`_ for
more examples as well as full API docs.

Code of Conduct
---------------

The ``unyt`` package is part of `The yt Project
<https://yt-project.org>`_. Participating in ``unyt`` development therefore
happens under the auspices of the `yt community code of conduct
<http://yt-project.org/doc/developing/developing.html#yt-community-code-of-conduct>`_. If
for any reason you feel that the code of conduct has been violated, please send
an e-mail to confidential@yt-project.org with details describing the
incident. All emails sent to this address will be treated with the strictest
confidence by an individual who does not normally participate in yt development.

License
-------

The unyt package is licensed under the BSD 3-clause license. If you make use of
unyt in a publication we would appreciate a mention in the text of the paper or
in the acknowledgements.

Credits
-------

This package was created with Cookiecutter_ and the
`audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
