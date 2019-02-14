====
unyt
====


.. image:: https://img.shields.io/pypi/v/unyt.svg
        :target: https://pypi.python.org/pypi/unyt

.. image:: https://img.shields.io/conda/vn/conda-forge/unyt.svg
        :target: https://anaconda.org/conda-forge/unyt
        :alt: conda-forge

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

.. image:: http://joss.theoj.org/papers/dbc27acb614dd33eb02b029ef20e7fe7/status.svg
        :target: http://joss.theoj.org/papers/dbc27acb614dd33eb02b029ef20e7fe7
        :alt: Code Paper

|

 .. image:: docs/_static/yt_logo_small.png
         :target: https://yt-project.org
         :alt: The yt Project

A package for handling numpy arrays with units.

Often writing code that deals with data that has units can be confusing. A
function might return an array but at least with plain NumPy arrays, there is no
way to easily tell what the units of the data are without somehow knowing *a
priori*.

The ``unyt`` package (pronounced like "unit") provides a subclass of NumPy's
``ndarray`` class that knows about units. For example, one could do:

    >>> import unyt as u
    >>> distance_traveled = [3.4, 5.8, 7.2] * u.mile
    >>> print(distance_traveled.to('km'))
    [ 5.4717696  9.3341952 11.5872768] km

And a whole lot more! See `the documentation <http://unyt.readthedocs.io>`_ for
installation instructions, more examples, and full API reference.

This package only depends on ``numpy`` and ``sympy``.  Notably, it does *not*
depend on ``yt`` and it is written in pure Python.

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

The unyt package is licensed under the BSD 3-clause license.

Citation
--------

If you make use of unyt in work that leads to a publication we would appreciate
a mention in the text of the paper or in the acknowledgements along with a
citation to our `paper
<https://joss.theoj.org/papers/dbc27acb614dd33eb02b029ef20e7fe7>`_ in the
Journal of Open Source Software. You can use the following BibTeX::

 @article{Goldbaum2018,
   doi = {10.21105/joss.00809},
   url = {https://doi.org/10.21105/joss.00809},
   year  = {2018},
   month = {aug},
   publisher = {The Open Journal},
   volume = {3},
   number = {28},
   pages = {809},
   author = {Nathan J. Goldbaum and John A. ZuHone and Matthew J. Turk and Kacper Kowalik and Anna L. Rosen},
   title = {unyt: Handle,  manipulate,  and convert data with units in Python},
   journal = {Journal of Open Source Software}
 }

Or the following citation format:

  Goldbaum et al., (2018). unyt: Handle, manipulate, and convert data with units in Python . Journal of Open Source Software, 3(28), 809, https://doi.org/10.21105/joss.00809
