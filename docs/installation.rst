.. highlight:: shell

============
Installation
============


Stable release
--------------

To install :mod:`unyt`, run this command in your terminal:

.. code-block:: console

    $ pip install unyt

If you have a C compiler available, we also suggest installing `fastcache`_,
which will improve the performance of `SymPy`_.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

If you use `conda`_, :mod:`unyt` is available via `conda-forge`_:

.. code-block:: console

   $ conda install -c conda-forge unyt

It is not necessary to explicitly install ``fastcache`` if you use ``conda``
because it will be installed automatically as a dependency of ``SymPy``.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/
.. _fastcache: https://github.com/pbrady/fastcache
.. _SymPy: http://sympy.org/
.. _conda: https://conda.io/
.. _conda-forge: https://conda-forge.org/

From source
-----------

The sources for :mod:`unyt` can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/yt-project/unyt

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/yt-project/unyt/tarball/master

Once you have a copy of the source, you can install it by navigating to the root of the installation and issuing the following command:

.. code-block:: console

    $ pip install .

If you would like to make an "editable" where you can directly edit the
Python source files of the installed version of ``unyt``, then you can do:

.. code-block:: console

    $ pip install -e .

.. _Github repo: https://github.com/yt-project/unyt
.. _tarball: https://github.com/yt-project/unyt/tarball/master

Running the tests
-----------------

You can check that :mod:`unyt` is working properly by running the unit tests
on your intalled copy:

.. doctest::

  >>> import unyt
  >>> unyt.test()  # doctest: +SKIP
