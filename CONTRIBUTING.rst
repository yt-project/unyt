.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

Code of Conduct
---------------

The development of ``unyt`` happens in the context of the `yt community code
of conduct
<http://yt-project.org/doc/developing/developing.html#yt-community-code-of-conduct>`_.
If for any reason you feel that the code of conduct has been violated in the
context of ``unyt`` development, please send an e-mail to
confidential@yt-project.org with details describing the incident. All emails
sent to this address will be treated with the strictest confidence by an
individual who does not normally participate in yt development.

Types of Contributions
----------------------

You can contribute in many ways:

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/yt-project/unyt/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in
  troubleshooting. This includes things like Python version and versions of any
  libraries being used, including unyt.
* If possible, detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

unyt could always use more documentation, whether as part of the
official unyt docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at
https://github.com/yt-project/unyt/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up ``unyt`` for local development.

The ``unyt`` test suite makes use of the ``tox`` test runner, which makes it
easy to run tests on multiple python versions. However, this means that if all
of the python versions needed by ``tox`` are not available, many of the ``tox``
tests will fail with errors about missing python executables.

This guide makes use of ``pyenv`` to set up all of the Python versions used in
the unyt test suite. You do not have to use ``pyenv`` if you have other ways of
managing your python evironment using your operating system's package manager or
``conda``.

1. Fork the ``unyt`` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/unyt.git
3. Install ``pyenv``::

    $ git clone https://github.com/pyenv/pyenv.git $HOME/.pyenv
    $ export PYENV_ROOT="$HOME/.pyenv"
    $ export PATH="$HOME/.pyenv/bin:$PATH
    $ eval "$(pyenv init -)"
    $ pyenv install -s 3.5.9
    $ pyenv install -s 3.6.10
    $ pyenv install -s 3.7.6
    $ pyenv install -s 3.8.1
    $ pip install tox tox-pyenv

3. Install your local copy into a virtualenv or conda environment. You can also
   use one of the python interpreters we installed using ``pyenv``:

    $ pyenv local 3.8.1
    $ cd unyt/
    $ python setup.py develop

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8, format
   the code with black, and run the tests, including testing several Python
   versions with tox::

    $ flake8 unyt
    $ black ./
    $ pytest --doctest-modules --doctest-rst --doctest-plus
    $ pyenv local 3.5.9 3.6.10 3.7.6 3.8.1
    $ tox
    $ pyenv local 3.8.1

   To get ``flake8``, ``black``, ``pytest``, ``pytest-doctestplus``, and
   ``tox``, just ``pip`` or ``conda`` install them into your python environment,
   as appropriate. For a ``pyenv`` environment you would use ``pip``.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Testing unyt
------------

We use the ``pytest`` test runner as well as the ``tox`` test wrapper to manage
running tests on various versions of python.

To run the tests on your copy of the ``unyt`` repository using your current
python evironment, run ``pytest`` in the root of the repository using the
following arguments::

   $ cd unyt/
   $ pytest --doctest-modules --doctest-rst --doctest-plus

These enable testing the docstrings and doctest examples scattered throughout
the unyt and its documentation.
   
You will need to install ``pytest`` and ``pytest-doctestplus`` to run this
command. Some tests depend on ``h5py``, ``Pint``, ``astropy``, ``matplotlib``
``black``, and ``flake8`` being installed.

If you would like to run the tests on multiple python versions, first ensure
that you have multiple python versions visible on your ``$PATH``, then simply
execute ``tox`` in the root of the ``unyt`` repository. For example, using the
``pyenv`` environment we set up above::

   $ pyenv local 3.5.9 3.6.10 3.7.6 3.8.1
   $ cd unyt
   $ tox

The ``tox`` package itself can be installed using the ``pip`` associated with
one of the python installations. See the ``tox.ini`` file in the root of the
repository for more details about our ``tox`` setup. Note that you do not need
to install anything besides ``tox`` and the ``python`` versions needed by
``tox`` for this to work, ``tox`` will handle setting up the test environment,
including installing any necessary dependencies via ``pip``.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests for functionality that is not already
   tested. We strive for 100% test coverage and pull requests should not add any
   new untested code. Use the `codecov.io reports
   <https://codecov.io/gh/yt-project/unyt>`_ on the pull request to gauge
   coverage. You can also generate coverage reports locally by running the
   ``tox`` tests.
2. If the pull request adds functionality the docs should be updated. If your
   new functionality adds new functions or classes to the public API, please add
   docstrings. If you modified an existing function or class in the public API,
   please update the existing docstrings. If you modify private implementation
   details, please use your judgment on documenting it with comments or
   docstrings.
3. The pull request should work for Python 3.5, 3.6, 3.7, and 3.8. Check in the
   GitHub interface for your pull request and make sure that the tests pass for
   all supported Python versions.

Deploying
---------

A reminder for the maintainers on how to deploy.  Make sure all your changes are
committed (including an entry in HISTORY.rst and adding any new contributors to
AUTHORS.rst).  Then run::

  $ git tag v1.x.x
  $ git push upstream master --tags

If the tests pass you can then subsequently manually upload to PyPi::

  $ rm -r build dist
  $ python setup.py sdist bdist_wheel --universal
  $ twine upload dist/*
