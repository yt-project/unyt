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

The ``unyt`` test suite makes use of ``uv``, which makes it
easy to run tests on multiple python versions.

This guide uses ``uv`` extensively, but note that you do not have to use
``uv``-managed Python interpreters, see uv's documentation
https://docs.astral.sh/uv/concepts/python-versions/#requiring-or-disabling-managed-python-versions


1. Fork the ``unyt`` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/unyt.git

4. Setup a development environment

    $ cd unyt/
    $ uv sync

5. Create a branch for local development::

    $ git switch -c name-of-your-bugfix-or-feature

6. Edit files in the ``unyt`` repository, using your local python installation
   to test your edits.

7. When you're done making changes, check that your changes pass linting,
   and run the tests

    $ pre-commit run --all-files
    $ uv run --group test pytest unyt --doctest-modules --doctest-rst --doctest-plus

8. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

9. Submit a pull request through the GitHub website.

Testing unyt
------------

We use the ``pytest`` test runner as well as ``uv`` to manage
running tests on various versions of python.

To run the tests on your copy of the ``unyt`` repository using your current
python environment, run ``pytest`` in the root of the repository using the
following arguments::

   $ cd unyt/
   $ uv run --group test pytest unyt --doctest-modules --doctest-rst --doctest-plus

These enable testing the docstrings and doctest examples scattered throughout
the unyt and its documentation.

Integration tests require additional dependencies, which you can get by passing
the following additional argument to ``uv run``: ``--group integration``

If you would like to run the tests on different Python versions, add
``--python=<version>``, where ``<version>`` can be a version number or a path.


Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests for functionality that is not already
   tested. We strive for 100% test coverage and pull requests should not add any
   new untested code. You can generate coverage reports locally by first
   generating coverage statistics with ``pytest`` through ``coverage`` as ``uv
   run --group covcheck coverage run -m pytest``. To generate reports from the
   coverage statistics, you can use ``uv run coverage report`` to print to
   screen or ``uv run coverage html`` to generate an html that you can open in a
   browser.
2. If the pull request adds functionality the docs should be updated. If your
   new functionality adds new functions or classes to the public API, please add
   docstrings. If you modified an existing function or class in the public API,
   please update the existing docstrings. If you modify private implementation
   details, please use your judgment on documenting it with comments or
   docstrings.
3. The pull request should work for Python 3.10, 3.11, 3.12, 3.13, and 3.14.
   Check in the GitHub interface for your pull request and make sure that the
   tests pass for all supported Python versions.

Deploying
---------

A reminder for the maintainers on how to deploy.  Make sure all your changes are
committed (including an entry in HISTORY.rst and adding any new contributors to
AUTHORS.rst).

The version number must also be updated, preferably in its own PR.
The preferred way to do it is through ``uv version --bump <major|minor|patch>``.


If doing a bugfix release, you may need to create a - or checkout an existing -
backport branch named ``vX.Y.x`` where ``X`` and ``Y`` represent the relevant
major and minor version numbers, and the lowercase ``x`` is literal. Otherwise
you may just release from the development branch. Once you are ready, create
a tag:

  $ git tag vX.Y.Z            # where X, Y and Z should be meaningful major, minor and micro version numbers

In any case, take care that the version number for the tag matches what was
chosen for the version number.

If the tests pass you can then subsequently manually do a test publication::

  $ uv build # builds a source distribution and a wheel under dist/
  $ uvx twine check dist/*
  $ uv publish --publish-url https://test.pypi.org/legacy/

Then, using a fresh environment here, and from outside the repository,
test the result::

  $ uv sync --only-group test
  $ uv pip install unyt --reinstall --only-binary --index-url https://test.pypi.org/simple/
  $ uv run --no-sync python -c "import unyt; unyt.test()"
  $ uv pip install unyt --reinstall --no-binary --index-url https://test.pypi.org/simple/
  $ uv run --no-sync python -c "import unyt; unyt.test()"

Finally, if everything works well, push the tag to the upstream repository::

  $ git push upstream --tag   # assuming the mother repo yt-project/unyt is set as a remote under the name "upstream"
