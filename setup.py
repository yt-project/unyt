"""The setup script."""

import os
import sys

from setuptools import find_packages, setup

# ensure the current directory is on sys.path
# so versioneer can be imported when pip uses
# PEP 517/518 build rules
sys.path.append(os.path.dirname(__file__))

import versioneer  # NOQA

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = ["numpy>=1.17.5", "sympy>=1.5"]

test_requirements = ["pytest"]

setup(
    author="The yt project",
    author_email="yt-dev@python.org",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="A package for handling numpy arrays with units",
    install_requires=requirements,
    license="BSD license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    package_data={"unyt": ["tests/data/old_json_registry.txt"]},
    keywords="unyt",
    name="unyt",
    packages=find_packages(include=["unyt", "unyt.tests", "unyt.tests.data"]),
    test_suite="tests",
    tests_require=test_requirements,
    python_requires=">=3.8",
    url="https://github.com/yt-project/unyt",
    version=versioneer.get_version(),
    zip_safe=False,
    cmdclass=versioneer.get_cmdclass(),
)
