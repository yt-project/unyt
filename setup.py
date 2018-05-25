#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
import versioneer

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy>"1.13.0"',
    'sympy',
    'six',
    'backports.functools_lru_cache;python_version<"3.3"'
]

test_requirements = [
    'pytest',
]

setup(
    author="The yt project",
    author_email='yt-dev@python.org',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="A package for handling numpy arrays with units",
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='unyt',
    name='unyt',
    packages=find_packages(include=['unyt']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/yt-project/unyt',
    version=versioneer.get_version(),
    zip_safe=False,
    cmdclass=versioneer.get_cmdclass(),
)
