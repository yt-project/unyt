---
title: 'unyt: Handle, manipulate, and convert data with units in Python'
tags:
  - units
  - quantities
  - Python
  - NumPy
  - sympy
authors:
  - name: Nathan J. Goldbaum
    orcid: 0000-0001-5557-267X
    affiliation: 1
  - name: John A. ZuHone
    orcid: 0000-0003-3175-2347
    affiliation: 2
  - name: Matthew J. Turk
    orcid: 0000-0002-5294-0198
    affiliation: 1
  - name: Kacper Kowalik
    orcid: 0000-0003-1709-3744
    affiliation: 1
  - name: Anna Rosen
    orcid: 0000-0003-4423-0660
    affiliation: 2
affiliations:
  - name: National Center for Supercomputing Applications, University of Illinois at Urbana-Champaign
    index: 1
  - name: Harvard-Smithsonian Center for Astrophysics
    index: 2
date: 24 May 2018
bibliography: paper.bib
---

# Summary

Software that processes data with physical units or that models real world
quantities with physical units must have some way of managing units. This might
be as simple as the convention that all floating point numbers are understood to
be in the same physical unit system (for example, the SI MKS units
system). While simple approaches like this do work in practice, they also are
fraught with possible error, both by programmers modifying the code who
unintentionally misinterpret the units, and by users of the software who must
take care to supply data in the correct units or who need to infer the units of
data returned by the software.

The `unyt` library is designed both to aid quick calculations at an interactive
python prompt and to be tightly integrated into a larger Python application or
library. To aid quick calculations, the top-level `unyt` namespace ships with a
large number of predefined units and physical constants to aid setting up quick
calculations without needing to look up unit data or the value of a physical
constant. Using the `unyt` library as an interactive calculation aid only
requires knowledge of basic Python syntax and awareness of a few of the methods
of the `unyt_array` class - for example, the `unyt_array.to()` method to convert
data to a different unit. As the complexity of the usage increases, `unyt`
provides a number of optional features to aid these cases, including custom unit
registries containing both predefined physical units as well as user-defined
units, built-in output to disk via the pickle protocol and to HDF5 files using
the h5py library [@h5py], and round-trip conversions to create units compatible
with other popular Python unit libraries.

Physical units in the `unyt` class are defined in terms of the dimensions of the
unit, a string representation, and a floating point scaling to the MKS unit
system. Rather than implementing algebra for unit expressions, we rely on the
`SymPy` symbolic algebra library [@SymPy] to handle symbolic algebraic
manipulation. The `unyt.Unit` object can represent arbitrary units formed out of
the seven base dimensions in the SI unit system: time, length, mass,
temperature, luminance, electric current, and amount of a substance. In
addition, `unyt` supports forming quantities defined in other unit systems - in
particular CGS Gaussian units common in astrophysics as well as geometrized
"natural" units common in relativistic calculations. In addition, `unyt` ships
with a number of other useful predefined unit systems based, including imperial
units, Planck units, a unit system for calculations in the solar system, and a
galactic unit system.

In addition to the `unyt.Unit` class, `unyt` also provides a two subclasses of
the NumPy [@NumPy] ndarray [@vanderwalt2011], `unyt.unyt_array` and
`unyt.unyt_quantity` to represent arrays and scalars with units attached,
respectively. In addition, `unyt` provides a `unyt.UnitRegistry` class to allow
custom systems of units, for example to track the internal unit system used in a
simulation. These subclasses are tightly integrated with the NumPy ufunc system,
which ensures that algebraic calculations that include data with units
automatically check to make sure the units are consistent, and allow automatic
converting of the final answer of a calculation into a convenient unit.

We direct users interested in usage examples and a guide for integrating `unyt`
into an exiting Python installation to the unyt documentation at hosted at
http://unyt.readthedocs.io/en/latest/.

# Comparison with ``Pint`` and ``astropy.units``

The scientific python ecosystem has a long history of efforts to develop a
library to handle unit conversions and enforce unit consistency. For a
relatively recent review of these efforts, see [@bekolay2013]. While we won't
exhaustively cover extant Python libraries for handling units in this paper, we
will focus on `Pint` [@Pint] and `astropy.units` [@astropy], which both a
provide a robust implementation of an array container with units and are
commonly used in research software projects. At time of writing a GitHub search
for `import astropy.units` returns approximately 10,500 results and a search for
`import pint` returns approximately 1,500 results.

While `unyt` provides functionality that overlaps with `astropy.units` and
`Pint`, there are important differences which we elaborate on below. In
addition, it's worth noting that all three codebases had origins at roughly the
same time period. In the case of `unyt`, it originated via the `dimensionful`
library [@dimensionful] in 2012. A few years later, the `dimensionful` was
elaborated on and improved to become `yt.units`, the unit system for the `yt`
library [@yt] at a `yt` developer workshop in 2013 and was subsequently released
as part of `yt 3.0` in 2014. Similarly, `Pint` initially began development in
2012 according to the git repository logs, and `astropy.units` was added in 2012
and was released as part of `astropy 0.2` in 2013, although the initial
implementation was adapted from the `pynbody` library [@pynbody], which started
in 2010 according to the git repository logs. That is to say, all three
libraries began roughly at the same time and are examples in many ways of
convergent evolution in software. We have decided to repackage and improve
`yt.units` in the form of `unyt` to both make it easier to work on and improve
the unit system and encourage use of the unit system for scientific python users
who do not want to install a heavy-weight dependency like `yt`.

Below we present a table comparing `unyt` with `astropy.units` and
`Pint`. Estimates for lines of code in the library were generated using the
`cloc` tool [@cloc]; blank and comment lines are excluded from the
estimate. Test coverage was estimated using the `coveralls` output for `Pint`
and `astropy.units` and using the `codecov.io` output for `unyt`.

| Library                        | `unyt`         | `astropy.units` | `Pint`     |
|--------------------------------|----------------|-----------------|------------|
| Lines of code                  | 5128           | 10163           | 8908       |
| Lines of code excluding tests  | 3195           | 5504            | 4499       |
| Test Coverage                  | 99.91%         | 93.63%          | 77.44%     |

We offer lines of code as a very rough estimate for the "hackability" of the
codebase. In general, smaller codebases with higher test coverage are have fewer
defects [@Lipow1982; @Koru2007; @Gopinath2014]. This comparison is somewhat
unfair in favor of `unyt` in that `astropy.units` only depends on `NumPy` and
`Pint` has no dependencies, while `unyt` depends on both `sympy` and
`NumPy`. Much of the reduction in the size of the `unyt` library can be
attributed to offloading the handling of algebra to `sympy` rather than needing
to implement the algebra of unit symbols directly in `unyt`. For potential users
who are wary of adding `sympy` as a dependency, that might argue in favor of
using `Pint` in favor of `unyt`.

## Astropy.units

The `astropy.units` subpackage provides a `PrefixUnit` class, a `Quantity` class
that represents both scalar and array data with attached units, and a large
number of predefined unit symbols. The preferred way to create `Quantity`
instances is via multiplication with a `PrefixUnit` instance. Similar to `unyt`,
the `Quantity` class is implemented via a subclass of the `NumPy` `ndarray`
class. Indeed, in many ways the everyday usage patterns of `astropy.units` and
`unyt` are similar, although `unyt` is not quite a drop-in replacement for
`astropy.units` as there are some API differences. The main functional
difference between `astropy.units` and `unyt` is that `astropy.units` is a
subpackage of the larger `astropy` package. This means that depending on
`astropy.units` requires depending on a large collection of astronomically
focused software, including a substantial amount of compiled C code. For users
who are not astronomers or do not need the observational astronomy capabilities
provided by `astropy`, depending on all of `astropy` just to use `astropy.units`
may be a tough sell.

## Pint

The `Pint` package provides a somewhat different API compared with `unyt` and
`astropy.units`. Rather than making units immediately importable from the `Pint`
namespace, instead `Pint` requires users to instantiate a `UnitRegistry`
instance (unrelated to the `unyt.UnitRegistry` class), which in turn has `Unit`
instances as attributes. Just like with `unyt` and `astropy.units`, creating a
`Quantity` instance requires multiplying an array or scalar by a `Unit`
instance. Exposing the `UnitRegistry` directly to all users like this does force
users of the library to think about which system of units they are working with,
which may be beneficial in some cases, however it also means that users have a
bit of extra cognitive overhead they need to deal with every time the use Pint.

In addition, the `Quantity` class provided by `Pint` is not a subclass of
numpy's ndarray. Instead, it is a wrapper around an internal `ndarray`
buffer. This somewhat simplifies the implementation of `Pint` by avoiding the
somewhat arcane process for creating an ndarray subclass, although the `Pint`
`Quantity` class must also be careful to emulate the full `NumPy` `ndarray` API
so that it can be a drop-in replacement for `ndarray`.

Finally, in carefully comparing the output of scripts using `Pint`,
`astropy.units`, and `unyt`, we found that in-place operations making use of a
numpy ufunc will unexpectedly strip units. For example, `np.add(a, b, out=out))`
using `Pint 0.8.1`, this will operate on `a` and `b` as if neither have units
attached. Interestingly, without the `out` keyword, `Pint` does get the correct
answer, so it's possible that this is a bug in `Pint` which we have reported
upstream (see https://github.com/hgrecco/pint/issues/644).

## Performance Comparison

Checking units will always add some overhead over using hard-coded unit
conversion factors. Thus a library that is entrusted with checking units in an
application should incur the minimum possible overhead to avoid triggering
performance regressions after integrating unit checking into an
application. Optimally, a unit library will add zero overhead regardless of the
size of the array. In practice that is not the case for any of the three
libraries under consideration, and there is a minimum array size above which the
overhead of doing a mathematical operation exceeds the overhead of checking
units. It is thus worth benchmarking unit libraries in a fair manner, comparing
with the same operation implemented using plain `NumPy`.

Here we present such a benchmark. We made use of the `perf` [@perf] Python
benchmarking tool, which not only provides facilities for establishing the
statistical significance of a benchmark run, but also can tune a linux system to
turn off operating system and hardware features like CPU throttling that might
introduce variance in a benchmark. We made use of a Dell Latitude E7270 laptop
equipped with an Intel i5-6300U CPU clocked at 2.4 Ghz. The testing environment
was based on `Python 3.6.3` and had `NumPy 1.14.2`, `Sympy 1.1.1`, `fastcache
1.0.2`, `Astropy 3.0.1`, and `Pint 0.8.1` installed. `fastcache` [@fastcache] is
an optional dependency of `sympy` that provides an optimized LRU cache
implemented in C that can substantially speed up `sympy`.

# Acknowledgements

# References