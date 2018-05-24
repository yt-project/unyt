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

The `unyt` library provides the `unyt.Unit` class that represents a physical unit. Physical units in the `unyt` class are defined in terms of the dimensions of the unit, a string representation, and a floating point scaling to the MKS unit system. Rather than implementing algebra for unit expressions, we rely on the `SymPy` symbolic algebra library [@SymPy] to handle symbolic algebraic manipulation and the `unyt.Unit` class wraps a SymPy `Expr` object. The `unyt.Unit` object can represent arbitrary units formed out of the seven base dimensions in the SI unit system: time, length, mass, temperature, luminance, electric current, and amount of a substance. In addition, `unyt` supports forming quantities defined in other unit systems - in particular CGS Gaussian units as well as geometrized "natural" units common in relativistic calculations. In addition, `unyt` ships with a number of other useful predefined unit systems based, including imperial units, Planck units, a unit system for calculations in the solar system, and a galactic unit system.

In addition to the `unyt.Unit` class, `unyt` also provides a two subclasses of the NumPy [@NumPy] ndarray [@vanderwalt2011], `unyt.unyt_array` and `unyt.unyt_quantity` to represent arrays and scalars with units attached, respectively. In addition, `unyt` provides a `unyt.UnitRegistry` class to allow custom systems of units, for example to track the internal unit system used in a simulation. These subclasses are tightly integrated with the NumPy ufunc system, which ensures that algebraic calculations that include data with units automatically check to make sure the units are consistent, and allow automatic converting of the final answer of a calculation into a convenient unit.

# Background

# Acknowledgements

# References
