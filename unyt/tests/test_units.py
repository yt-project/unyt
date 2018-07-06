"""
Test symbolic unit handling.




"""

# -----------------------------------------------------------------------------
# Copyright (c) 2018, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------


import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_allclose,
    assert_array_almost_equal_nulp,
    assert_equal
)
import operator
import pytest
from six.moves import cPickle as pickle
from sympy import Symbol
from unyt._testing import assert_allclose_units
from unyt.unit_registry import UnitRegistry
from unyt import electrostatic_unit, elementary_charge_cgs
from unyt.dimensions import (
    mass,
    length,
    time,
    temperature,
    energy,
    magnetic_field_cgs,
    magnetic_field_mks,
    power,
    rate
)
from unyt.exceptions import (
    InvalidUnitOperation,
    UnitsNotReducible,
    UnitConversionError,
)
from unyt.unit_object import (
    default_unit_registry,
    Unit,
    UnitParseError,
)
from unyt.unit_systems import UnitSystem
from unyt._unit_lookup_table import (
    default_unit_symbol_lut,
    unit_prefixes
)
import unyt.unit_symbols as unit_symbols
from unyt._physical_ratios import (
    m_per_pc,
    sec_per_year,
    m_per_km,
    m_per_mpc,
    mass_sun_kg
)


def test_no_conflicting_symbols():
    """
    Check unit symbol definitions for conflicts.

    """
    full_set = set(default_unit_symbol_lut.keys())

    # go through all possible prefix combos
    for symbol in default_unit_symbol_lut.keys():
        if default_unit_symbol_lut[symbol][4]:
            keys = unit_prefixes.keys()
        else:
            keys = [symbol]
        for prefix in keys:
            new_symbol = "%s%s" % (prefix, symbol)

            # test if we have seen this symbol
            assert new_symbol not in full_set, ("Duplicate symbol: %s" %
                                                new_symbol)

            full_set.add(new_symbol)


def test_dimensionless():
    """
    Create dimensionless unit and check attributes.

    """
    u1 = Unit()

    assert u1.is_dimensionless
    assert u1.expr == 1
    assert u1.base_value == 1
    assert u1.dimensions == 1
    assert u1 != 'hello!'
    assert (u1 == 'hello') is False

    u2 = Unit("")

    assert u2.is_dimensionless
    assert u2.expr == 1
    assert u2.base_value == 1
    assert u2.dimensions == 1

    assert_equal(u1.latex_repr, '')
    assert_equal(u2.latex_repr, '')


def test_create_from_string():
    """
    Create units with strings and check attributes.

    """

    u1 = Unit("kg * m**2 * s**-2")
    assert u1.dimensions == energy
    assert u1.base_value == 1.0

    # make sure order doesn't matter
    u2 = Unit("m**2 * s**-2 * kg")
    assert u2.dimensions == energy
    assert u2.base_value == 1.0

    # Test rationals
    u3 = Unit("kg**0.5 * m**-0.5 * s**-1")
    assert u3.dimensions == magnetic_field_cgs
    assert u3.base_value == 1.0

    # sqrt functions
    u4 = Unit("sqrt(kg)/sqrt(m)/s")
    assert u4.dimensions == magnetic_field_cgs
    assert u4.base_value == 1.0

    # commutative sqrt function
    u5 = Unit("sqrt(kg/m)/s")
    assert u5.dimensions == magnetic_field_cgs
    assert u5.base_value == 1.0

    # nonzero CGS conversion factor
    u6 = Unit("Msun/pc**3")
    assert u6.dimensions == mass/length**3
    assert_array_almost_equal_nulp(np.array([u6.base_value]),
                                   np.array([mass_sun_kg/m_per_pc**3]))

    with pytest.raises(UnitParseError):
        Unit('m**m')
    with pytest.raises(UnitParseError):
        Unit('m**g')
    with pytest.raises(UnitParseError):
        Unit('m+g')
    with pytest.raises(UnitParseError):
        Unit('m-g')
    with pytest.raises(UnitParseError):
        Unit('hello!')

    cm = Unit('cm')
    data = 1*cm

    assert Unit(data) == cm


def test_create_from_expr():
    """
    Create units from sympy Exprs and check attributes.

    """
    pc_mks = m_per_pc
    yr_mks = sec_per_year

    # Symbol expr
    s1 = Symbol("pc", positive=True)
    s2 = Symbol("yr", positive=True)
    # Mul expr
    s3 = s1 * s2
    # Pow expr
    s4 = s1**2 * s2**(-1)

    u1 = Unit(s1)
    u2 = Unit(s2)
    u3 = Unit(s3)
    u4 = Unit(s4)

    assert u1.expr == s1
    assert u2.expr == s2
    assert u3.expr == s3
    assert u4.expr == s4

    assert_allclose_units(u1.base_value, pc_mks, 1e-12)
    assert_allclose_units(u2.base_value, yr_mks, 1e-12)
    assert_allclose_units(u3.base_value, pc_mks * yr_mks, 1e-12)
    assert_allclose_units(u4.base_value, pc_mks**2 / yr_mks, 1e-12)

    assert u1.dimensions == length
    assert u2.dimensions == time
    assert u3.dimensions == length * time
    assert u4.dimensions == length**2 / time


def test_create_with_duplicate_dimensions():
    """
    Create units with overlapping dimensions. Ex: km/Mpc.

    """

    u1 = Unit("J * s**-1")
    u2 = Unit("km/s/Mpc")
    km_mks = m_per_km
    Mpc_mks = m_per_mpc

    assert u1.base_value == 1
    assert u1.dimensions == power

    assert_allclose_units(u2.base_value, km_mks / Mpc_mks, 1e-12)
    assert u2.dimensions == rate


def test_create_new_symbol():
    """
    Create unit with unknown symbol.

    """
    u1 = Unit("abc", base_value=42, dimensions=(mass/time))

    assert u1.expr == Symbol("abc", positive=True)
    assert u1.base_value == 42
    assert u1.dimensions == mass / time

    u1 = Unit("abc", base_value=42, dimensions=length**3)

    assert u1.expr == Symbol("abc", positive=True)
    assert u1.base_value == 42
    assert u1.dimensions == length**3

    u1 = Unit("abc", base_value=42, dimensions=length*(mass*length))

    assert u1.expr == Symbol("abc", positive=True)
    assert u1.base_value == 42
    assert u1.dimensions == length**2*mass

    with pytest.raises(UnitParseError):
        Unit('abc', base_value=42, dimensions=length**length)
    with pytest.raises(UnitParseError):
        Unit('abc', base_value=42, dimensions=length**(length*length))
    with pytest.raises(UnitParseError):
        Unit('abc', base_value=42, dimensions=length-mass)
    with pytest.raises(UnitParseError):
        Unit('abc', base_value=42, dimensions=length+mass)


def test_create_fail_on_unknown_symbol():
    """
    Fail to create unit with unknown symbol, without base_value and dimensions.

    """
    with pytest.raises(UnitParseError):
        Unit(Symbol("jigawatts"))


def test_create_fail_on_bad_symbol_type():
    """
    Fail to create unit with bad symbol type.

    """
    with pytest.raises(UnitParseError):
        Unit([1])  # something other than Expr and str


def test_create_fail_on_bad_dimensions_type():
    """
    Fail to create unit with bad dimensions type.

    """
    with pytest.raises(UnitParseError):
        Unit("a", base_value=1, dimensions="(mass)")


def test_create_fail_on_dimensions_content():
    """
    Fail to create unit with bad dimensions expr.

    """
    a = Symbol("a")
    with pytest.raises(UnitParseError):
        Unit("a", base_value=1, dimensions=a)


def test_create_fail_on_base_value_type():
    """
    Fail to create unit with bad base_value type.

    """
    with pytest.raises(UnitParseError):
        Unit("a", base_value="a", dimensions=(mass/time))


def test_string_representation():
    """
    Check unit string representation.

    """
    pc = Unit("pc")
    Myr = Unit("Myr")
    speed = pc / Myr
    dimensionless = Unit()

    assert str(pc) == "pc"
    assert str(Myr) == "Myr"
    assert str(speed) == "pc/Myr"
    assert repr(speed) == "pc/Myr"
    assert str(dimensionless) == "dimensionless"
    assert repr(dimensionless) == "(dimensionless)"


def test_multiplication():
    """
    Multiply two units.

    """
    msun_mks = mass_sun_kg
    pc_mks = m_per_pc

    # Create symbols
    msun_sym = Symbol("Msun", positive=True)
    pc_sym = Symbol("pc", positive=True)
    s_sym = Symbol("s", positive=True)

    # Create units
    u1 = Unit("Msun")
    u2 = Unit("pc")

    # Mul operation
    u3 = u1 * u2

    assert u3.expr == msun_sym * pc_sym
    assert_allclose_units(u3.base_value, msun_mks * pc_mks, 1e-12)
    assert u3.dimensions == mass * length

    # Pow and Mul operations
    u4 = Unit("pc**2")
    u5 = Unit("Msun * s")

    u6 = u4 * u5

    assert u6.expr == pc_sym**2 * msun_sym * s_sym
    assert_allclose_units(u6.base_value, pc_mks**2 * msun_mks, 1e-12)
    assert u6.dimensions == length**2 * mass * time


def test_division():
    """
    Divide two units.

    """
    pc_mks = m_per_pc
    km_mks = m_per_km

    # Create symbols
    pc_sym = Symbol("pc", positive=True)
    km_sym = Symbol("km", positive=True)
    s_sym = Symbol("s", positive=True)

    # Create units
    u1 = Unit("pc")
    u2 = Unit("km * s")

    u3 = u1 / u2

    assert u3.expr == pc_sym / (km_sym * s_sym)
    assert_allclose_units(u3.base_value, pc_mks / km_mks, 1e-12)
    assert u3.dimensions == 1 / time


def test_power():
    """
    Take units to some power.

    """
    from sympy import nsimplify

    pc_mks = m_per_pc
    mK_mks = 1e-3
    u1_dims = mass * length**2 * time**-3 * temperature**4
    u1 = Unit("kg * pc**2 * s**-3 * mK**4")

    u2 = u1**2

    assert u2.dimensions == u1_dims**2
    assert_allclose_units(u2.base_value, (pc_mks**2 * mK_mks**4)**2, 1e-12)

    u3 = u1**(-1.0/3)

    assert u3.dimensions == nsimplify(u1_dims**(-1.0/3))
    assert_allclose_units(u3.base_value, (pc_mks**2 * mK_mks**4)**(-1.0/3),
                          1e-12)


def test_equality():
    """
    Check unit equality with different symbols, but same dimensions and
    base_value.

    """
    u1 = Unit("km * s**-1")
    u2 = Unit("m * ms**-1")

    assert u1 == u2


def test_invalid_operations():
    u1 = Unit('cm')
    u2 = Unit('m')

    with pytest.raises(InvalidUnitOperation):
        u1+u2
    with pytest.raises(InvalidUnitOperation):
        u1 += u2
    with pytest.raises(InvalidUnitOperation):
        1+u1
    with pytest.raises(InvalidUnitOperation):
        u1+1
    with pytest.raises(InvalidUnitOperation):
        u1-u2
    with pytest.raises(InvalidUnitOperation):
        u1 -= u2
    with pytest.raises(InvalidUnitOperation):
        1-u1
    with pytest.raises(InvalidUnitOperation):
        u1-1
    with pytest.raises(InvalidUnitOperation):
        u1 *= u2
    with pytest.raises(InvalidUnitOperation):
        u1*'hello!'
    with pytest.raises(InvalidUnitOperation):
        u1 /= u2
    with pytest.raises(InvalidUnitOperation):
        u1/'hello!'


def test_base_equivalent():
    """
    Check base equivalent of a unit.

    """
    Msun_mks = mass_sun_kg
    Mpc_mks = m_per_mpc

    u1 = Unit("Msun * Mpc**-3")
    u2 = Unit("kg * m**-3")
    u3 = u1.get_base_equivalent()

    assert u2.expr == u3.expr
    assert u2 == u3

    assert_allclose_units(u1.base_value, Msun_mks / Mpc_mks**3, 1e-12)
    assert u2.base_value == 1
    assert u3.base_value == 1

    mass_density = mass / length**3

    assert u1.dimensions == mass_density
    assert u2.dimensions == mass_density
    assert u3.dimensions == mass_density

    assert_allclose_units(u1.get_conversion_factor(u3)[0],
                          Msun_mks / Mpc_mks**3, 1e-12)

    with pytest.raises(UnitConversionError):
        u1.get_conversion_factor(Unit('m'))

    with pytest.raises(UnitConversionError):
        u1.get_conversion_factor(Unit('degF'))


def test_temperature_offsets():
    u1 = Unit('degC')
    u2 = Unit('degF')

    with pytest.raises(InvalidUnitOperation):
        operator.mul(u1, u2)
    with pytest.raises(InvalidUnitOperation):
        operator.truediv(u1, u2)


def test_latex_repr():
    registry = UnitRegistry()

    # create a fake comoving unit
    registry.add('pccm', registry.lut['pc'][0]/(1+2), length,
                 "\\rm{pc}/(1+z)", prefixable=True)

    test_unit = Unit('Mpccm', registry=registry)
    assert_almost_equal(test_unit.base_value, m_per_mpc/3)
    assert_equal(test_unit.latex_repr, r'\rm{Mpc}/(1+z)')

    test_unit = Unit('cm**-3', base_value=1.0, registry=registry)
    assert_equal(test_unit.latex_repr, '\\frac{1}{\\rm{cm}^{3}}')

    test_unit = Unit('m_geom/l_geom**3')
    assert_equal(test_unit.latex_repr, '\\frac{1}{M_\\odot^{2}}')

    test_unit = Unit('1e9*cm')
    assert_equal(test_unit.latex_repr, '1.0 \\times 10^{9}\\ \\rm{cm}')

    test_unit = Unit('1.0*cm')
    assert_equal(test_unit.latex_repr, '\\rm{cm}')


def test_latitude_longitude():
    lat = unit_symbols.lat
    lon = unit_symbols.lon
    deg = unit_symbols.deg
    assert_equal(lat.units.base_offset, 90.0)
    assert_equal((deg*90.0).in_units("lat").value, 0.0)
    assert_equal((deg*180).in_units("lat").value, -90.0)
    assert_equal((lat*0.0).in_units("deg"), deg*90.0)
    assert_equal((lat*-90).in_units("deg"), deg*180)

    assert_equal(lon.units.base_offset, -180.0)
    assert_equal((deg*0.0).in_units("lon").value, -180.0)
    assert_equal((deg*90.0).in_units("lon").value, -90.0)
    assert_equal((deg*180).in_units("lon").value, 0.0)
    assert_equal((deg*360).in_units("lon").value, 180.0)

    assert_equal((lon*-180.0).in_units("deg"), deg*0.0)
    assert_equal((lon*-90.0).in_units("deg"), deg*90.0)
    assert_equal((lon*0.0).in_units("deg"), deg*180.0)
    assert_equal((lon*180.0).in_units("deg"), deg*360)


def test_registry_json():
    reg = UnitRegistry()
    json_reg = reg.to_json()
    unserialized_reg = UnitRegistry.from_json(json_reg)

    assert_equal(reg.lut, unserialized_reg.lut)


def test_creation_from_ytarray():
    u1 = Unit(electrostatic_unit)
    assert_equal(str(u1), 'esu')
    assert_equal(u1, Unit('esu'))
    assert_equal(u1, electrostatic_unit.units)

    u2 = Unit(elementary_charge_cgs)
    assert_equal(str(u2), '4.8032056e-10*esu')
    assert_equal(u2, Unit('4.8032056e-10*esu'))
    assert_equal(u1, elementary_charge_cgs.units)

    assert_allclose((u1/u2).base_value,
                    electrostatic_unit/elementary_charge_cgs)

    with pytest.raises(UnitParseError):
        Unit([1, 2, 3]*elementary_charge_cgs)


def test_list_same_dimensions():
    from unyt import m
    reg = default_unit_registry
    for equiv in reg.list_same_dimensions(m):
        assert Unit(equiv).dimensions is length


def test_decagram():
    dag = Unit('dag')
    g = Unit('g')
    assert dag.get_conversion_factor(g) == (10.0, None)


def test_pickle():
    cm = Unit('cm')
    assert cm == pickle.loads(pickle.dumps(cm))


def test_preserve_offset():
    from unyt import degF, dimensionless

    new_unit = degF*dimensionless

    assert new_unit is not degF
    assert new_unit == degF
    assert new_unit.base_offset == degF.base_offset

    new_unit = degF/dimensionless

    assert new_unit is not degF
    assert new_unit == degF
    assert new_unit.base_offset == degF.base_offset

    with pytest.raises(InvalidUnitOperation):
        dimensionless/degF


def test_code_unit():
    from unyt import UnitRegistry

    ureg = UnitRegistry()
    ureg.add('code_length', 10., length)
    ureg.add('code_magnetic_field', 2.0, magnetic_field_mks)
    u = Unit('code_length', registry=ureg)
    assert u.is_code_unit is True
    assert u.get_base_equivalent() == Unit('m')
    u = Unit('cm')
    assert u.is_code_unit is False

    u = Unit('code_magnetic_field', registry=ureg)
    assert u.get_base_equivalent('mks') == Unit('T')
    assert u.get_base_equivalent('cgs') == Unit('gauss')

    UnitSystem(ureg.unit_system_id, 'code_length', 'kg', 's', registry=ureg)

    u = Unit('cm', registry=ureg)
    ue = u.get_base_equivalent('code')

    assert str(ue) == 'code_length'
    assert ue.base_value == 10
    assert ue.dimensions is length

    class FakeDataset(object):
        unit_registry = ureg

    ds = FakeDataset()

    UnitSystem(ds, 'code_length', 'kg', 's', registry=ureg)

    u = Unit('cm', registry=ureg)
    ue = u.get_base_equivalent(ds)

    assert str(ue) == 'code_length'
    assert ue.base_value == 10
    assert ue.dimensions is length

    with pytest.raises(UnitParseError):
        Unit('code_length')


def test_bad_equivalence():
    from unyt import cm

    with pytest.raises(KeyError):
        cm.has_equivalent('dne')


def test_em_unit_base_equivalent():
    from unyt import A, cm

    with pytest.raises(UnitsNotReducible):
        (A/cm).get_base_equivalent('cgs')


def test_define_unit_error():
    from unyt import define_unit

    with pytest.raises(RuntimeError):
        define_unit('foobar', 'baz')
    with pytest.raises(RuntimeError):
        define_unit('foobar', 12)


def test_symbol_lut_length():
    for v in default_unit_symbol_lut.values():
        assert len(v) == 5
