def test_no_duplicates():
    import unyt.unit_symbols as us
    from unyt._unit_lookup_table import (
        physical_constants,
    )

    symbol_names = {name for name in vars(us) if not name.startswith("_")}
    constant_names = set(physical_constants.keys())

    dups = symbol_names.intersection(constant_names)
    try:
        assert dups == {}
    except AssertionError:
        raise ValueError(
            f"Duplicate names found between unit symbols and constants: {dups}"
        )
