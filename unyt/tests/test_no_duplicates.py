def test_no_duplicates():
    import unyt
    from unyt import physical_constants, unit_symbols
    from unyt.array import unyt_quantity

    symbol_names = {key for key in unit_symbols.__dict__ if not key.startswith("_")}
    constant_names = {
        key for key in physical_constants.__dict__ if not key.startswith("_")
    }

    dups = symbol_names.intersection(constant_names)
    for dup in dups:
        assert isinstance(getattr(unyt, dup), unyt_quantity)
