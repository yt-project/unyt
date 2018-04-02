# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = -W
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = unyt
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
html:
	SPHINX_APIDOC_OPTIONS=members,undoc-members,show-inheritance sphinx-apidoc -M -e -d 0 -o modules ../unyt ../unyt/tests ../unyt/testing.py ../unyt/on_demand_imports.py ../unyt/physical_ratios.py ../unyt/pint_conversions.py ../unyt/unit_lookup_table.py
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	-rm -rf $(BUILDDIR)/*
	-rm -rf modules/
