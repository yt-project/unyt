from docutils.parsers.rst import Directive

try:
    maketrans = "".maketrans
except AttributeError:
    # python2 fallback
    from string import maketrans

import numpy as np
from sympy.core.mul import Mul
from sympy.core.symbol import Symbol

import unyt
import unyt.dimensions as dims
from unyt import Unit
from unyt._unit_lookup_table import name_alternatives, physical_constants
from unyt.exceptions import UnitsNotReducible
from unyt.unit_registry import default_unit_registry
from unyt.unit_systems import _split_prefix

all_dims = {}
for d in dims.__dict__.keys():
    if isinstance(getattr(dims, d), (Symbol, Mul)):
        if d == "dimensionless":
            continue
        all_dims[getattr(dims, d)] = d


def setup(app):
    app.add_directive("show_all_units", ShowAllUnits)
    app.add_directive("show_all_constants", ShowAllConstants)
    setup.app = app
    setup.config = app.config
    setup.confdir = app.confdir

    retdict = dict(version="0.1")

    return retdict


class ShowAllUnits(Directive):
    required_arguments = 0
    optional_arguments = 0

    def run(self):
        lines = []

        for name, alt_names in name_alternatives.items():
            prefix, base = _split_prefix(name, default_unit_registry.lut)
            if prefix != "":
                continue
            lut_entry = default_unit_registry[name]
            u = Unit(name)
            try:
                dimensions = all_dims[u.dimensions]
            except KeyError:
                if u.is_dimensionless:
                    dimensions = "dimensionless"
                else:
                    dimensions = u.dimensions
            try:
                mks_value = (1 * u).in_mks()
            except UnitsNotReducible:
                mks_value = "N/A"
            try:
                cgs_value = (1 * u).in_cgs()
            except UnitsNotReducible:
                cgs_value = "N/A"

            def generate_table_value(value):
                if value == "N/A":
                    return value
                approx_string = "{:.4e}".format(value)
                real_string = str(value)
                fv = value.value
                close_value = float("{:.4e}".format(fv))

                if (close_value - fv) / fv < 1e-6 and len(str(fv)) > 8:
                    return approx_string

                if fv < 1e-4 or fv > 1e4 or len(str(fv)) > 8:
                    return approx_string

                return real_string

            latex_repr = "``" + u.latex_repr + "``"
            if latex_repr == "````":
                latex_repr = ""
            with np.printoptions(precision=4, suppress=False, floatmode="maxprec"):
                lines.append(
                    (
                        name,
                        str(dimensions),
                        generate_table_value(mks_value),
                        generate_table_value(cgs_value),
                        latex_repr,
                        str(lut_entry[4]),
                        ", ".join([a for a in alt_names if a != name]),
                    )
                )
        lines.insert(
            0,
            [
                "Unit Name",
                "Dimensions",
                "MKS value",
                "CGS Value",
                "LaTeX Representation",
                "SI Prefixable?",
                "Alternate Names",
            ],
        )
        lines = as_rest_table(lines, full=False).split("\n")
        rst_file = self.state_machine.document.attributes["source"]
        self.state_machine.insert_input(lines, rst_file)
        return []


class ShowAllConstants(Directive):
    required_arguments = 0
    optional_arguments = 0

    def run(self):
        lines = []

        for name, (_value, _unit, alternate_names) in physical_constants.items():
            val = getattr(unyt.physical_constants, name)
            if val > 1e4 or val < 1e-4:
                default_value = "{:.4e}".format(val)
            else:
                default_value = str(val)
            lines.append((name, default_value, ", ".join(alternate_names)))

        lines.insert(0, ["Constant Name", "Value", "Alternate Names"])
        lines = as_rest_table(lines, full=False).split("\n")
        rst_file = self.state_machine.document.attributes["source"]
        self.state_machine.insert_input(lines, rst_file)
        return []


def as_rest_table(data, full=False):
    """
    Originally from ActiveState recipes, copy/pasted from GitHub
    where it is listed with an MIT license.

    https://github.com/ActiveState/code/tree/master/recipes/Python/579054_Generate_Sphinx_table

    """
    data = data if data else [["No Data"]]
    table = []
    # max size of each column
    sizes = list(map(max, zip(*[[len(str(elt)) for elt in member] for member in data])))
    num_elts = len(sizes)

    if full:
        start_of_line = "| "
        vertical_separator = " | "
        end_of_line = " |"
        line_marker = "-"
    else:
        start_of_line = ""
        vertical_separator = "  "
        end_of_line = ""
        line_marker = "="

    meta_template = vertical_separator.join(
        ["{{{{{0}:{{{0}}}}}}}".format(i) for i in range(num_elts)]
    )
    template = "{0}{1}{2}".format(
        start_of_line, meta_template.format(*sizes), end_of_line
    )
    # determine top/bottom borders
    if full:
        to_separator = maketrans("| ", "+-")
    else:
        to_separator = maketrans("|", "+")
    start_of_line = start_of_line.translate(to_separator)
    vertical_separator = vertical_separator.translate(to_separator)
    end_of_line = end_of_line.translate(to_separator)
    separator = "{0}{1}{2}".format(
        start_of_line,
        vertical_separator.join([x * line_marker for x in sizes]),
        end_of_line,
    )
    # determine header separator
    th_separator_tr = maketrans("-", "=")
    start_of_line = start_of_line.translate(th_separator_tr)
    line_marker = line_marker.translate(th_separator_tr)
    vertical_separator = vertical_separator.translate(th_separator_tr)
    end_of_line = end_of_line.translate(th_separator_tr)
    th_separator = "{0}{1}{2}".format(
        start_of_line,
        vertical_separator.join([x * line_marker for x in sizes]),
        end_of_line,
    )
    # prepare result
    table.append(separator)
    # set table header
    titles = data[0]
    table.append(template.format(*titles))
    table.append(th_separator)

    for d in data[1:-1]:
        table.append(template.format(*d))
        if full:
            table.append(separator)
    table.append(template.format(*data[-1]))
    table.append(separator)

    return "\n".join(table)
