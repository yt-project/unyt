"""
parsing utilities



"""


import token

from sympy import Basic, Float, Integer, Rational, Symbol, sqrt
from sympy.parsing.sympy_parser import auto_number, parse_expr, rationalize

from unyt._unit_lookup_table import inv_name_alternatives
from unyt.exceptions import UnitParseError


def _auto_positive_symbol(tokens, local_dict, global_dict):
    """
    Inserts calls to ``Symbol`` for undefined variables.
    Passes in positive=True as a keyword argument.
    Adapted from sympy.sympy.parsing.sympy_parser.auto_symbol
    """
    result = []

    tokens.append((None, None))  # so zip traverses all tokens
    for tok, nextTok in zip(tokens, tokens[1:]):
        tokNum, tokVal = tok
        nextTokNum, nextTokVal = nextTok
        if tokNum == token.NAME:
            name = tokVal
            if name in global_dict:
                obj = global_dict[name]
                if isinstance(obj, (Basic, type)) or callable(obj):
                    result.append((token.NAME, name))
                    continue

            # try to resolve known alternative unit name
            try:
                used_name = inv_name_alternatives[str(name)]
            except KeyError:
                # if we don't know this name it's a user-defined unit name
                # so we should create a new symbol for it
                used_name = str(name)

            result.extend(
                [
                    (token.NAME, "Symbol"),
                    (token.OP, "("),
                    (token.NAME, repr(used_name)),
                    (token.OP, ","),
                    (token.NAME, "positive"),
                    (token.OP, "="),
                    (token.NAME, "True"),
                    (token.OP, ")"),
                ]
            )
        else:
            result.append((tokNum, tokVal))

    return result


global_dict = {
    "Symbol": Symbol,
    "Integer": Integer,
    "Float": Float,
    "Rational": Rational,
    "sqrt": sqrt,
}

unit_text_transform = (_auto_positive_symbol, auto_number, rationalize)


def parse_unyt_expr(unit_expr):
    if not unit_expr:
        # Bug catch...
        # if unit_expr is an empty string, parse_expr fails hard...
        unit_expr = "1"
    # Avoid a parse error if someone uses the percent unit and the
    # parser tries to interpret it as the modulo operator
    unit_expr = unit_expr.replace("%", "percent")
    unit_expr = unit_expr.replace("°", "deg")
    unit_expr = unit_expr.replace("$", "dollars")
    unit_expr = unit_expr.replace("¢", "cents")
    unit_expr = unit_expr.replace("\u20ac", "euros")
    unit_expr = unit_expr.replace("\u00a3", "pounds")
    unit_expr = unit_expr.replace("\u00a5", "yen")
    try:
        unit_expr = parse_expr(
            unit_expr, global_dict=global_dict, transformations=unit_text_transform
        )
    except Exception as e:
        msg = "Unit expression '{}' raised an error during parsing:\n{}".format(
            unit_expr,
            repr(e),
        )
        raise UnitParseError(msg)
    return unit_expr
