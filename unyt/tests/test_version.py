from importlib.metadata import version

import unyt


def test_version():
    expected = version("unyt")
    assert unyt.__version__ == expected
