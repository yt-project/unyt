import warnings
from typing import Optional


def warn_deprecated(
    name,
    /,
    *,
    stacklevel: int = 3,
    replacement: Optional[str] = None,
    since_version: str,
) -> None:
    msg = (
        f"{name} is deprecated and will be removed in a future version\n"
        f"Instead, {replacement}\n"
        f"(deprecated since unyt v{since_version})"
    )
    warnings.warn(
        msg,
        category=DeprecationWarning,
        stacklevel=stacklevel,
    )
