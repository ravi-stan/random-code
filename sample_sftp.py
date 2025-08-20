from typing import Mapping, TypeVar, Any

T = TypeVar("T")
_MISSING = object()

def get_value_by_prefix(
    data: Mapping[str, T],
    prefix: str,
    *,
    case_sensitive: bool = True,
    default: Any = _MISSING,
    require_unique: bool = False,
) -> T:
    """
    Return the value for the first key in `data` that starts with `prefix`.

    - case_sensitive: match respecting case (default True).
    - default: value to return if no match; if not provided, KeyError is raised.
    - require_unique: if True and multiple keys match, KeyError is raised.

    Complexity: O(n) over number of keys.
    """
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a str")

    norm = (lambda s: s) if case_sensitive else (lambda s: s.lower())
    p = norm(prefix)

    matches = [(k, v) for k, v in data.items() if isinstance(k, str) and norm(k).startswith(p)]

    if require_unique and len(matches) > 1:
        raise KeyError(f"Multiple keys start with {prefix!r}: {[k for k, _ in matches]}")

    if matches:
        return matches[0][1]

    if default is not _MISSING:
        return default

    raise KeyError(f"No key starts with {prefix!r}")

