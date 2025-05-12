_DUMMY_PAT = re.compile(r"(?:^|[^A-Za-z0-9])(?:dummy|test|placeholder)(?:[^A-Za-z0-9]|$)", re.I)

def _is_dummy(value: str | None) -> bool:
    """
    Return True when *value* is None **or** its text contains any of the
    reserved dummy keywords (“dummy”, “test”, “placeholder”), ignoring case.

    Examples considered *dummy*::
        None, "", "dummy", "test-db", "orders_placeholder_2025"

    Examples considered *real*::
        "production", "attestation", "contest"
    """
    if value is None:
        return True
    return bool(_DUMMY_PAT.search(value.strip()))
