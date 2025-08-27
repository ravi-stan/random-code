Unit Tests with Pytest Fixture Plugins (Clean, Fast, Hermetic)
TL;DR
Keep each concern in its own fixture module (e.g., redis_unit.py, llm_unit.py, statefun_unit.py).
Expose them as local pytest plugins via pytest_plugins so tests don’t import fixtures directly.
Put the plugin loader in top‑level tests/conftest.py, not nested (future‑proof and no deprecation warnings).
Default to unit‑only collection with testpaths = tests/unit for fast runs; add integration tests elsewhere later.
References:
Pytest fixtures & good practices: https://docs.pytest.org/en/stable/how-to/fixtures.html
Fakeredis docs: https://fakeredis.readthedocs.io/
Folder layout (unit tests only)
pytest.ini
tests/
  conftest.py                 # top-level: loads local fixture “plugins”
  unit/
    fixtures/
      __init__.py
      redis_unit.py           # Redis/fakeredis fixtures (async)
      llm_unit.py             # LLM stubs & monkeypatch helpers
      statefun_unit.py        # (optional) StateFun harness for unit logic
    test_cache.py             # sample unit test using redis fixture
    test_llm.py               # sample unit test using llm fixture
Why this layout?
tests/conftest.py is the tests root conftest, so loading plugins from here is supported and future‑proof.
Fixture modules under tests/unit/fixtures/ stay nicely separated by concern.
With testpaths = tests/unit you get fast default runs; integration suites can live elsewhere and be invoked explicitly.
Setup
Install dev dependencies
pip install -U pytest pytest-asyncio redis fakeredis
If you’ll use the optional StateFun harness, you only need it for your app code; the harness uses no extra packages.
pytest.ini
[pytest]
testpaths = 
  tests/unit
addopts = -q
asyncio_mode = auto
Top‑level plugin loader — tests/conftest.py
Keep this light: just point pytest at your local fixture modules.
# tests/conftest.py
pytest_plugins = [
    "tests.unit.fixtures.redis_unit",
    "tests.unit.fixtures.llm_unit",
    "tests.unit.fixtures.statefun_unit",  # optional; safe to include even if unused
]
Loading pytest_plugins in a non‑top‑level conftest is discouraged. Keeping it here avoids warnings and surprises.
