# Unit Test Fixtures (Global Registration)

This repo includes **prebuilt pytest fixtures** that mock external services so you can write **fast, deterministic, no‑network** unit tests.

> **Scope:** All fixtures are **globally registered** via `tests/unit/conftest.py`. You don’t need to import or register anything per test file.

---

## What’s here

| Area | Fixture module | What it mocks | Example tests |
|---|---|---|---|
| Google BigQuery | `tests/unit/fixtures/google_bigquery_unit.py` | BigQuery client interactions (queries, datasets, tables) | `tests/unit/sample/test_google_bigquery.py` |
| Google Dialogflow | `tests/unit/fixtures/google_dialogflow_unit.py` | Dialogflow sessions/intents flows | `tests/unit/sample/test_google_dialogflow.py` |
| Redis | `tests/unit/fixtures/redis_unit.py` | Redis client operations and behavior | `tests/unit/sample/test_redis.py` |
| Vertex AI (LLM calls) | `tests/unit/fixtures/vertex_ai_llm_unit.py` | Vertex AI text/LLM call surfaces | `tests/unit/sample/test_vertex_ai_llm.py` |
| MCP server (multi) | `tests/unit/fixtures/mcp_multi_unit.py` | Multi‑tool MCP server interactions | `tests/unit/sample/test_mcp_multi.py` |
| MCP tool mocks | `tests/unit/fixtures/mcp_tool_factory/` | Helper code to **mock/construct MCP tools** used by the MCP fixtures | Used by `mcp_multi_unit.py` and `test_mcp_multi.py` |

> 💡 **Tip:** The `tests/unit/sample/` directory is the canonical reference for how to use each fixture in real tests.

---

## Quick start

1. **Install test deps** (pytest, etc.).
2. **Run sample tests** to verify setup:
   ```bash
   pytest -q tests/unit/sample
   ```

If that passes, you’re ready to use the fixtures in your own tests.

---

## Global fixture registration (already enabled)

All fixtures are made available to your tests by declaring them as pytest plugins in `tests/unit/conftest.py`:

```python
# tests/unit/conftest.py
pytest_plugins = [
    "tests.unit.fixtures.google_bigquery_unit",
    "tests.unit.fixtures.google_dialogflow_unit",
    "tests.unit.fixtures.redis_unit",
    "tests.unit.fixtures.vertex_ai_llm_unit",
    "tests.unit.fixtures.mcp_multi_unit",
]
```

> When you add a new fixture module under `tests/unit/fixtures/`, **append it to this list** so it’s available suite‑wide.

**Verify the plugins are loaded:**

```bash
pytest --fixtures -q | grep -Ei "bigquery|dialogflow|redis|vertex|mcp" || true
```

---

## Using a fixture in a test

1. **Reference fixtures by name** as parameters to your test function.
2. **Arrange–Act–Assert** using the provided mocks/stubs.
3. See the **sample tests** for the exact fixture names and typical flows.

Minimal pattern:

```python
def test_my_behavior(dep_fixture_one, dep_fixture_two):
    # Arrange using the provided fixtures
    # Act by calling your code under test
    # Assert expected outcomes (no real network calls)
```

Open the related samples to see the real fixture names and usage:

- `tests/unit/sample/test_google_bigquery.py`
- `tests/unit/sample/test_google_dialogflow.py`
- `tests/unit/sample/test_redis.py`
- `tests/unit/sample/test_vertex_ai_llm.py`
- `tests/unit/sample/test_mcp_multi.py`

---

## Directory layout

```
tests/
└─ unit/
   ├─ fixtures/
   │  ├─ google_bigquery_unit.py
   │  ├─ google_dialogflow_unit.py
   │  ├─ redis_unit.py
   │  ├─ vertex_ai_llm_unit.py
   │  ├─ mcp_multi_unit.py
   │  └─ mcp_tool_factory/
   │     └─ ... (helpers to mock MCP tools)
   └─ sample/
      ├─ test_google_bigquery.py
      ├─ test_google_dialogflow.py
      ├─ test_mcp_multi.py
      ├─ test_redis.py
      └─ test_vertex_ai_llm.py
```

---

## Best practices

- **Keep unit tests hermetic.** These fixtures avoid real network calls; don’t replace them with live clients.
- **One behavior per test.** Use only the fixtures you need.
- **Read the sample tests first.** Mirror the patterns (fixture names, scopes, return values).
- **Extend locally when needed.** You can wrap a provided fixture with your own fixture that depends on it to inject test‑specific data.
- **Document customizations.** If you extend/parametrize, add a short note in your test module.

---

## Troubleshooting

- **`fixture 'XYZ' not found`**
  - Confirm the fixture name from `pytest --fixtures -q`.
  - Ensure the module exporting the fixture is listed in `pytest_plugins` (see above).
- **Import path issues**
  - Run pytest from the **repo root** (so `tests/` is under the rootdir).
  - If needed, set `PYTHONPATH=.` when running pytest.
- **Unexpected live network calls**
  - Check you didn’t import a real SDK client in your code path.
  - Verify you enabled the correct fixture module and are using the expected mocked client object (see samples).

---

## Adding or updating fixtures

1. Put new fixtures under `tests/unit/fixtures/` (or a subfolder, e.g., `mcp_tool_factory/` for helpers).
2. Keep behavior **deterministic** and **side‑effect free**.
3. Add or update a **sample test** in `tests/unit/sample/` that demonstrates usage.
4. Update the **table at the top** of this README.
5. Append the new module to `pytest_plugins` in `tests/unit/conftest.py`.

---

## FAQ

**Q: Where do I find the exact fixture names?**  
A: Run `pytest --fixtures -q` and check the related sample test under `tests/unit/sample/`.

**Q: Do these fixtures require cloud credentials?**  
A: No. They are built to run offline and should not rely on environment credentials.

**Q: Can I extend a fixture for my test?**  
A: Yes—create a small wrapper fixture in your test module that depends on the base fixture(s), or use parametrization.

---

**Maintainers:** If you change a fixture’s interface or behavior, update the corresponding sample test and this README to keep the suite discoverable.
