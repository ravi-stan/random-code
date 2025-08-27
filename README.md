# Unit Tests with Pytest Fixture Plugins (Clean, Fast, Hermetic)

## TL;DR

- Keep **each concern in its own fixture module** (e.g., `redis_unit.py`, `llm_unit.py`, `statefun_unit.py`).
- Expose them as **local pytest plugins** via `pytest_plugins` so tests don’t import fixtures directly.
- Put the plugin loader in **top‑level `tests/conftest.py`**, not nested (future‑proof and avoids the warning about non‑top‑level `pytest_plugins`).
- Default to **unit‑only collection** with `testpaths = tests/unit` for fast runs; add integration tests elsewhere later.

**References**  
- Pytest fixtures & good practices: https://docs.pytest.org/en/stable/how-to/fixtures.html  
- Fakeredis: https://fakeredis.readthedocs.io/

---

## Folder layout (unit tests only)

```
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
```

### Why this layout?

- **`tests/conftest.py`** is the *tests root* conftest, so loading plugins from here is supported and future‑proof.  
- Fixture modules under `tests/unit/fixtures/` stay **nicely separated by concern**.  
- With `testpaths = tests/unit` you get **fast default runs**; integration suites can live elsewhere and be invoked explicitly.

---

## Setup

Install dev dependencies:

```bash
pip install -U pytest pytest-asyncio redis fakeredis
```

> If you’ll use the optional StateFun harness, you only need it for your app code; the harness itself uses no extra packages.

---

## `pytest.ini`

```ini
[pytest]
testpaths = 
  tests/unit
addopts = -q
asyncio_mode = auto
```

---

## Top‑level plugin loader — `tests/conftest.py`

Keep this light: just point pytest at your local fixture modules.

```python
# tests/conftest.py
pytest_plugins = [
    "tests.unit.fixtures.redis_unit",
    "tests.unit.fixtures.llm_unit",
    "tests.unit.fixtures.statefun_unit",  # optional; safe to include even if unused
]
```

> Loading `pytest_plugins` in a **non‑top‑level** conftest is discouraged. Keeping it here avoids warnings and surprises.

---

## Redis fixtures — `tests/unit/fixtures/redis_unit.py`

**Goal:** a fast, hermetic Redis for unit tests using **fakeredis’ async API**; per‑test cleanup; **binary‑safe** (bytes).  

```python
# tests/unit/fixtures/redis_unit.py
from __future__ import annotations
import uuid
import pytest
from fakeredis.aioredis import FakeRedis, FakeServer  # async FakeRedis
from redis.asyncio import Redis

@pytest.fixture(scope="session")
def _fake_redis_server() -> FakeServer:
    """One in-memory Redis server shared by all tests (via FakeServer)."""
    return FakeServer()

@pytest.fixture
async def redis_client(_fake_redis_server) -> Redis:
    """
    Async Redis client backed by fakeredis, flushed before and after each test.
    Uses binary-safe (decode_responses=False) to match many prod setups.
    """
    client: Redis = FakeRedis(server=_fake_redis_server, decode_responses=False)
    await client.flushdb()
    try:
        yield client
    finally:
        await client.flushdb()
        await client.close()

@pytest.fixture
def redis_namespace() -> str:
    """Unique namespace prefix per test to keep keys isolated."""
    return f"test:{uuid.uuid4().hex}"

# Convenience helper you can use in tests
def _ns(ns: str, *parts: str) -> str:
    return ns + ":" + ":".join(parts)

@pytest.fixture
def redis_key():
    """Build namespaced keys: redis_key(ns, 'user', '42') -> 'ns:user:42'."""
    return _ns
```

> If you already have a cache wrapper (e.g., `app.cache.Cache`), add a `cache` fixture here that constructs it from `redis_client + redis_namespace`.

---

## LLM fixtures — `tests/unit/fixtures/llm_unit.py`

**Goal:** drop‑in fakes for embeddings + chat completions that you can monkeypatch into your code paths.

```python
# tests/unit/fixtures/llm_unit.py
from __future__ import annotations
import pytest

# Minimal stubs that mimic the Azure OpenAI client surface we use.
class _EmbData: 
    def __init__(self, vec): self.embedding = vec
class _EmbResp: 
    def __init__(self, vec): self.data = [_EmbData(vec)]
class _ChatMsg: 
    def __init__(self, content): self.content = content
class _Choice: 
    def __init__(self, content): self.message = _ChatMsg(content)
class _ChatResp: 
    def __init__(self, content): self.choices = [_Choice(content)]

class LLMStub:
    """Always-success LLM stub."""
    def __init__(self, answer="stubbed answer", emb_dim=8):
        self._answer = answer
        self._emb = [0.0] * emb_dim
        class _Emb:
            def __init__(self, outer): self._o = outer
            def create(self, model, input): return _EmbResp(self._o._emb)
        class _ChatComps:
            def __init__(self, outer): self._o = outer
            def create(self, model, messages, temperature=0.2):
                return _ChatResp(self._o._answer)
        class _Chat:
            def __init__(self, outer): self._o = outer
            @property
            def completions(self): return _ChatComps(self._o)
        self.embeddings = _Emb(self)
        self.chat = _Chat(self)

class RateLimitOnceStub(LLMStub):
    """Raises once (to test retry code), then succeeds."""
    def __init__(self, answer="recovered after 429"):
        super().__init__(answer)
        self._first = True
        class _ChatComps:
            def __init__(self, outer): self._o = outer
            def create(self, model, messages, temperature=0.2):
                if self._o._first:
                    self._o._first = False
                    raise RuntimeError("429 too many requests (simulated)")
                return _ChatResp(self._o._answer)
        class _Chat:
            def __init__(self, outer): self._o = outer
            @property
            def completions(self): return _ChatComps(self._o)
        self.chat = _Chat(self)

@pytest.fixture
def llm_success():
    """Yield an LLM stub object that tests can inject/monkeypatch."""
    return LLMStub(answer="This is a concise answer.")

@pytest.fixture
def llm_rate_limit_once():
    """Yield an LLM stub that fails first call, then succeeds."""
    return RateLimitOnceStub()
```

> You can **inject** these with `monkeypatch` inside tests (e.g., `app.functions.aoai = llm_success`) or provide a convenience fixture that performs the monkeypatch for you.

---

## Optional: StateFun harness — `tests/unit/fixtures/statefun_unit.py`

**Goal:** unit‑test StateFun‑style functions (pure Python callables) without HTTP or the runtime.  
This “harness” gives you a minimal `Context`/`Message` and captures `ctx.send(...)`/`ctx.send_egress(...)`.

```python
# tests/unit/fixtures/statefun_unit.py
from __future__ import annotations
import pytest
from typing import Any, Callable, Dict, Tuple, List

class _Storage(dict): pass

class Ctx:
    def __init__(self, typename: str, id: str):
        self.address = type("A", (), {"typename": typename, "id": id})
        self.storage = _Storage()
        self._outbox: List[dict] = []
        self._egress: List[dict] = []
        self.caller = None
    def send(self, msg): self._outbox.append(msg)
    def send_egress(self, msg): self._egress.append(msg)
    @property
    def outbox(self): return list(self._outbox)
    @property
    def egress(self): return list(self._egress)

class Msg:
    def __init__(self, value, value_type=None):
        self._v = value; self._t = value_type
    def is_type(self, t): return t == self._t
    def as_type(self, t):
        if not self.is_type(t): raise TypeError("bad type")
        return self._v

class StateFunHarness:
    def __init__(self, registry: Dict[str, Callable], value_type: Any):
        self.registry = registry; self.value_type = value_type
        self._ctxs: Dict[Tuple[str, str], Ctx] = {}
    def ctx(self, typename, id):
        self._ctxs.setdefault((typename,id), Ctx(typename,id))
        return self._ctxs[(typename,id)]
    async def invoke(self, typename, id, value):
        fn = self.registry[typename]
        ctx = self.ctx(typename, id)
        await fn(ctx, Msg(value, self.value_type))
        return ctx

@pytest.fixture
def statefun_harness():
    """Factory to build a harness when needed in a test."""
    def _make(registry: Dict[str, Callable], value_type: Any) -> StateFunHarness:
        return StateFunHarness(registry, value_type)
    return _make
```

---

## Sample unit tests

### `tests/unit/test_cache.py` — using Redis fixture

```python
import pytest

@pytest.mark.asyncio
async def test_redis_roundtrip(redis_client, redis_namespace, redis_key):
    key = redis_key(redis_namespace, "user", "42")
    await redis_client.set(key, b"hello", ex=5)
    got = await redis_client.get(key)
    assert got == b"hello"
```
### `tests/unit/test_llm.py` — using LLM fixture + monkeypatch

```python
import pytest

@pytest.mark.asyncio
async def test_llm_success_monkeypatch(monkeypatch, llm_success):
    # Example: patch a module-level client your code uses (e.g., app.functions.aoai)
    import app.functions as f
    monkeypatch.setattr(f, "aoai", llm_success, raising=True)

    # Simulate your code path using the client
    resp = f.aoai.chat.completions.create(model="x", messages=[{"role":"user","content":"hi"}])
    assert resp.choices[0].message.content.startswith("This is a concise answer.")

@pytest.mark.asyncio
async def test_llm_rate_limit_then_recover(monkeypatch, llm_rate_limit_once):
    import app.functions as f
    monkeypatch.setattr(f, "aoai", llm_rate_limit_once, raising=True)
    # First call raises
    with pytest.raises(RuntimeError):
        f.aoai.chat.completions.create(model="x", messages=[])
    # Second call succeeds
    resp = f.aoai.chat.completions.create(model="x", messages=[])
    assert "recovered" in resp.choices[0].message.content
```

---

## Running tests

```bash
pytest           # runs only tests/unit/** (fast unit pass)
pytest -q -k redis  # run only tests matching "redis"
```

---

## Common Q&A

**Why binary‑safe (bytes) for Redis?**  
Many production clients use `decode_responses=False`. It prevents surprises when you later store compressed or non‑UTF‑8 payloads.

**Where should I put integration tests?**  
Outside `tests/unit/`—e.g., `tests/integration/` with its own `conftest.py` and heavier fixtures (containers, real services). Run them explicitly: `pytest tests/integration`.

**My app constructs Redis clients internally; how do tests see the same fake server?**  
Centralize client creation in your app (e.g., `get_redis_client()`), then patch that in tests. Or patch `Redis.from_url` / `RedisCluster.from_url` to return `fakeredis` pointing to the same `FakeServer`.

---

## Tips & Extensions

- Add a `cache` fixture that instantiates your `app.cache.Cache` with `redis_client` and `redis_namespace`.  
- If you use the **sync** Redis API, create a sibling `redis_sync_unit.py` using `fakeredis.FakeRedis` (sync) instead of `fakeredis.aioredis.FakeRedis`.  
- For **Redis Cluster** code paths, most unit tests can use a simple proxy that forwards to `FakeRedis`. Keep real cluster semantics for a thin integration suite.

---

## Troubleshooting

- **“Fixture not found”** → ensure the plugin module path matches in `tests/conftest.py`.  
- **Import order issues** → prefer monkeypatching factory methods (`from_url`) over replacing classes; it’s less sensitive to early imports.  
- **Async errors** → include `pytest-asyncio` and set `asyncio_mode = auto` (as above).
