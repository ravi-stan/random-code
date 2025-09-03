# MCP Tool Harness (Lightweight Registry + Tool Factory + Backend)

A compact, production‑style **test harness** for apps that use
`langchain_mcp_adapters.client.MultiServerMCPClient`.

**What you get**
- One **wrapper** around `MultiServerMCPClient` → a single, stable patch point.
- A **declarative endpoint registry** (server → endpoints).
- A **tool factory** that turns registry rows into real LangChain `BaseTool`/`StructuredTool`s.
- A pluggable **Backend**: fast in‑memory for tests; swap to HTTP/DB later.
- Minimal **pytest** fixture + **tiny tests**. No real MCP servers required.

> Call this “**tool harness**” or “**scaffolding**.” It’s small, predictable, and grows with your project.

---

## Quick Start

```bash
pip install -U langchain-core pydantic pytest pytest-asyncio
```

Suggested layout:
```
yourpkg/
  mcp_client.py          # wrapper: re-export MultiServerMCPClient (single patch point)
  tooling.py             # tool factory + backend protocol(s)
  toolspecs.py           # endpoint registry (declarative)
  your_module.py         # SUT that asks client for tools/resources
tests/
  conftest.py            # mocked client fixture using InMemoryBackend
  test_tools_smoke.py    # tiny smoke tests
```

---

## 1) Wrapper: single patch point

```python
# yourpkg/mcp_client.py
from langchain_mcp_adapters.client import MultiServerMCPClient  # re-export
```
Use it everywhere in your app:
```python
from yourpkg.mcp_client import MultiServerMCPClient
```
In tests, always patch: `"yourpkg.mcp_client.MultiServerMCPClient"`.

---

## 2) Tool factory + Backend

```python
# yourpkg/tooling.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Type
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

# ---- Args schemas ----
class IdArgs(BaseModel):
    id: str = Field(..., description="Resource identifier")

class IdWithIncludeArgs(BaseModel):
    id: str = Field(..., description="Resource identifier")
    include: List[str] | None = Field(None, description="Optional fields to include")

# ---- Endpoint spec ----
@dataclass(frozen=True)
class EndpointSpec:
    name: str
    description: str
    args_model: Type[BaseModel] = IdArgs
    response_model: Optional[Type[BaseModel]] = None  # optional output validation
    tags: tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"

# ---- Backend protocol + impls ----
class Backend(Protocol):
    async def query(self, server: str, endpoint: str, args: BaseModel) -> Any: ...

class InMemoryBackend:
    # data[server][endpoint][id] -> JSON payload
    def __init__(self, data: Dict[str, Dict[str, Dict[str, Any]]]):
        self.data = data
    async def query(self, server: str, endpoint: str, args: BaseModel) -> Any:
        _id = args.model_dump().get("id")
        return self.data[server][endpoint][_id]

# ---- Tool builders ----
def mk_endpoint_tool(server: str, spec: EndpointSpec, backend: Backend) -> StructuredTool:
    async def run(**kwargs):
        args = spec.args_model(**kwargs)  # validate inputs
        result = await backend.query(server, spec.name, args)
        if spec.response_model is not None:
            try:  # pydantic v2
                result = spec.response_model.model_validate(result).model_dump()
            except AttributeError:  # pydantic v1
                result = spec.response_model.parse_obj(result).dict()
        return result

    tool_name = f"{server}:{spec.name}"
    tags = [*spec.tags, f"server:{server}", "mcp"]
    metadata = {**spec.metadata, "server": server, "endpoint": spec.name, "version": spec.version}

    return StructuredTool.from_function(
        coroutine=run,
        name=tool_name,
        description=spec.description,
        args_schema=spec.args_model,
        tags=tags,
        metadata=metadata,
        return_direct=False,  # keep tools composable
        # response_format="content",  # enable only if your LC version supports it
    )

def build_tools_for_server(server: str, specs: List[EndpointSpec], backend: Backend):
    return [mk_endpoint_tool(server, s, backend) for s in specs]
```

Why this works:
- Tools are real `BaseTool`s (via `StructuredTool`) → agents, tracing, and `invoke/ainvoke` behave like prod.
- Pydantic models give stable contracts for both inputs and (optionally) outputs.
- Backends abstract I/O; tests run fast with `InMemoryBackend`.

---

## 3) Endpoint registry (declarative)

```python
# yourpkg/toolspecs.py
from __future__ import annotations
from typing import Dict, List
from pydantic import BaseModel
from .tooling import EndpointSpec, IdArgs, IdWithIncludeArgs

# Optional response models
class User(BaseModel):
    id: str
    name: str
    email: str

class Order(BaseModel):
    id: str
    total: float

REGISTRY: Dict[str, List[EndpointSpec]] = {
    "users": [
        EndpointSpec(
            name="get_user",
            description="Get user by id.",
            args_model=IdArgs,
            response_model=User,
            version="1.2.0",
            tags=("read",),
        ),
        EndpointSpec(
            name="get_user_meta",
            description="Get user metadata.",
            args_model=IdWithIncludeArgs,
        ),
    ],
    "orders": [
        EndpointSpec(
            name="get_order",
            description="Get order by id.",
            args_model=IdArgs,
            response_model=Order,
            version="1.1.0",
            tags=("read",),
        ),
    ],
}
```

Add endpoints by appending `EndpointSpec` rows; seed test data with matching IDs.

---

## 4) Example SUT (uses the wrapper)

```python
# yourpkg/your_module.py
from yourpkg.mcp_client import MultiServerMCPClient

CONNECTIONS = {
    "users":  {"url": "http://localhost:8000/mcp", "transport": "streamable_http"},
    "orders": {"command": "python", "args": ["./orders_server.py"], "transport": "stdio"},
}

async def load_all_tools():
    client = MultiServerMCPClient(CONNECTIONS)
    try:
        return await client.get_tools()
    finally:
        await client.close()
```

---

## 5) Pytest fixture (mocked client via `monkeypatch`)

```python
# tests/conftest.py
from __future__ import annotations
import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from yourpkg.tooling import InMemoryBackend, build_tools_for_server
from yourpkg.toolspecs import REGISTRY

@pytest.fixture
def backend_data():
    return {
        "users": {
            "get_user": {"u-1": {"id": "u-1", "name": "Ada", "email": "ada@example.com"}},
            "get_user_meta": {"u-1": {"id": "u-1", "roles": ["admin"]}},
        },
        "orders": {
            "get_order": {"o-9": {"id": "o-9", "total": 123.45}},
        },
    }

@pytest.fixture
def mock_multi_server_mcp_client(monkeypatch, backend_data):
    fake = MagicMock(name="FakeMultiServerMCPClient")
    backend = InMemoryBackend(backend_data)
    tools_by_server = {s: build_tools_for_server(s, specs, backend) for s, specs in REGISTRY.items()}

    async def _get_tools(*, server_name: str | None = None):
        if server_name is None:
            out = []
            for lst in tools_by_server.values(): out.extend(lst)
            return out
        return list(tools_by_server.get(server_name, []))

    fake.get_tools = AsyncMock(side_effect=_get_tools)
    fake.get_resources = AsyncMock(return_value=[])

    @asynccontextmanager
    async def _session(server_name: str, *, auto_initialize: bool = True):
        yield MagicMock(name=f"session:{server_name}")
    fake.session = _session
    fake.close = AsyncMock()

    # Patch the wrapper path (stable everywhere)
    monkeypatch.setattr("yourpkg.mcp_client.MultiServerMCPClient", lambda *a, **k: fake, raising=True)
    return fake
```

---

## 6) Tiny tests

```python
# tests/test_tools_smoke.py
import pytest

@pytest.mark.asyncio
async def test_users_server_tools(mock_multi_server_mcp_client):
    users_tools = await mock_multi_server_mcp_client.get_tools(server_name="users")
    names = {t.name for t in users_tools}
    assert "users:get_user" in names

    get_user = next(t for t in users_tools if t.name.endswith(":get_user"))
    # Async invocation (Runnable interface)
    out = await get_user.ainvoke({"id": "u-1"})
    assert out["email"] == "ada@example.com"

@pytest.mark.asyncio
async def test_orders_server_tool(mock_multi_server_mcp_client):
    orders_tools = await mock_multi_server_mcp_client.get_tools(server_name="orders")
    get_order = next(t for t in orders_tools if t.name.endswith(":get_order"))
    out = await get_order.ainvoke({"id": "o-9"})
    assert out["total"] == 123.45
```

---

## Design Notes (short)

- **Stable names**: `"{server}:{endpoint}"` avoids collisions across servers.
- **Contracts**: `args_model` validates inputs; `response_model` (optional) validates outputs.
- **Backends**: swap `InMemoryBackend` for an HTTP/DB backend in prod wiring.
- **Patch once**: wrapper module ⇒ one consistent test patch target.

---

## Extending

- **Add endpoint**: append `EndpointSpec` to `REGISTRY[server]`; seed `backend_data` accordingly.
- **Error/latency** (for resilience tests): add flags to `InMemoryBackend` (e.g., `fail_on`, `latency_ms`).
- **Versioning**: keep tool names stable; store semver in `EndpointSpec.version` → `tool.metadata["version"]`.
  For breaking changes, add a new endpoint (`get_user_v2`) rather than renaming the old one.

---

## Troubleshooting

- **“fixture `mocker` not found”** → either install `pytest-mock` *or* use `monkeypatch` (shown here).
- **Pydantic v1 vs v2** → code uses `model_validate` with a `parse_obj` fallback.
- **Patch target** → always patch `"yourpkg.mcp_client.MultiServerMCPClient"` (the wrapper), not the library path.
- **Import order** → import the SUT *inside* tests after the fixture applies, or rely on the wrapper for stability.

---

## License
Use, modify, and copy freely in your project.
