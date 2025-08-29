# yourpkg/tooling.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Type
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

# ---------- Args schemas ----------

class IdArgs(BaseModel):
    id: str = Field(..., description="Resource identifier")

# Example: endpoint with extra flags
class IdWithIncludeArgs(BaseModel):
    id: str = Field(..., description="Resource identifier")
    include: List[str] | None = Field(None, description="Optional fields to include")


# ---------- Declarative spec ----------

@dataclass(frozen=True)
class EndpointSpec:
    name: str
    description: str
    args_model: Type[BaseModel] = IdArgs
    # Optionally validate JSON responses against a Pydantic model
    response_model: Optional[Type[BaseModel]] = None
    tags: tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Optional semver tracked in metadata; avoid putting it in the tool name unless you must
    version: str = "1.0.0"


# ---------- Backend protocol + test/prod impls ----------

class Backend(Protocol):
    async def query(self, server: str, endpoint: str, args: BaseModel) -> Any: ...

class InMemoryBackend:
    """
    data shape: data[server][endpoint][id] -> JSON-serializable object
    """
    def __init__(self, data: Dict[str, Dict[str, Dict[str, Any]]]):
        self.data = data

    async def query(self, server: str, endpoint: str, args: BaseModel) -> Any:
        _id = args.model_dump().get("id")
        return self.data[server][endpoint][_id]

# You can later add an HttpBackend here that does real I/O.


# ---------- Tool builders ----------

def mk_endpoint_tool(server: str, spec: EndpointSpec, backend: Backend) -> StructuredTool:
    async def run(**kwargs):
        # Validate inputs via args_model
        args = spec.args_model(**kwargs)
        result = await backend.query(server, spec.name, args)
        # Optionally validate outputs
        if spec.response_model is not None:
            try:  # pydantic v2
                result = spec.response_model.model_validate(result).model_dump()
            except AttributeError:  # pydantic v1 fallback
                result = spec.response_model.parse_obj(result).dict()
        return result

    # Keep the name stable & unique across servers
    tool_name = f"{server}:{spec.name}"
    tags = [*spec.tags, f"server:{server}", "mcp"]
    metadata = {**spec.metadata, "server": server, "endpoint": spec.name, "version": spec.version}

    # NOTE: avoid passing response_format if you need max compatibility across LC versions
    return StructuredTool.from_function(
        coroutine=run,
        name=tool_name,
        description=spec.description,
        args_schema=spec.args_model,
        tags=tags,
        metadata=metadata,
        return_direct=False,
        # response_format="content",  # uncomment if your LC version supports it
    )

def build_tools_for_server(server: str, specs: List[EndpointSpec], backend: Backend):
    return [mk_endpoint_tool(server, s, backend) for s in specs]



# yourpkg/toolspecs.py
from __future__ import annotations
from typing import Dict, List
from pydantic import BaseModel, Field
from .tooling import EndpointSpec, IdArgs, IdWithIncludeArgs

# Optional response models (stronger contracts)
class User(BaseModel):
    id: str
    name: str
    email: str

class Order(BaseModel):
    id: str
    total: float

# Registry: server -> list[EndpointSpec]
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
            args_model=IdWithIncludeArgs,  # demonstrates extra args beyond id
            version="1.0.0",
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





# yourpkg/your_module.py
from yourpkg.mcp_client import MultiServerMCPClient  # <— wrapper import

CONNECTIONS = {
    "users": {"url": "http://localhost:8000/mcp", "transport": "streamable_http"},
    "orders": {"command": "python", "args": ["./orders_server.py"], "transport": "stdio"},
}

async def load_all_tools():
    client = MultiServerMCPClient(CONNECTIONS)
    try:
        return await client.get_tools()
    finally:
        await client.close()





# tests/conftest.py
from __future__ import annotations
import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from yourpkg.tooling import InMemoryBackend, build_tools_for_server
from yourpkg.toolspecs import REGISTRY

@pytest.fixture
def backend_data():
    # server -> endpoint -> id -> payload
    return {
        "users": {
            "get_user": {
                "u-1": {"id": "u-1", "name": "Ada", "email": "ada@example.com"},
            },
            "get_user_meta": {
                "u-1": {"id": "u-1", "signup_ts": "2020-01-01", "roles": ["admin"]},
            },
        },
        "orders": {
            "get_order": {
                "o-9": {"id": "o-9", "total": 123.45},
            },
        },
    }

@pytest.fixture
def mock_multi_server_mcp_client(monkeypatch, backend_data):
    fake = MagicMock(name="FakeMultiServerMCPClient")

    backend = InMemoryBackend(backend_data)
    tools_by_server = {srv: build_tools_for_server(srv, specs, backend) for srv, specs in REGISTRY.items()}

    async def _get_tools(*, server_name: str | None = None):
        if server_name is None:
            out = []
            for lst in tools_by_server.values():
                out.extend(lst)
            return out
        return list(tools_by_server.get(server_name, []))

    fake.get_tools = AsyncMock(side_effect=_get_tools)
    fake.get_resources = AsyncMock(return_value=[])

    @asynccontextmanager
    async def _session(server_name: str, *, auto_initialize: bool = True):
        yield MagicMock(name=f"session:{server_name}")
    fake.session = _session
    fake.close = AsyncMock()

    # Patch the single stable wrapper path
    monkeypatch.setattr("yourpkg.mcp_client.MultiServerMCPClient", lambda *a, **k: fake, raising=True)

    # Useful handles for tests
    fake.registry = REGISTRY
    fake.backend = backend
    fake.tools_by_server = tools_by_server
    fake.data = backend_data
    return fake





# tests/test_tools_smoke.py
import pytest

@pytest.mark.asyncio
async def test_users_server_tools(mock_multi_server_mcp_client):
    users_tools = await mock_multi_server_mcp_client.get_tools(server_name="users")
    names = {t.name for t in users_tools}
    assert "users:get_user" in names
    assert "users:get_user_meta" in names

    get_user = next(t for t in users_tools if t.name.endswith(":get_user"))
    out = await get_user.arun({"id": "u-1"})
    assert out["email"] == "ada@example.com"    # validated by response_model

    get_meta = next(t for t in users_tools if t.name.endswith(":get_user_meta"))
    out2 = await get_meta.arun({"id": "u-1", "include": ["roles"]})
    assert "roles" in out2

@pytest.mark.asyncio
async def test_orders_server_tool(mock_multi_server_mcp_client):
    orders_tools = await mock_multi_server_mcp_client.get_tools(server_name="orders")
    get_order = next(t for t in orders_tools if t.name.endswith(":get_order"))
    out = await get_order.arun({"id": "o-9"})
    assert out["total"] == 123.45

