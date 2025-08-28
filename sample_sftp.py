# tests/fixtures/mcp_multiserver_fake.py
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping

import pytest
import pytest_asyncio
from langchain_core.documents import Blob

# --- Minimal "MCP types" that match attributes used by langchain_mcp_adapters.tools ---
class _Tool:
    def __init__(self, name: str, description: str, input_schema: dict):
        self.name = name
        self.description = description
        # Be tolerant: some adapters look for either attribute name.
        self.input_schema = input_schema
        self.inputSchema = input_schema

class _ListToolsResult:
    def __init__(self, tools): self.tools = tools

class _TextContent:
    type = "text"
    def __init__(self, text: str): self.text = text

class _PromptMessage:
    # MCP PromptMessage has role + content[list[TextContent | ...]]
    def __init__(self, role: str, text: str):
        self.role = role
        self.content = [_TextContent(text)]

class _GetPromptResult:
    def __init__(self, messages): self.messages = messages

class _Resource:
    def __init__(self, uri: str, name: str | None = None, mime_type: str | None = None):
        self.uri = uri
        self.name = name or uri
        self.mimeType = mime_type

class _ListResourcesResult:
    def __init__(self, resources): self.resources = resources

class _ReadResourceResult:
    def __init__(self, content): self.content = content

class _CallToolResult:
    def __init__(self, structured: Any | None = None, text: str | None = None):
        # Being generous: return both structured and text to satisfy older adapters.
        self.structuredContent = structured
        self.content = [] if text is None else [_TextContent(text)]


# --- Fake server registry ------------------------------------------------------
@dataclass
class FakeToolSpec:
    description: str
    schema: dict[str, Any]
    impl: Callable[..., Any]

@dataclass
class FakeServer:
    tools: Dict[str, FakeToolSpec]               # tool_name -> spec
    prompts: Dict[str, list[tuple[str, str]]]    # prompt_name -> [(role, text), ...]
    resources: Dict[str, str]                    # uri -> text payload


class _FakeSession:
    """Tiny in-memory MCP 'session' that matches the attributes the adapter reads."""
    def __init__(self, server: FakeServer, headers: dict[str, str] | None = None):
        self._server = server
        self._headers = headers or {}
        self.initialized = False

    async def initialize(self) -> None:
        self.initialized = True

    # --- tool discovery + execution ---
    async def list_tools(self) -> _ListToolsResult:
        tools = [
            _Tool(name=name, description=spec.description, input_schema=spec.schema)
            for name, spec in self._server.tools.items()
        ]
        return _ListToolsResult(tools)

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> _CallToolResult:
        if name not in self._server.tools:
            raise KeyError(f"Unknown tool: {name}")
        fn = self._server.tools[name].impl
        result = fn(**(dict(arguments) if arguments else {}))
        # Return both structured + text to be adapter-version-proof
        return _CallToolResult(structured=result, text=str(result))

    # --- prompts ---
    async def get_prompt(self, name: str, arguments: Mapping[str, Any] | None = None) -> _GetPromptResult:
        if name not in self._server.prompts:
            raise KeyError(f"Unknown prompt: {name}")
        msgs = []
        for role, text in self._server.prompts[name]:
            msgs.append(_PromptMessage(role, text.format(**(arguments or {}))))
        return _GetPromptResult(messages=msgs)

    # --- resources ---
    async def list_resources(self) -> _ListResourcesResult:
        return _ListResourcesResult([_Resource(uri=u) for u in self._server.resources])

    async def read_resource(self, uri: str) -> _ReadResourceResult:
        if uri not in self._server.resources:
            raise KeyError(f"Unknown resource: {uri}")
        return _ReadResourceResult([_TextContent(self._server.resources[uri])])


# ---------- Pytest fixtures ----------------------------------------------------

@pytest.fixture
def fake_servers_registry() -> dict[str, FakeServer]:
    """Two deterministic fake MCP servers."""
    alpha = FakeServer(
        tools={
            "add": FakeToolSpec(
                description="Add two integers",
                schema={"type": "object",
                        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                        "required": ["a", "b"]},
                impl=lambda a, b: a + b,
            ),
            "upper": FakeToolSpec(
                description="Uppercase text",
                schema={"type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"]},
                impl=lambda text: text.upper(),
            ),
        },
        prompts={"greet": [("system", "You are concise."), ("user", "Say hi to {name}.")]},
        resources={"alpha://note": "hello from alpha"},
    )
    beta = FakeServer(
        tools={
            "echo": FakeToolSpec(
                description="Echo text",
                schema={"type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"]},
                impl=lambda text: text,
            )
        },
        prompts={"bye": [("user", "Say bye to {name}.")]},
        resources={"beta://readme": "beta resource content"},
    )
    return {"alpha": alpha, "beta": beta}


@pytest_asyncio.fixture
async def multiserver_mcp_client_fake(monkeypatch, fake_servers_registry):
    """
    Returns a REAL `langchain_mcp_adapters.client.MultiServerMCPClient` instance
    whose `.session()` is patched to yield in-memory fake sessions per server.
    Also exposes counters to assert 'new session per tool call'.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient  # <-- real class

    # We'll store per-server session open counts and "seen headers"
    counters: dict[str, int] = {k: 0 for k in fake_servers_registry}
    seen_headers: dict[str, list[dict[str, str]]] = {k: [] for k in fake_servers_registry}

    # For resilience to internal attr names, try several likely locations for connections
    def _extract_headers(self, server_name: str) -> dict[str, str]:
        for attr in ("connections", "_connections", "config", "_config", "__dict__"):
            cfg = getattr(self, attr, None)
            if isinstance(cfg, dict) and server_name in cfg:
                maybe = cfg[server_name].get("headers") or cfg[server_name].get("mcp_server", {}).get("headers")
                if isinstance(maybe, dict):
                    return dict(maybe)
        # Default to no headers if we can't find any
        return {}

    @asynccontextmanager
    async def fake_session(self, server_name: str, *, auto_initialize: bool = True):
        counters[server_name] += 1
        headers = _extract_headers(self, server_name)
        seen_headers[server_name].append(headers)
        sess = _FakeSession(fake_servers_registry[server_name], headers=headers)
        if auto_initialize:
            await sess.initialize()
        try:
            yield sess
        finally:
            pass

    # Patch the bound method on the CLASS so all instances use the fake
    monkeypatch.setattr(MultiServerMCPClient, "session", fake_session, raising=True)

    # Build a realistic connection config (headers only matter for http/sse in prod)
    client = MultiServerMCPClient(
        connections={
            "alpha": {"transport": "streamable_http", "url": "http://fake-alpha/mcp", "headers": {"x-session-id": "alpha-123"}},
            "beta":  {"transport": "streamable_http", "url": "http://fake-beta/mcp",  "headers": {"x-session-id": "beta-456"}},
        }
    )

    # Expose spies for assertions in tests
    client._session_open_count = counters          # type: ignore[attr-defined]
    client._seen_headers_by_server = seen_headers  # type: ignore[attr-defined]
    return client





############

import pytest
from langchain_core.messages import BaseMessage

@pytest.mark.asyncio
async def test_get_tools_and_invoke(multiserver_mcp_client_fake):
    client = multiserver_mcp_client_fake

    tools = await client.get_tools()   # flattened across alpha + beta
    names = {t.name for t in tools}
    assert {"add", "upper", "echo"} <= names

    add = next(t for t in tools if t.name == "add")
    upper = next(t for t in tools if t.name == "upper")
    echo = next(t for t in tools if t.name == "echo")

    # Each invoke should cause the adapter to open a fresh session under the hood
    assert await add.ainvoke({"a": 2, "b": 5}) == 7
    assert await upper.ainvoke({"text": "abc"}) == "ABC"
    assert await echo.ainvoke({"text": "hi"}) == "hi"

    # list_tools + 3 calls -> at least 4 sessions opened across servers
    total_opens = sum(client._session_open_count.values())  # from the fixture
    assert total_opens >= 4

@pytest.mark.asyncio
async def test_get_tools_filtered_server(multiserver_mcp_client_fake):
    client = multiserver_mcp_client_fake

    beta_tools = await client.get_tools(server_name="beta")
    assert {t.name for t in beta_tools} == {"echo"}
    # Ensure only beta saw a session for discovery
    assert client._session_open_count["beta"] >= 1

@pytest.mark.asyncio
async def test_get_prompt_returns_langchain_messages(multiserver_mcp_client_fake):
    msgs = await multiserver_mcp_client_fake.get_prompt("alpha", "greet", {"name": "Ada"})
    assert all(isinstance(m, BaseMessage) for m in msgs)
    # content should contain the formatted argument
    assert any("Ada" in getattr(m, "content", "") for m in msgs)

@pytest.mark.asyncio
async def test_get_resources_returns_blobs(multiserver_mcp_client_fake):
    blobs = await multiserver_mcp_client_fake.get_resources("alpha")
    # The adapter should convert our fake resources to LangChain Blobs
    txts = [b.as_string() for b in blobs]
    assert any("hello from alpha" in t for t in txts)

@pytest.mark.asyncio
async def test_runtime_headers_captured(multiserver_mcp_client_fake):
    seen = multiserver_mcp_client_fake._seen_headers_by_server
    # At least one session open → at least one header capture
    assert any(h.get("x-session-id") == "alpha-123" for h in seen["alpha"])
    assert any(h.get("x-session-id") == "beta-456" for h in seen["beta"])
