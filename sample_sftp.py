# tests/fixtures/mcp_multi_unit.py
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import pytest
import pytest_asyncio

# Optional: LangChain Tool + Blob for "langchain-compatible" return types
from langchain_core.tools import Tool
from langchain_core.documents import Blob  # Blob.from_data(...) etc.

# ---------- A tiny in-memory "registry" model for fake servers ----------

@dataclass
class FakeToolSpec:
    description: str
    schema: dict[str, Any]  # JSON schema your client may read
    impl: Callable[..., Any]  # what to run on call_tool

@dataclass
class FakeServer:
    tools: dict[str, FakeToolSpec]                 # name -> spec
    prompts: dict[str, list[tuple[str, str]]]      # name -> [(role, text), ...]
    resources: dict[str, str]                      # uri -> text payload


# ---------- Minimal "mcp.types-like" result objects your client may touch -----
# We don't depend on mcp SDK for unit tests; we emulate the shapes your client reads.

class _Tool:
    def __init__(self, name: str, description: str, input_schema: dict):
        self.name = name
        self.description = description
        # cover both snake_case and camelCase, depending on how your client reads it
        self.input_schema = input_schema
        self.inputSchema = input_schema

class _ListToolsResult:
    def __init__(self, tools: list[_Tool]): self.tools = tools

class _TextContent:
    type = "text"
    def __init__(self, text: str): self.text = text

class _PromptMessage:
    # Simplified; real SDK has richer content blocks. This works for most adapters.
    def __init__(self, role: str, text: str):
        self.role = role
        self.content = [_TextContent(text)]

class _GetPromptResult:
    def __init__(self, messages: list[_PromptMessage]): self.messages = messages

class _Resource:
    def __init__(self, uri: str, name: str | None = None, mime_type: str | None = None):
        self.uri = uri
        self.name = name or uri
        self.mimeType = mime_type

class _ListResourcesResult:
    def __init__(self, resources: list[_Resource]): self.resources = resources

class _ReadResourceResult:
    def __init__(self, content: list[_TextContent]): self.content = content

class _CallToolResult:
    def __init__(self, structured: Any | None = None, text: str | None = None):
        # mirror common fields a client reads back
        self.structuredContent = structured
        self.content = [_TextContent(text)] if text is not None else []


# ---------- A fake session your MultiServerMCPClient.session will yield --------

class _FakeSession:
    def __init__(self, server: FakeServer):
        self._server = server
        self._initialized = False

    async def initialize(self) -> None:
        self._initialized = True

    async def list_tools(self) -> _ListToolsResult:
        tools = [
            _Tool(name=n, description=spec.description, input_schema=spec.schema)
            for n, spec in self._server.tools.items()
        ]
        return _ListToolsResult(tools=tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> _CallToolResult:
        if name not in self._server.tools:
            raise KeyError(f"Tool {name} not found")
        fn = self._server.tools[name].impl
        result = fn(**(arguments or {}))
        # choose structured vs text for convenience
        if isinstance(result, (dict, list, int, float, bool)) or result is None:
            return _CallToolResult(structured=result)
        return _CallToolResult(text=str(result))

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> _GetPromptResult:
        if name not in self._server.prompts:
            raise KeyError(f"Prompt {name} not found")
        # crude templating for unit tests
        messages = []
        for role, text in self._server.prompts[name]:
            if arguments:
                text = text.format(**arguments)
            messages.append(_PromptMessage(role, text))
        return _GetPromptResult(messages=messages)

    async def list_resources(self) -> _ListResourcesResult:
        resources = [_Resource(uri=u) for u in self._server.resources.keys()]
        return _ListResourcesResult(resources=resources)

    async def read_resource(self, uri: str) -> _ReadResourceResult:
        if uri not in self._server.resources:
            raise KeyError(f"Resource {uri} not found")
        return _ReadResourceResult([_TextContent(self._server.resources[uri])])


# ---------- Pytest fixtures: fake registry + patched client.session ------------

@pytest.fixture
def fake_servers_registry() -> dict[str, FakeServer]:
    """Two deterministic fake MCP servers."""
    alpha = FakeServer(
        tools={
            "add": FakeToolSpec(
                description="Add two ints",
                schema={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]},
                impl=lambda a, b: a + b,
            ),
            "upper": FakeToolSpec(
                description="Uppercase a string",
                schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
                impl=lambda text: text.upper(),
            ),
        },
        prompts={
            "greet": [("system", "You are warm and concise."), ("user", "Say hi to {name}.")],
        },
        resources={
            "alpha://foo": "hello from alpha",
            "alpha://bar": "more alpha text",
        },
    )
    beta = FakeServer(
        tools={
            "echo": FakeToolSpec(
                description="Echo back text",
                schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
                impl=lambda text: text,
            )
        },
        prompts={"farewell": [("user", "Say bye to {name}.")]},
        resources={"beta://readme": "beta resource content"},
    )
    return {"alpha": alpha, "beta": beta}


@pytest_asyncio.fixture
async def multiserver_client_fake(monkeypatch, fake_servers_registry):
    """
    Returns a real MultiServerMCPClient with its .session method patched so that:
      - each entry uses the in-memory fake server above
      - we count openings to prove 'new session per tool call'
    """
    # Adjust import path to your codebase:
    from app.mcp_multi import MultiServerMCPClient  # <-- change if needed

    # A tiny spy to count session opens per server
    counters: dict[str, int] = {"alpha": 0, "beta": 0}

    @asynccontextmanager
    async def fake_session(self, server_key: str):
        counters[server_key] += 1
        sess = _FakeSession(fake_servers_registry[server_key])
        await sess.initialize()
        try:
            yield sess
        finally:
            pass

    monkeypatch.setattr(MultiServerMCPClient, "session", fake_session, raising=True)

    # headers aren't used by the fake, but we include them to mirror prod config
    client = MultiServerMCPClient(
        connections={
            "alpha": {"mcp_server": {"transport": "http", "url": "http://fake-alpha/mcp", "headers": {"x-session-id": "alpha-123"}}},
            "beta":  {"mcp_server": {"transport": "http", "url": "http://fake-beta/mcp",  "headers": {"x-session-id": "beta-456"}}},
        }
    )
    # attach counters so tests can assert behavior
    client._session_open_count = counters
    return client



#####################################

import pytest

from langchain_core.documents import Blob
from langchain_core.tools import Tool

@pytest.mark.asyncio
async def test_get_tools_aggregates(monkeypatch, multiserver_client_fake):
    tools = await multiserver_client_fake.get_tools()
    names = {t.name for t in tools}
    assert {"add", "upper", "echo"} <= names
    # Should have opened sessions to list tools on both servers
    assert multiserver_client_fake._session_open_count["alpha"] >= 1
    assert multiserver_client_fake._session_open_count["beta"]  >= 1

@pytest.mark.asyncio
async def test_tool_invocation_opens_new_session_each_time(multiserver_client_fake):
    tools = await multiserver_client_fake.get_tools()
    add = next(t for t in tools if t.name == "add")
    upper = next(t for t in tools if t.name == "upper")

    # Call twice; each call should create a NEW session via the patched contextmanager
    assert await add.invoke({"a": 2, "b": 5}) == 7
    assert await upper.invoke({"text": "hello"}) == "HELLO"

    # Two calls above -> at least 2 fresh sessions somewhere (server-specific counts may vary)
    total_sessions = sum(multiserver_client_fake._session_open_count.values())
    assert total_sessions >= 3  # list_tools + 2 calls

@pytest.mark.asyncio
async def test_get_prompt_renders(monkeypatch, multiserver_client_fake):
    prompt = await multiserver_client_fake.get_prompt("alpha", "greet", {"name": "Ada"})
    # Your client may return different shapes (messages or a LangChain prompt).
    # Here we accept "messages list" OR a ChatPromptTemplate rendered result.
    if isinstance(prompt, list):
        text = " ".join([blk.text for msg in prompt for blk in msg.content])
        assert "Ada" in text
    else:
        # e.g., ChatPromptTemplate — just ensure rendering works
        rendered = prompt.format_messages(name="Ada")
        assert any("Ada" in (c.content if hasattr(c, "content") else str(c)) for c in rendered)

@pytest.mark.asyncio
async def test_get_resources_returns_langchain_blobs(multiserver_client_fake):
    blobs = await multiserver_client_fake.get_resources("beta")
    assert all(isinstance(b, Blob) for b in blobs)
    # Despite faking, Blob API should behave properly
    assert any("beta resource content" in b.as_string() for b in blobs)
