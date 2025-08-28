# tests/conftest.py
from __future__ import annotations

import inspect
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional, Type
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool, BaseTool      # BaseTool contract
from langchain_core.documents import Blob                      # for realistic resources


# ---------- Utilities ----------

@pytest.fixture
def mk_blob():
    """Factory for LangChain Blobs with sane defaults (text or binary)."""
    def _mk(
        data: str | bytes,
        *,
        mime: str = "text/plain",
        uri: str = "app://resource",
        metadata: Optional[dict] = None,
    ) -> Blob:
        md = {"uri": uri, **(metadata or {})}
        return Blob.from_data(data=data, mime_type=mime, metadata=md)
    return _mk


@pytest.fixture
def mk_tool():
    """
    Factory that returns a REAL BaseTool (StructuredTool) matching LangChain's interface.

    Supported BaseTool attributes:
      - name, description, args_schema, return_direct, response_format
      - tags, metadata
      - handle_tool_error, handle_validation_error (optional)

    Use:
        add = mk_tool(
            server="math",
            name="add",
            description="Add two integers.",
            args_model=AddArgs,
            func=lambda a, b: a + b,
            tags=["unit-test"],
            metadata={"team": "platform"},
            return_direct=False,
            response_format="content",
        )
    """
    def _mk(
        server: str,
        *,
        name: str,
        description: str,
        args_model: Type[BaseModel],
        func: Optional[Callable[..., Any]] = None,
        coroutine: Optional[Callable[..., Any]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
        return_direct: bool = False,
        response_format: str = "content",
        handle_tool_error: bool | str | Callable[..., str] | None = False,
        handle_validation_error: bool | str | Callable[..., str] | None = False,
    ) -> BaseTool:
        if func is None and coroutine is None:
            # Default no-op behavior that still exercises plumbing
            def func(**kwargs):  # type: ignore[no-redef]
                return {"echo": kwargs}

        # Build name unique across servers to avoid collisions
        tool_name = f"{server}:{name}"

        # Compose tags/metadata with MCP context for tracing
        tags = (tags or []) + [f"server:{server}", "mcp"]
        metadata = {**(metadata or {}), "mcp": {"server": server, "tool": name}}

        # StructuredTool exposes the BaseTool fields and forwards **kwargs.
        # We can set response_format, return_direct, tags, metadata, and error handlers here.
        tool = StructuredTool.from_function(
            func=func,
            coroutine=coroutine,
            name=tool_name,
            description=description,
            args_schema=args_model,
            return_direct=return_direct,
            response_format=response_format,
            handle_tool_error=handle_tool_error,
            handle_validation_error=handle_validation_error,
            tags=tags,
            metadata=metadata,
        )
        return tool

    return _mk


# ---------- Example args schemas you can reuse in tests ----------

class AddArgs(BaseModel):
    a: int = Field(..., description="First addend")
    b: int = Field(..., description="Second addend")

class ForecastArgs(BaseModel):
    location: str = Field(..., description="City or 'lat,long'")


# ---------- Default tool spec (override per-test if desired) ----------

@pytest.fixture
def mcp_tool_specs(mk_tool):
    """Default server->tools mapping; tests can mutate/replace this as needed."""
    return {
        "math": [
            mk_tool(
                "math",
                name="add",
                description="Add two integers.",
                args_model=AddArgs,
                func=lambda a, b: a + b,
            ),
            mk_tool(
                "math",
                name="fail",
                description="Always fails (demonstrates handle_tool_error).",
                args_model=AddArgs,
                func=lambda a, b: 1 / 0,  # ZeroDivisionError
                handle_tool_error=lambda e: "division by zero blocked",  # convert to str
            ),
        ],
        "weather": [
            mk_tool(
                "weather",
                name="get_weather",
                description="Get weather for a location.",
                args_model=ForecastArgs,
                func=lambda location: f"Sunny in {location}",
                tags=["demo"],
                metadata={"owner": "wx-team"},
            ),
        ],
    }


# ---------- Primary fixture: mocked MultiServerMCPClient ----------

@pytest.fixture
def mock_multi_server_mcp_client(mocker, mk_blob, mcp_tool_specs):
    """
    Patches MultiServerMCPClient in *your* module under test and returns a fake instance.

    Supports:
      - await client.get_tools(server_name=None|str) -> list[BaseTool]
      - await client.get_resources(server_name, uris=None|str|list[str]) -> list[Blob]
      - async with client.session(server_name) as session: ...
      - await client.close()

    You can override/extend tools and resources in tests via:
        mock._tools_by_server["math"] = [...]
        mock._resources_by_server["docs"].append(...)
    """
    fake = MagicMock(
        name="FakeMultiServerMCPClient",
        spec_set=["get_tools", "get_resources", "session", "close"],
    )

    # Tool & resource backing stores (tests can mutate these)
    fake._tools_by_server: Dict[str, List[BaseTool]] = mcp_tool_specs
    fake._resources_by_server: Dict[str, List[Blob]] = {
        "docs": [
            mk_blob("# README\nHello!", mime="text/markdown", uri="file:///README.md"),
            mk_blob('{"ok": true}', mime="application/json", uri="app://config"),
        ],
        "weather": [
            mk_blob("NYC: 72F", mime="text/plain", uri="app://weather/nyc"),
            mk_blob(b"\x89PNG\r\n\x1a\n..", mime="image/png", uri="file:///wx.png"),
        ],
    }

    # get_tools(server_name?: str)
    async def _get_tools(*, server_name: Optional[str] = None) -> List[BaseTool]:
        if server_name is None:
            out: List[BaseTool] = []
            for tools in fake._tools_by_server.values():
                out.extend(tools)
            return out
        return list(fake._tools_by_server.get(server_name, []))

    fake.get_tools = AsyncMock(side_effect=_get_tools)

    # get_resources(server_name: str, uris?: str|list[str])
    async def _get_resources(server_name: str, *, uris: Optional[str | List[str]] = None) -> List[Blob]:
        blobs = list(fake._resources_by_server.get(server_name, []))
        if uris:
            want = [uris] if isinstance(uris, str) else list(uris)
            blobs = [b for b in blobs if (b.metadata or {}).get("uri") in want]
        return blobs

    fake.get_resources = AsyncMock(side_effect=_get_resources)

    # session(server_name) -> async context manager
    @asynccontextmanager
    async def _session(server_name: str, *, auto_initialize: bool = True):
        yield MagicMock(name=f"session:{server_name}")

    fake.session = _session

    # close()
    fake.close = AsyncMock()

    # 🔁 Patch where your SUT imports the class (adjust the string below!)
    mocker.patch("yourpkg.your_module.MultiServerMCPClient", return_value=fake)

    return fake
