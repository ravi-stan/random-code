# tests/test_fixture_smoke.py
import sys
import types
import pytest
from langchain_core.tools import BaseTool
from langchain_core.documents import Blob

# --- Make sure the patch target exists so the fixture's mocker.patch doesn't fail ---
@pytest.fixture(autouse=True, scope="session")
def _ensure_patch_target():
    if "yourpkg" not in sys.modules:
        sys.modules["yourpkg"] = types.ModuleType("yourpkg")
    if "yourpkg.your_module" not in sys.modules:
        sys.modules["yourpkg.your_module"] = types.ModuleType("yourpkg.your_module")


# -----------------------
# Very small fixture tests
# -----------------------

@pytest.mark.asyncio
async def test_tools_minimal(mock_multi_server_mcp_client):
    tools = await mock_multi_server_mcp_client.get_tools()
    assert isinstance(tools, list) and tools, "tools list should be non-empty"
    assert all(isinstance(t, BaseTool) for t in tools)
    # quick invocation of a known tool
    add = next(t for t in tools if t.name.endswith(":add"))
    assert add.invoke({"a": 2, "b": 3}) == 5


@pytest.mark.asyncio
async def test_tools_by_server_minimal(mock_multi_server_mcp_client):
    math_tools = await mock_multi_server_mcp_client.get_tools(server_name="math")
    assert math_tools and all(t.name.startswith("math:") for t in math_tools)


@pytest.mark.asyncio
async def test_resources_minimal(mock_multi_server_mcp_client):
    blobs = await mock_multi_server_mcp_client.get_resources(server_name="docs")
    assert isinstance(blobs, list) and blobs
    assert all(isinstance(b, Blob) for b in blobs)
    # At least one text-like resource should be string-decodable
    assert any(isinstance(b.as_string(), str) for b in blobs if (b.mime_type or "").startswith(("text/", "application/")))


@pytest.mark.asyncio
async def test_resources_filter_minimal(mock_multi_server_mcp_client):
    # The default fixture seeds a JSON resource at uri="app://config"
    only_config = await mock_multi_server_mcp_client.get_resources("docs", uris=["app://config"])
    assert len(only_config) == 1
    assert only_config[0].metadata.get("uri") == "app://config"
    assert '"ok": true' in only_config[0].as_string()


@pytest.mark.asyncio
async def test_session_minimal(mock_multi_server_mcp_client):
    async with mock_multi_server_mcp_client.session("weather") as session:
        # The session is a MagicMock named "session:weather"
        assert "session:weather" in repr(session)


@pytest.mark.asyncio
async def test_close_minimal(mock_multi_server_mcp_client):
    await mock_multi_server_mcp_client.close()
    mock_multi_server_mcp_client.close.assert_awaited_once()


# -----------------------
# Optional tiny checks for mk_tool / mk_blob factories
# -----------------------

def test_mk_blob_minimal(mk_blob):
    txt = mk_blob("hello", mime="text/plain", uri="app://hello")
    binb = mk_blob(b"\x00\x01", mime="application/octet-stream", uri="app://bin")
    assert txt.as_string() == "hello"
    assert len(binb.as_bytes()) == 2


def test_mk_tool_minimal(mk_tool):
    from pydantic import BaseModel
    class Args(BaseModel):
        x: int
        y: int
    tool = mk_tool("math", name="add_simple", description="adds", args_model=Args, func=lambda x, y: x + y)
    assert isinstance(tool, BaseTool)
    assert tool.name.endswith(":add_simple")
    assert tool.invoke({"x": 1, "y": 4}) == 5
