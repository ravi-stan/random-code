import pytest
from unittest.mock import AsyncMock
from types import SimpleNamespace

import importlib
m = importlib.import_module("agent_module")  # replace with your real module

@pytest.fixture
def span_double(mocker):
    span = mocker.MagicMock(name="span")
    ctx = SimpleNamespace(trace_id=0x1, span_id=0x2, is_remote=False)
    span.get_span_context.return_value = ctx
    span.gen_span_context.return_value = ctx
    return span

@pytest.fixture
async def agent(mocker):
    agent = m.ResearchAgent()
    async def init(session_id):
        agent.llm = object()
        agent.tool_manager = m.ToolManager({})
    agent.initialize = AsyncMock(side_effect=init)
    return agent

@pytest.mark.asyncio
async def test_no_selected_tools_early_return_and_logging(mocker, agent, span_double):
    agent.tool_manager = m.ToolManager({})
    mocker.patch.object(m.trace, "get_current_span", return_value=span_double)
    set_sid = mocker.patch.object(m, "set_session_id")
    logger = mocker.patch.object(m, "logger")

    sub_headings, err, trace_msgs = await agent._run_agent_loop(
        session_id="s1", query="q", tools=["UnknownTool"], parent_task="parent",
    )

    agent.initialize.assert_awaited_once_with("s1")
    set_sid.assert_called_once_with("s1")
    span_double.set_attribute.assert_any_call("Tools", ["UnknownTool"])
    span_double.set_attribute.assert_any_call("parent_task", "parent")
    span_double.set_attribute.assert_any_call("query", "q")
    assert trace_msgs is None
    assert len(sub_headings) == 1 and sub_headings[0].text == "Research Process"
    assert "does not exists" in err
    logger.error.assert_called_once()

@pytest.mark.asyncio
async def test_research_findings_path_returns_subheadings_and_trace(mocker, agent, span_double):
    agent.tool_manager = m.ToolManager({"T1": "Doc1"})
    mocker.patch.object(m.trace, "get_current_span", return_value=span_double)

    subhs = [m.KnowledgeCard.SubHeading(text="A"), m.KnowledgeCard.SubHeading(text="B")]
    agent._format_findings_to_subheadings = mocker.MagicMock(return_value=(subhs, ""))

    class Runnable:
        def __init__(self): self.calls = 0
        async def ainvoke(self, data, config=None):
            self.calls += 1
            if self.calls == 1:
                msgs = data["messages"]
                assert isinstance(msgs[0], m.HumanMessage)
                assert "Tool Documentation for T1:\nDoc1" in msgs[0].content
            return SimpleNamespace(
                tool_calls=[{"name": m.ResearchFindings.__name__,
                            "id": "rf1",
                            "args": {"no_relevant_bulletpoints_found": True}}],
                invalid_tool_calls=[]
            )

    mocker.patch.object(m, "create_agent_runnable", return_value=Runnable())
    logger = mocker.patch.object(m, "logger")

    sub_headings, err, trace_msgs = await agent._run_agent_loop(
        session_id="s1", query="what is X?", tools=["T1"], parent_task="parent",
        return_messages_trace=True,
    )

    assert sub_headings == subhs
    assert err == ""
    assert trace_msgs is not None and len(trace_msgs) >= 2
    logger.info.assert_called_once()
    logger.debug.assert_any_call(
        "Agent generated tool calls: [{'name': 'ResearchFindings', 'id': 'rf1', 'args': {'no_relevant_bulletpoints_found': True}}]"
    )

@pytest.mark.asyncio
async def test_tool_calls_then_continue_then_findings(mocker, agent, span_double):
    agent.tool_manager = m.ToolManager({"T1": "Doc1"})
    mocker.patch.object(m.trace, "get_current_span", return_value=span_double)

    agent._format_findings_to_subheadings = mocker.MagicMock(
        return_value=([m.KnowledgeCard.SubHeading(text="H")], "")
    )
    tm = m.ToolMessage(content="ok", tool_call_id="id1", status="ok")
    agent._execute_tool_calls = AsyncMock(return_value=[tm])

    first = SimpleNamespace(tool_calls=[{"name": "SomeTool", "id": "id1", "args": {"x": 1}}], invalid_tool_calls=[])
    second = SimpleNamespace(tool_calls=[{"name": m.ResearchFindings.__name__, "id": "rf1", "args": {"bullet_points": ["a"]}}], invalid_tool_calls=[])
    runnable = mocker.MagicMock()
    runnable.ainvoke = AsyncMock(side_effect=[first, second])

    logger = mocker.patch.object(m, "logger")
    mocker.patch.object(m, "create_agent_runnable", return_value=runnable)

    sub_headings, err, trace_msgs = await agent._run_agent_loop(
        session_id="s1", query="q", tools=["T1"], parent_task="p", return_messages_trace=True,
    )

    agent._format_findings_to_subheadings.assert_called_once()
    agent._execute_tool_calls.assert_awaited_once()
    args, kwargs = agent._execute_tool_calls.call_args
    assert args[0] == "s1" and args[1] == first.tool_calls
    logger.debug.assert_any_call(
        "Agent generated tool calls: [{'name': 'SomeTool', 'id': 'id1', 'args': {'x': 1}}]"
    )
    assert any(isinstance(mg, m.ToolMessage) for mg in trace_msgs)

@pytest.mark.asyncio
async def test_invalid_tool_calls_are_logged_and_continue(mocker, agent, span_double):
    agent.tool_manager = m.ToolManager({"T1": "Doc1"})
    mocker.patch.object(m.trace, "get_current_span", return_value=span_double)

    agent._format_findings_to_subheadings = mocker.MagicMock(
        return_value=([m.KnowledgeCard.SubHeading(text="H")], "")
    )
    invalids = [{"id": "bad1", "error": "bad json"}, {"id": "bad2", "error": "missing args"}]
    first = SimpleNamespace(tool_calls=[], invalid_tool_calls=invalids)
    second = SimpleNamespace(tool_calls=[{"name": m.ResearchFindings.__name__, "id": "rf1", "args": {}}], invalid_tool_calls=[])
    runnable = mocker.MagicMock()
    runnable.ainvoke = AsyncMock(side_effect=[first, second])
    logger = mocker.patch.object(m, "logger")
    mocker.patch.object(m, "create_agent_runnable", return_value=runnable)

    sub_headings, err, trace_msgs = await agent._run_agent_loop(
        session_id="s1", query="q", tools=["T1"], parent_task="p", return_messages_trace=True,
    )

    errs = [c.args[0] for c in logger.error.call_args_list]
    assert "Malformed tool call: bad json" in errs[0]
    assert "Malformed tool call: missing args" in errs[1]
    assert sum(1 for mg in trace_msgs if isinstance(mg, m.ToolMessage)) >= 2

@pytest.mark.asyncio
async def test_no_decision_fallback(mocker, agent, span_double):
    agent.tool_manager = m.ToolManager({"T1": "Doc1"})
    mocker.patch.object(m.trace, "get_current_span", return_value=span_double)

    first = SimpleNamespace(tool_calls=[], invalid_tool_calls=[])
    runnable = mocker.MagicMock()
    runnable.ainvoke = AsyncMock(return_value=first)
    logger = mocker.patch.object(m, "logger")
    mocker.patch.object(m, "create_agent_runnable", return_value=runnable)

    sub_headings, err, trace_msgs = await agent._run_agent_loop(
        session_id="s1", query="q", tools=["T1"], parent_task="p",
    )

    assert sub_headings[0].text == "Research concluded without a final answer."
    assert "Agent loop completed" in err
    assert trace_msgs is None
    logger.warning.assert_called_once()

@pytest.mark.asyncio
async def test_exception_path_returns_error_heading_and_trace(mocker, agent, span_double):
    agent.tool_manager = m.ToolManager({"T1": "Doc1"})
    mocker.patch.object(m.trace, "get_current_span", return_value=span_double)

    runnable = mocker.MagicMock()
    runnable.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
    logger = mocker.patch.object(m, "logger")
    mocker.patch.object(m, "create_agent_runnable", return_value=runnable)

    sub_headings, err, trace_msgs = await agent._run_agent_loop(
        session_id="s1", query="q", tools=["T1"], parent_task="p", return_messages_trace=True,
    )

    assert sub_headings[0].text == "An unexpected error occurred during research."
    assert "Agent loop failed with error: boom" in err
    assert trace_msgs is not None and isinstance(trace_msgs[0], m.HumanMessage)
    _, kwargs = logger.error.call_args
    assert kwargs.get("exc_info") is True

