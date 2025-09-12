# test_run_agent_loop.py
import pytest
from unittest.mock import AsyncMock
from types import SimpleNamespace

import agent_module as m  # <-- change to your module

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
async def test_create_agent_runnable_called_and_initial_message_contains_all_fields(mocker, agent, span_double):
    agent.tool_manager = m.ToolManager({"T1": "Doc1", "T2": ""})  # T2 doc empty -> excluded from docs section
    mocker.patch.object(m.trace, "get_current_span", return_value=span_double)
    mocker.patch.object(m, "set_session_id")

    # Runnable asserts config + message content and immediately returns ResearchFindings
    class Runnable:
        async def ainvoke(self, data, config=None):
            assert isinstance(config, m.RunnableConfig)
            assert config.run_name == "research_agent.tool_calling"
            msgs = data["messages"]
            assert len(msgs) == 1 and isinstance(msgs[0], m.HumanMessage)
            c = msgs[0].content
            assert "<original_research_task>PARENT</original_research_task>" in c
            assert "<caller_context>CTX</caller_context>" in c
            assert "<sub_task>QUERY</sub_task>" in c
            assert "Tool Documentation for T1:\nDoc1" in c
            assert "Tool Documentation for T2" not in c
            return SimpleNamespace(
                tool_calls=[{"name": m.ResearchFindings.__name__, "id": "rf1", "args": {"bullet_points": ["a"]}}],
                invalid_tool_calls=[]
            )

    car = mocker.patch.object(m, "create_agent_runnable", return_value=Runnable())
    agent._format_findings_to_subheadings = mocker.MagicMock(return_value=([m.KnowledgeCard.SubHeading(text="A")], ""))

    sub_headings, err, trace_msgs = await agent._run_agent_loop(
        session_id="S", query="QUERY", tools=["T1", "T2"], parent_task="PARENT",
        caller_context="CTX", instructions="custom", return_messages_trace=True,
    )

    # create_agent_runnable called with llm, selected_tools (T1,T2), and instructions
    args, _ = car.call_args
    assert args[0] is agent.llm
    assert sorted([t.name for t in args[1]]) == ["T1", "T2"]
    assert args[2] == "custom"

    assert [h.text for h in sub_headings] == ["A"]
    assert err == ""
    assert trace_msgs and len(trace_msgs) == 2
    assert isinstance(trace_msgs[0], m.HumanMessage)
    assert trace_msgs[1].tool_calls[0]["args"]["bullet_points"] == ["a"]

@pytest.mark.asyncio
async def test_invalid_tool_calls_append_tool_messages_and_continue(mocker, agent, span_double):
    agent.tool_manager = m.ToolManager({"T1": "D"})
    mocker.patch.object(m.trace, "get_current_span", return_value=span_double)
    logger = mocker.patch.object(m, "logger")

    class Runnable:
        def __init__(self): self.calls = 0
        async def ainvoke(self, data, config=None):
            self.calls += 1
            msgs = data["messages"]
            if self.calls == 1:
                assert len(msgs) == 1 and isinstance(msgs[0], m.HumanMessage)
                return SimpleNamespace(
                    tool_calls=[],
                    invalid_tool_calls=[{"id":"bad1","error":"bad json"},{"id":"bad2","error":"missing args"}]
                )
            else:
                tool_errs = [mg for mg in msgs if isinstance(mg, m.ToolMessage)]
                assert {tm.tool_call_id for tm in tool_errs} >= {"bad1","bad2"}
                return SimpleNamespace(
                    tool_calls=[{"name": m.ResearchFindings.__name__, "id":"rf", "args": {}}],
                    invalid_tool_calls=[],
                )

    mocker.patch.object(m, "create_agent_runnable", return_value=Runnable())
    agent._format_findings_to_subheadings = mocker.MagicMock(return_value=([m.KnowledgeCard.SubHeading(text="H")], ""))

    sub_headings, err, trace_msgs = await agent._run_agent_loop(
        session_id="S", query="Q", tools=["T1"], parent_task="P", return_messages_trace=True,
    )

    errs = [c.args[0] for c in logger.error.call_args_list]
    assert "Malformed tool call: bad json" in errs[0]
    assert "Malformed tool call: missing args" in errs[1]
    assert sum(1 for mg in trace_msgs if isinstance(mg, m.ToolMessage) and mg.status == "error") >= 2
    assert [h.text for h in sub_headings] == ["H"]
    assert err == ""

@pytest.mark.asyncio
async def test_tool_calls_then_execute_then_continue_includes_tool_messages_in_next_invoke(mocker, agent, span_double):
    agent.tool_manager = m.ToolManager({"Search": "Doc"})
    mocker.patch.object(m.trace, "get_current_span", return_value=span_double)
    logger = mocker.patch.object(m, "logger")

    tm = m.ToolMessage(content="RESULT", tool_call_id="id1", status="ok")
    agent._execute_tool_calls = AsyncMock(return_value=[tm])
    agent._format_findings_to_subheadings = mocker.MagicMock(return_value=([m.KnowledgeCard.SubHeading(text="Done")], ""))

    class Runnable:
        def __init__(self): self.calls = 0
        async def ainvoke(self, data, config=None):
            self.calls += 1
            msgs = data["messages"]
            if self.calls == 1:
                return SimpleNamespace(tool_calls=[{"name":"Search","id":"id1","args":{"q":"x"}}], invalid_tool_calls=[])
            else:
                assert any(isinstance(mg, m.ToolMessage) and mg.tool_call_id == "id1" for mg in msgs)
                return SimpleNamespace(tool_calls=[{"name": m.ResearchFindings.__name__, "id":"rf", "args":{"bullet_points":["bp"]}}], invalid_tool_calls=[])

    mocker.patch.object(m, "create_agent_runnable", return_value=Runnable())

    sub_headings, err, trace_msgs = await agent._run_agent_loop(
        session_id="S", query="Q", tools=["Search"], parent_task="P", return_messages_trace=True,
    )

    agent._execute_tool_calls.assert_awaited_once()
    args, _ = agent._execute_tool_calls.call_args
    assert args[0] == "S" and args[1] == [{"name":"Search","id":"id1","args":{"q":"x"}}]
    assert any("Agent generated tool calls" in c.args[0] for c in logger.debug.call_args_list)
    assert any(c.args[0].startswith("Tool result - tool_call_id: id1") for c in logger.debug.call_args_list if c.args)
    assert [h.text for h in sub_headings] == ["Done"]
    assert err == ""
    assert any(isinstance(mg, m.ToolMessage) for mg in trace_msgs)

@pytest.mark.asyncio
async def test_no_decision_fallback(mocker, agent, span_double):
    agent.tool_manager = m.ToolManager({"T1": "D"})
    mocker.patch.object(m.trace, "get_current_span", return_value=span_double)
    logger = mocker.patch.object(m, "logger")

    class Runnable:
        async def ainvoke(self, data, config=None):
            return SimpleNamespace(tool_calls=[], invalid_tool_calls=[])

    mocker.patch.object(m, "create_agent_runnable", return_value=Runnable())

    sub_headings, err, trace_msgs = await agent._run_agent_loop(
        session_id="S", query="Q", tools=["T1"], parent_task="P",
    )

    assert sub_headings[0].text == "Research concluded without a final answer."
    assert "Agent loop completed" in err
    assert trace_msgs is None
    logger.warning.assert_called_once_with("Agent did not make a decision. Completing with no findings.")

@pytest.mark.asyncio
async def test_exception_inside_loop_returns_error_and_trace(mocker, agent, span_double):
    agent.tool_manager = m.ToolManager({"T1": "D"})
    mocker.patch.object(m.trace, "get_current_span", return_value=span_double)
    logger = mocker.patch.object(m, "logger")

    class Runnable:
        async def ainvoke(self, data, config=None):
            raise RuntimeError("boom")

    mocker.patch.object(m, "create_agent_runnable", return_value=Runnable())

    sub_headings, err, trace_msgs = await agent._run_agent_loop(
        session_id="S", query="Q", tools=["T1"], parent_task="P", return_messages_trace=True,
    )

    assert sub_headings[0].text == "An unexpected error occurred during research."
    assert "Agent loop failed with error: boom" in err
    assert trace_msgs and isinstance(trace_msgs[0], m.HumanMessage)
    _, kwargs = logger.error.call_args
    assert kwargs.get("exc_info") is True

@pytest.mark.asyncio
async def test_early_return_when_no_selected_tools(mocker, agent, span_double):
    agent.tool_manager = m.ToolManager({})
    mocker.patch.object(m.trace, "get_current_span", return_value=span_double)
    logger = mocker.patch.object(m, "logger")

    sub_headings, err, trace_msgs = await agent._run_agent_loop(
        session_id="S", query="Q", tools=["Nope"], parent_task="P",
    )

    assert sub_headings[0].text == "Research Process"
    assert "does not exists" in err
    assert trace_msgs is None
    logger.error.assert_called_once()

