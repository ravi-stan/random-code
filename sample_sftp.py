import pytest
from unittest.mock import AsyncMock
from google.protobuf.message import Message  # only base class, NOT SubTask

# IMPORTANT: replace this with your actual module under test
import your_module as m


class FakeSpan:
    def __init__(self):
        self.calls = []
        self.attrs = {}
    def set_attribute(self, key, value):
        self.calls.append((key, value))
        self.attrs[key] = value


@pytest.fixture
def agent():
    # create the class instance that has execute_research
    # e.g., m.ResearchAgent() — adjust if your class name differs
    return m.ResearchAgent()


def build_internal_message(session_id: str = "sess-1",
                           sub_task_id: str = "SUB123",
                           parent_task_id: str = "TASK999",
                           instruction: str = "research topic X"):
    """
    Build a real InternalMessage and populate the nested SubTask WITHOUT
    importing SubTask. We mutate the nested message via `im.sub_task`.
    """
    im = m.InternalMessage()              # same symbol your module uses
    im.session_id = session_id

    # Access the nested message and set fields — no SubTask import needed.
    st = im.sub_task                      # this is a real SubTask protobuf instance
    st.sub_task_id = sub_task_id
    st.parent_task_id = parent_task_id
    st.instruction = instruction

    # If tools exists (repeated field), leave it as default to avoid type assumptions.
    # If you KNOW it's repeated string, you could do: st.tools.extend(["web", "code"])

    # If parent_task / caller_context are message fields, we don't need to set them;
    # we will assert identity forwarding of whatever default submessage object exists.
    return im


@pytest.mark.asyncio
async def test_success_flow_sets_span_and_returns_result(mocker, agent):
    internal_message = build_internal_message()

    # Sanity: SubTask is a real protobuf Message even though we didn't import it
    assert isinstance(internal_message.sub_task, Message)

    # Mocks
    fake_span = FakeSpan()
    mocker.patch.object(m, "set_session_id")
    mocker.patch.object(m.trace, "get_current_span", return_value=fake_span)
    mocker.patch.object(m, "get_task_type", return_value="RESEARCH")

    run_loop = mocker.patch.object(agent, "_run_agent_loop", new_callable=AsyncMock)
    # Keep sub_headings empty to avoid depending on its concrete type
    sub_headings = []
    llm_err_desc = None
    message_trace = [{"role": "system", "content": "trace"}]
    run_loop.return_value = (sub_headings, llm_err_desc, message_trace)

    # Act
    out_msg, messages_trace = await agent.execute_research(
        internal_message,
        instructions="override-instructions",
        return_messages_trace=True,
    )

    # Side-effect assertions
    m.set_session_id.assert_called_once_with("sess-1")
    m.get_task_type.assert_called_once_with("TASK999")
    assert ("task_id", "TASK999") in fake_span.calls
    assert ("task_type", "RESEARCH") in fake_span.calls

    # _run_agent_loop call signature — assert identity of nested fields w/o importing SubTask
    args, kwargs = run_loop.call_args
    assert args[0] == internal_message.session_id
    assert args[1] == internal_message.sub_task.instruction
    assert args[2] is internal_message.sub_task.tools            # identity, not content
    assert args[3] is internal_message.sub_task.parent_task      # identity, type-agnostic
    assert args[4] is internal_message.sub_task.caller_context   # identity, type-agnostic
    assert args[5] == "override-instructions"
    assert kwargs.get("return_message_trace") is True            # catches the name mismatch bug

    # Return object assertions
    assert isinstance(out_msg, m.InternalMessage)
    ir = out_msg.intermediate_result
    assert ir.success is True
    assert ir.sub_task_id == internal_message.sub_task.sub_task_id
    assert ir.parent_task_id == internal_message.sub_task.parent_task_id
    # Avoid strict equality on list-like fields across frameworks; just check length
    assert getattr(ir, "sub_headings") is not None
    assert len(ir.sub_headings) == 0
    # sub_task should be carried into the result; check value equality (protobuf supports ==)
    assert ir.sub_task == internal_message.sub_task
    # messages trace returned
    assert messages_trace == message_trace


@pytest.mark.asyncio
async def test_exception_flow_builds_failure_result_and_logs(mocker, agent):
    internal_message = build_internal_message(
        session_id="sess-err",
        sub_task_id="SUB999",
        parent_task_id="TASK000",
        instruction="will fail",
    )

    mocker.patch.object(m.trace, "get_current_span", return_value=FakeSpan())
    mocker.patch.object(m, "get_task_type", return_value="RESEARCH")
    mocker.patch.object(m, "set_session_id")

    logger_mock = mocker.patch.object(m, "logger")
    run_loop = mocker.patch.object(agent, "_run_agent_loop", new_callable=AsyncMock)
    run_loop.side_effect = RuntimeError("boom")

    out_msg, messages_trace = await agent.execute_research(internal_message)

    m.set_session_id.assert_called_once_with("sess-err")
    logger_mock.error.assert_called_once()
    msg_text = logger_mock.error.call_args.args[0]
    kwargs = logger_mock.error.call_args.kwargs
    assert "Research failed for sub-task SUB999: boom" in msg_text
    assert kwargs.get("exc_info") is True

    ir = out_msg.intermediate_result
    assert ir.success is False
    assert ir.sub_task_id == "SUB999"
    assert ir.parent_task_id == "TASK000"
    assert getattr(ir, "sub_headings") is not None
    assert len(ir.sub_headings) == 1
    # We only depend on the well-known error text from your function
    assert getattr(ir.sub_headings[0], "text") == "An unexpected error occurred during research."
    assert messages_trace is None


@pytest.mark.asyncio
async def test_argument_forwarding_defaults(mocker, agent):
    internal_message = build_internal_message(session_id="s-1", sub_task_id="S1", parent_task_id="P1", instruction="instr")

    mocker.patch.object(m.trace, "get_current_span", return_value=FakeSpan())
    mocker.patch.object(m, "get_task_type", return_value="RESEARCH")
    mocker.patch.object(m, "set_session_id")
    run_loop = mocker.patch.object(agent, "_run_agent_loop", new_callable=AsyncMock)
    run_loop.return_value = ([], None, None)

    await agent.execute_research(internal_message)  # use defaults

    args, kwargs = run_loop.call_args
    # instructions default is None
    assert args[5] is None
    # return_messages_trace default False must be forwarded as return_message_trace=False
    assert kwargs.get("return_message_trace") is False

