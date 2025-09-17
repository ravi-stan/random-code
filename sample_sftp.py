@pytest.mark.asyncio
async def test_research_agent_core_happy_path(monkeypatch):
    """
    - message is SubTaskType
    - Processor.execute_research returns a result
    - get_task_type called with parent_task_id and span attribute set
    - _send_result_to_planner called with (context, result, parent_task_id)
    """
    # Setup logger
    _setup_logger(monkeypatch, level=logging.DEBUG)

    # Prepare SubTaskType and message
    subtask_type = _TypeStub("sub_task/test.InternalMessage")
    monkeypatch.setattr(m, "SubTaskType", subtask_type, raising=False)

    sub_task = _SubTaskStub("ST-1", "PT-1", "S-1")
    internal = types.SimpleNamespace(session_id="S-1", sub_task=sub_task)
    msg = FakeMessage.from_value(internal, subtask_type)

    # Trace & task-type plumbing
    class _Span:
        def __init__(self):
            self.set_calls = []
        def set_attribute(self, k, v):
            self.set_calls.append((k, v))
    span = _Span()
    class _Trace:
        def get_current_span(self_inner):
            return span
    monkeypatch.setattr(m, "trace", _Trace(), raising=False)

    get_task_type_calls = {}
    def fake_get_task_type(pid):
        get_task_type_calls["arg"] = pid
        return "analysis"
    monkeypatch.setattr(m, "get_task_type", fake_get_task_type, raising=False)

    # Processor capture
    created = {}
    result_obj = object()
    class _ProcessorFake:
        def __init__(self, cache):
            created["cache"] = cache
            created["self"] = self
        async def execute_research(self, internal_msg, instruction=None, return_message_trace=False):
            created["exec_args"] = (internal_msg, instruction, return_message_trace)
            return (result_obj, object())
    monkeypatch.setattr(m, "Processor", _ProcessorFake, raising=False)

    # Planner sender capture
    sent = {}
    def fake_send(ctx, research_result, parent_task_id):
        sent["args"] = (ctx, research_result, parent_task_id)
    monkeypatch.setattr(m, "_send_result_to_planner", fake_send, raising=False)

    # Act
    ctx = object()
    cache = object()
    await m._research_agent_core(ctx, msg, cache_client=cache)

    # Assert processor usage
    assert created["cache"] is cache
    assert created["exec_args"] == (internal, None, False)

    # Assert span/task_type
    assert get_task_type_calls["arg"] == "PT-1"
    assert ("task_type", "analysis") in span.set_calls

    # Assert planner send
    assert sent["args"] == (ctx, result_obj, "PT-1")


@pytest.mark.asyncio
async def test_research_agent_core_unexpected_type_uses_fallback(monkeypatch, caplog):
    """
    - message.is_type(SubTaskType) -> False
    - _create_error_result returns None
    - _extract_fallback_result provides (fallback_result, pid)
    - finally sends fallback result
    """
    caplog.set_level(logging.ERROR)
    _setup_logger(monkeypatch, level=logging.ERROR)

    # Message stub with value_typename attribute for logging
    class _WrongTypeMessage:
        value_typename = "wrong/type"
        def is_type(self, _t): return False
        def as_type(self, _t): raise AssertionError("should not be called")
    msg = _WrongTypeMessage()

    # Force create_error_result to return None, so fallback is used
    monkeypatch.setattr(m, "_create_error_result", lambda *a, **k: None, raising=False)

    # Fallback result capture
    fallback_result = object()
    fb = {"called": False, "error_str": None}
    def fake_fallback(message, err):
        fb["called"] = True
        fb["error_str"] = str(err)
        return fallback_result, "PT-FB"
    monkeypatch.setattr(m, "_extract_fallback_result", fake_fallback, raising=False)

    sent = {}
    def fake_send(ctx, research_result, parent_task_id):
        sent["args"] = (ctx, research_result, parent_task_id)
    monkeypatch.setattr(m, "_send_result_to_planner", fake_send, raising=False)

    # Act (should not raise)
    ctx = object()
    await m._research_agent_core(ctx, msg, cache_client=None)

    # Assert: fallback used and send called with fallback payload
    assert fb["called"] is True
    assert "Unexpected message type" in fb["error_str"]
    assert sent["args"] == (ctx, fallback_result, "PT-FB")

    # Log messages present
    assert "Unexpected message type: wrong/type" in caplog.text
    assert "Error in research_agent: Unexpected message type: wrong/type" in caplog.text


@pytest.mark.asyncio
async def test_research_agent_core_processor_raises_uses_create_error(monkeypatch, caplog):
    """
    - message is SubTaskType
    - Processor.execute_research raises
    - _create_error_result returns error_result
    - _extract_fallback_result is NOT called
    - finally sends error_result
    """
    caplog.set_level(logging.ERROR)
    _setup_logger(monkeypatch, level=logging.ERROR)

    subtask_type = _TypeStub("sub_task/test.InternalMessage")
    monkeypatch.setattr(m, "SubTaskType", subtask_type, raising=False)

    sub_task = _SubTaskStub("ST-1", "PT-1", "S-1")
    internal = types.SimpleNamespace(session_id="S-1", sub_task=sub_task)
    msg = FakeMessage.from_value(internal, subtask_type)

    # Trace & task-type to ensure they're executed before processor
    class _Span:
        def __init__(self): self.set_calls = []
        def set_attribute(self, k, v): self.set_calls.append((k, v))
    span = _Span()
    class _Trace:
        def get_current_span(self_inner): return span
    monkeypatch.setattr(m, "trace", _Trace(), raising=False)
    monkeypatch.setattr(m, "get_task_type", lambda pid: "analysis", raising=False)

    # Processor that raises
    class _ProcessorBoom:
        def __init__(self, cache): pass
        async def execute_research(self, *a, **k): raise RuntimeError("boom")
    monkeypatch.setattr(m, "Processor", _ProcessorBoom, raising=False)

    # create_error_result returns this object
    error_result = object()
    cer = {"called": False}
    def fake_create(sub_msg, sess, parent, sid, e):
        cer["called"] = True
        # verify inputs are what we expect from the try-block
        assert sub_msg is sub_task
        assert sess == "S-1"
        assert parent == "PT-1"
        assert sid == "ST-1"
        assert str(e) == "boom"
        return error_result
    monkeypatch.setattr(m, "_create_error_result", fake_create, raising=False)

    # fallback should NOT be called in this branch
    def bad_fallback(*a, **k): raise AssertionError("fallback should not be called")
    monkeypatch.setattr(m, "_extract_fallback_result", bad_fallback, raising=False)

    sent = {}
    def fake_send(ctx, research_result, parent_task_id):
        sent["args"] = (ctx, research_result, parent_task_id)
    monkeypatch.setattr(m, "_send_result_to_planner", fake_send, raising=False)

    # Act
    ctx = object()
    await m._research_agent_core(ctx, msg, cache_client=None)

    # Assert
    assert cer["called"] is True
    assert sent["args"] == (ctx, error_result, "PT-1")
    assert "Error in research_agent: boom" in caplog.text


@pytest.mark.asyncio
async def test_research_agent_core_create_error_returns_none_then_fallback(monkeypatch):
    """
    - message is SubTaskType
    - Processor.execute_research raises
    - _create_error_result returns None
    - fallback returns (fallback_result, pid)
    - finally sends fallback result
    """
    subtask_type = _TypeStub("sub_task/test.InternalMessage")
    monkeypatch.setattr(m, "SubTaskType", subtask_type, raising=False)

    sub_task = _SubTaskStub("ST-2", "PT-2", "S-2")
    internal = types.SimpleNamespace(session_id="S-2", sub_task=sub_task)
    msg = FakeMessage.from_value(internal, subtask_type)

    # Minimal trace/task plumbing
    class _Span:
        def set_attribute(self, k, v): pass
    class _Trace:
        def get_current_span(self_inner): return _Span()
    monkeypatch.setattr(m, "trace", _Trace(), raising=False)
    monkeypatch.setattr(m, "get_task_type", lambda pid: "analysis", raising=False)

    class _ProcessorBoom:
        def __init__(self, cache): pass
        async def execute_research(self, *a, **k): raise RuntimeError("kaboom")
    monkeypatch.setattr(m, "Processor", _ProcessorBoom, raising=False)

    monkeypatch.setattr(m, "_create_error_result", lambda *a, **k: None, raising=False)

    fallback_result = object()
    def fake_fallback(message, e):
        return fallback_result, "PT-FB"
    monkeypatch.setattr(m, "_extract_fallback_result", fake_fallback, raising=False)

    sent = {}
    def fake_send(ctx, research_result, parent_task_id):
        sent["args"] = (ctx, research_result, parent_task_id)
    monkeypatch.setattr(m, "_send_result_to_planner", fake_send, raising=False)

    # Act
    ctx = object()
    await m._research_agent_core(ctx, msg, cache_client=None)

    # Assert
    assert sent["args"] == (ctx, fallback_result, "PT-FB")


@pytest.mark.asyncio
async def test_research_agent_core_finally_always_calls_send(monkeypatch):
    """
    Even if:
      - message type is unexpected
      - _create_error_result returns None
      - _extract_fallback_result returns (None, None)
    ...the finally block must still call _send_result_to_planner(context, None, None).
    """
    # Message stub with unexpected type
    class _WrongTypeMessage:
        value_typename = "wrong/type"
        def is_type(self, _t): return False
    msg = _WrongTypeMessage()

    # Both error constructors yield no result
    monkeypatch.setattr(m, "_create_error_result", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(m, "_extract_fallback_result", lambda *a, **k: (None, None), raising=False)

    called = {}
    def fake_send(ctx, research_result, parent_task_id):
        called["args"] = (ctx, research_result, parent_task_id)
    monkeypatch.setattr(m, "_send_result_to_planner", fake_send, raising=False)

    # Act
    ctx = object()
    await m._research_agent_core(ctx, msg, cache_client=None)

    # Assert: still called once with (None, None)
    assert called["args"] == (ctx, None, None)
