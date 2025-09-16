def test_send_result_logs_error_if_context_send_raises(monkeypatch, caplog, dummy_result):
    # Capture all ERROR logs regardless of the logger name
    caplog.set_level(logging.ERROR)

    # Ensure the module uses our test logger (propagating) so logs reach caplog
    _setup_logger(monkeypatch, level=logging.ERROR)

    # Arrange: builder returns a sentinel; context.send raises
    sent_sentinel = object()
    builder_mock = MagicMock(return_value=sent_sentinel)
    monkeypatch.setattr(m, "message_builder", builder_mock, raising=False)

    IR_TYPE = object()
    monkeypatch.setattr(m, "IntermediateResultType", IR_TYPE, raising=False)

    send_mock = MagicMock(side_effect=RuntimeError("network down"))
    ctx = types.SimpleNamespace(send=send_mock)

    # Act
    m._send_result_to_planner(ctx, dummy_result, parent_task_id="PT-42")

    # Assert: send attempted exactly once with the builder's result
    assert send_mock.call_count == 1
    assert send_mock.call_args.args == (sent_sentinel,)

    # Assert: an error log with the exception message is present
    assert "Failed to send result to Research Planner: network down" in caplog.text

