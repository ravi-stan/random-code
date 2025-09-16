# tests/unit/test_research_planner.py
import logging
import types
from unittest.mock import MagicMock
import importlib
import pytest
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Set this to your actual module path that defines _send_result_to_planner
# and _extract_fallback_result. Example: "app.research.planner" or "planner"
MODULE_UNDER_TEST = "planner"  # <-- CHANGE ME to your module's import path
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Import the module under test as "m"
m = importlib.import_module(MODULE_UNDER_TEST)


# =============================================================================
# Embedded FakeMessage helper (no external imports needed)
# =============================================================================

def _typename_of(t_or_name: Any) -> str:
    if isinstance(t_or_name, str):
        return t_or_name
    name = getattr(t_or_name, "typename", None)
    if not isinstance(name, str):
        raise TypeError("Expected a StateFun Type (with .typename) or a typename string")
    return name


def _serdes_from(value_type: Any) -> Tuple[str, Callable[[Any], bytes], Callable[[bytes], Any]]:
    """
    Accepts both modern (.serialize/.deserialize) and legacy (.serializer.serialize/.serializer.deserialize)
    style StateFun types.
    """
    typename = _typename_of(value_type)

    ser = getattr(value_type, "serialize", None)
    de = getattr(value_type, "deserialize", None)
    if callable(ser) and callable(de):
        return typename, ser, de

    s = getattr(value_type, "serializer", None)
    if s and callable(getattr(s, "serialize", None)) and callable(getattr(s, "deserialize", None)):
        return typename, s.serialize, s.deserialize

    raise TypeError(
        "value_type must provide serialize/deserialize or serializer.serialize/serializer.deserialize"
    )


@dataclass
class FakeMessage:
    """
    A robust mock for Flink StateFun's Message used in Python handlers.

    Supports:
      - is_type(type_or_name)
      - is_type_name(name)
      - as_type(type_obj)
      - value_type_name() -> str
      - is_empty() -> bool
    """
    _typename: str
    _value_bytes: Optional[bytes] = None
    _decoded: Optional[Any] = None
    _decoder: Optional[Callable[[bytes], Any]] = None
    _encoder: Optional[Callable[[Any], bytes]] = None

    # ----- Message API -----
    def is_type(self, type_or_name: Any) -> bool:
        return self._typename == _typename_of(type_or_name)

    def is_type_name(self, name: str) -> bool:
        return self._typename == name

    def value_type_name(self) -> str:
        return self._typename

    def as_type(self, type_obj: Any):
        expected, _enc, dec = _serdes_from(type_obj)
        if self._typename != expected:
            raise ValueError(f"Expected message of type {expected!r}, got {self._typename!r}")
        if self._decoded is not None:
            return self._decoded
        if self._value_bytes is None:
            raise ValueError("Message has no value (empty), cannot decode as_type(...)")
        self._decoded = dec(self._value_bytes)
        return self._decoded

    # ----- Helpers -----
    def is_empty(self) -> bool:
        return self._decoded is None and (self._value_bytes is None or self._value_bytes == b"")

    def raw_value_bytes(self) -> Optional[bytes]:
        if self._value_bytes is not None:
            return self._value_bytes
        if self._decoded is not None and callable(self._encoder):
            self._value_bytes = self._encoder(self._decoded)
            return self._value_bytes
        return None

    # ----- Constructors -----
    @classmethod
    def from_value(cls, value: Any, value_type: Any) -> "FakeMessage":
        typename, enc, dec = _serdes_from(value_type)
        try:
            b = enc(value)
        except Exception as ex:
            raise TypeError(f"Failed to serialize value with provided type {value_type}: {ex}") from ex
        return cls(_typename=typename, _value_bytes=b, _decoded=value, _decoder=dec, _encoder=enc)

    @classmethod
    def from_bytes(
        cls,
        typename: str,
        payload_bytes: Optional[bytes],
        decoder: Optional[Callable[[bytes], Any]] = None,
        encoder: Optional[Callable[[Any], bytes]] = None,
    ) -> "FakeMessage":
        return cls(_typename=typename, _value_bytes=payload_bytes, _decoded=None,
                   _decoder=decoder, _encoder=encoder)

    @classmethod
    def empty(cls, typename: str) -> "FakeMessage":
        return cls(_typename=typename, _value_bytes=None, _decoded=None)


# =============================================================================
# Common test utilities / fixtures
# =============================================================================

def _setup_logger(monkeypatch, level=logging.INFO):
    """
    Replace the module's logger with a dedicated test logger so caplog can capture records.
    """
    logger_name = "test.research_planner"
    test_logger = logging.getLogger(logger_name)
    test_logger.propagate = True   # let caplog capture it
    test_logger.setLevel(level)
    monkeypatch.setattr(m, "logger", test_logger, raising=True)
    return logger_name


class DummyInternalMessage:
    """Simple stand‑in for generic payloads in _send_result_to_planner tests."""
    def __init__(self, content="ok", sub_task=None):
        self.content = content
        self.sub_task = sub_task


@pytest.fixture
def dummy_msg():
    return DummyInternalMessage("hello")


# =============================================================================
# Tests for: _send_result_to_planner
# =============================================================================

def test_success_sends_message_and_logs_info(monkeypatch, caplog, dummy_msg):
    logger_name = _setup_logger(monkeypatch, level=logging.INFO)
    caplog.set_level(logging.INFO, logger=logger_name)

    # Arrange
    sent_sentinel = object()
    builder_mock = MagicMock(return_value=sent_sentinel)
    # Use raising=False to be robust even if the symbol wasn't imported at module top-level
    monkeypatch.setattr(m, "message_builder", builder_mock, raising=False)

    # Use a sentinel for IntermediateResultType; the function should pass this through
    IR_TYPE = object()
    monkeypatch.setattr(m, "IntermediateResultType", IR_TYPE, raising=False)

    # Context with a send mock
    ctx = types.SimpleNamespace(send=MagicMock())

    # Act
    m._send_result_to_planner(ctx, dummy_msg, parent_task_id="PT-123")

    # Assert builder called with exact kwargs
    assert builder_mock.call_count == 1
    kwargs = builder_mock.call_args.kwargs
    assert kwargs["target_typename"] == "agent-assist/research-planner"
    assert kwargs["target_id"] == "PT-123"
    assert kwargs["value"] is dummy_msg
    assert kwargs["value_type"] is IR_TYPE

    # Assert send called once with the builder's return value
    ctx.send.assert_called_once_with(sent_sentinel)

    # Assert info log
    assert any(
        rec.levelno == logging.INFO and "Research complete. Sending intermediate results" in rec.getMessage()
        for rec in caplog.records
    )


@pytest.mark.parametrize(
    "research_result,parent_task_id",
    [
        (None, "PT-1"),                      # missing result
        (DummyInternalMessage("x"), None),   # missing parent id
        (None, None),                        # both missing
    ]
)
def test_early_return_logs_error_and_does_not_send(monkeypatch, caplog, research_result, parent_task_id):
    logger_name = _setup_logger(monkeypatch, level=logging.ERROR)
    caplog.set_level(logging.ERROR, logger=logger_name)

    # If message_builder were called, fail the test (it should not be)
    monkeypatch.setattr(
        m, "message_builder", MagicMock(side_effect=AssertionError("message_builder should not be called")),
        raising=False
    )

    ctx = types.SimpleNamespace(send=MagicMock())

    # Act
    m._send_result_to_planner(ctx, research_result, parent_task_id)

    # Assert: no send, error log present
    ctx.send.assert_not_called()
    assert any(
        rec.levelno == logging.ERROR and "CRITICAL: Could not send result to Research Planner" in rec.getMessage()
        for rec in caplog.records
    )


def test_logs_error_if_message_builder_raises(monkeypatch, caplog, dummy_msg):
    logger_name = _setup_logger(monkeypatch, level=logging.ERROR)
    caplog.set_level(logging.ERROR, logger=logger_name)

    # Arrange: builder throws
    builder_mock = MagicMock(side_effect=RuntimeError("builder boom"))
    monkeypatch.setattr(m, "message_builder", builder_mock, raising=False)

    IR_TYPE = object()
    monkeypatch.setattr(m, "IntermediateResultType", IR_TYPE, raising=False)

    ctx = types.SimpleNamespace(send=MagicMock())

    # Act
    m._send_result_to_planner(ctx, dummy_msg, parent_task_id="PT-9")

    # Assert: send not called, error logged with exc_info
    ctx.send.assert_not_called()
    assert any(
        rec.levelno == logging.ERROR
        and "Failed to send result to Research Planner: builder boom" in rec.getMessage()
        and rec.exc_info is not None
        for rec in caplog.records
    )


def test_logs_error_if_context_send_raises(monkeypatch, caplog, dummy_msg):
    logger_name = _setup_logger(monkeypatch, level=logging.ERROR)
    caplog.set_level(logging.ERROR, logger=logger_name)

    # Arrange: builder OK, but send raises
    sent_sentinel = object()
    builder_mock = MagicMock(return_value=sent_sentinel)
    monkeypatch.setattr(m, "message_builder", builder_mock, raising=False)

    IR_TYPE = object()
    monkeypatch.setattr(m, "IntermediateResultType", IR_TYPE, raising=False)

    send_mock = MagicMock(side_effect=RuntimeError("network down"))
    ctx = types.SimpleNamespace(send=send_mock)

    # Act
    m._send_result_to_planner(ctx, dummy_msg, parent_task_id="PT-42")

    # Assert: send attempted once, error logged with exc_info
    send_mock.assert_called_once_with(sent_sentinel)
    assert any(
        rec.levelno == logging.ERROR
        and "Failed to send result to Research Planner: network down" in rec.getMessage()
        and rec.exc_info is not None
        for rec in caplog.records
    )


# =============================================================================
# Tests for: _extract_fallback_result
# =============================================================================

class _TypeStub:
    """Minimal StateFun Type stub (typename + serialize/deserialize)."""
    def __init__(self, typename: str):
        self.typename = typename
    def serialize(self, value) -> bytes:   # only needed by FakeMessage.from_value
        return b"x"
    def deserialize(self, data: bytes):    # not used because FakeMessage caches decoded value
        raise NotImplementedError


class _ConstructedIntermediateResult:
    """Captures constructor args to assert on them."""
    def __init__(self, sub_task_id, parent_task_id, success, error_description, sub_task):
        self.sub_task_id = sub_task_id
        self.parent_task_id = parent_task_id
        self.success = success
        self.error_description = error_description
        self.sub_task = sub_task


class _ConstructedInternalMessage:
    """Captures constructor args to assert on them."""
    def __init__(self, session_id, intermediate_result):
        self.session_id = session_id
        self.intermediate_result = intermediate_result


class _BoomMessage:
    """A message that 'matches' the type but raises on as_type()."""
    def __init__(self, session_id="S-err"):
        self.session_id = session_id
    def is_type(self, _t) -> bool:
        return True
    def as_type(self, _t):
        raise RuntimeError("decode fail")


def test_extract_fallback_result_success_builds_and_returns(monkeypatch, caplog):
    logger_name = _setup_logger(monkeypatch, level=logging.ERROR)
    caplog.set_level(logging.ERROR, logger=logger_name)

    # Arrange: patch model classes used inside the function
    monkeypatch.setattr(m, "IntermediateResult", _ConstructedIntermediateResult, raising=False)
    monkeypatch.setattr(m, "InternalMessage", _ConstructedInternalMessage, raising=False)

    # Set the SubTaskType expected by the function
    subtask_type = _TypeStub("sub_task/Acme.InternalMessage")
    monkeypatch.setattr(m, "SubTaskType", subtask_type, raising=False)

    # Build a message whose payload decodes to an object with a sub_task
    fallback_sub_task = types.SimpleNamespace(sub_task_id="ST-88", parent_task_id="PT-77")
    decoded_internal = types.SimpleNamespace(sub_task=fallback_sub_task)

    msg = FakeMessage.from_value(decoded_internal, subtask_type)
    msg.session_id = "S-22"  # the function reads message.session_id for the wrapper

    # Act
    research_result, parent_id = m._extract_fallback_result(msg, error=RuntimeError("disk I/O"))

    # Assert: wrapper built and fields copied correctly
    assert isinstance(research_result, _ConstructedInternalMessage)
    assert research_result.session_id == "S-22"
    assert isinstance(research_result.intermediate_result, _ConstructedIntermediateResult)
    ir = research_result.intermediate_result
    assert ir.sub_task_id == "ST-88"
    assert ir.parent_task_id == "PT-77"
    assert ir.success is False
    assert ir.sub_task is fallback_sub_task
    assert ir.error_description == "Critical error in research_agent: disk I/O"
    assert parent_id == "PT-77"

    # No error logs on the happy path
    assert not any(rec.levelno >= logging.ERROR for rec in caplog.records)


def test_extract_fallback_result_returns_none_when_wrong_type(monkeypatch, caplog):
    logger_name = _setup_logger(monkeypatch, level=logging.ERROR)
    caplog.set_level(logging.ERROR, logger=logger_name)

    # Function expects this type
    expected_type = _TypeStub("sub_task/Expected")
    monkeypatch.setattr(m, "SubTaskType", expected_type, raising=False)

    # Message is of a different type => is_type(...) is False => early (None, None)
    other_type = _TypeStub("other/Type")
    decoded_internal = types.SimpleNamespace(sub_task=types.SimpleNamespace(
        sub_task_id="ST-x", parent_task_id="PT-x"
    ))
    msg = FakeMessage.from_value(decoded_internal, other_type)
    msg.session_id = "S-ignored"

    research_result, parent_id = m._extract_fallback_result(msg, error=RuntimeError("boom"))
    assert research_result is None and parent_id is None

    # Should not log an error in this branch
    assert not any(
        rec.levelno >= logging.ERROR and "Failed to create fallback error result" in rec.getMessage()
        for rec in caplog.records
    )


def test_extract_fallback_result_logs_error_when_as_type_raises(monkeypatch, caplog):
    logger_name = _setup_logger(monkeypatch, level=logging.ERROR)
    caplog.set_level(logging.ERROR, logger=logger_name)

    # Set expected type (value irrelevant since _BoomMessage.is_type returns True)
    monkeypatch.setattr(m, "SubTaskType", _TypeStub("sub_task/Any"), raising=False)

    # Message will raise during as_type(...)
    msg = _BoomMessage(session_id="S-err")

    research_result, parent_id = m._extract_fallback_result(msg, error=ValueError("decode fail"))
    assert research_result is None and parent_id is None

    # Error logged with traceback
    assert any(
        rec.levelno == logging.ERROR
        and "Failed to create fallback error result: decode fail" in rec.getMessage()
        and rec.exc_info is not None
        for rec in caplog.records
    )


def test_extract_fallback_result_logs_error_when_missing_sub_task(monkeypatch, caplog):
    logger_name = _setup_logger(monkeypatch, level=logging.ERROR)
    caplog.set_level(logging.ERROR, logger=logger_name)

    # Function expects this type
    subtask_type = _TypeStub("sub_task/Acme.InternalMessage")
    monkeypatch.setattr(m, "SubTaskType", subtask_type, raising=False)

    # Decode yields an object with NO 'sub_task' attribute -> AttributeError inside the function
    decoded_internal = types.SimpleNamespace()  # missing 'sub_task'
    msg = FakeMessage.from_value(decoded_internal, subtask_type)
    msg.session_id = "S-33"

    research_result, parent_id = m._extract_fallback_result(msg, error=RuntimeError("oops"))
    assert research_result is None and parent_id is None

    # Error logged with traceback
    assert any(
        rec.levelno == logging.ERROR
        and "Failed to create fallback error result:" in rec.getMessage()
        and rec.exc_info is not None
        for rec in caplog.records
    )

