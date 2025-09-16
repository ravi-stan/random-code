# tests/unit/test_research_planner.py
import logging
import types
from unittest.mock import MagicMock
import importlib
import pytest
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

# -----------------------------------------------------------------------------
# Set this to your actual module path that defines the three functions:
#   _send_result_to_planner, _extract_fallback_result, _create_error_result
# Examples: "app.research.planner" or "planner"
MODULE_UNDER_TEST = "planner"  # <-- CHANGE ME
# -----------------------------------------------------------------------------

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
    monkeypatch.setattr(m, "logger", test_logger, raising=False)
    return logger_name


class DummyInternalMessage:
    """Simple stand‑in payload for _send_result_to_planner success-path tests."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture
def dummy_result():
    # The function passes this object straight to message_builder(value=...),
    # so a plain object is sufficient.
    return DummyInternalMessage(some="payload")


# =============================================================================
# Tests for: _send_result_to_planner
# =============================================================================

def test_send_result_success(monkeypatch, caplog, dummy_result):
    logger_name = _setup_logger(monkeypatch, level=logging.INFO)
    caplog.set_level(logging.INFO, logger=logger_name)

    sent_sentinel = object()
    builder_mock = MagicMock(return_value=sent_sentinel)
    monkeypatch.setattr(m, "message_builder", builder_mock, raising=False)

    IR_TYPE = object()  # value_type should be passed through verbatim
    monkeypatch.setattr(m, "IntermediateResultType", IR_TYPE, raising=False)

    ctx = types.SimpleNamespace(send=MagicMock())

    m._send_result_to_planner(ctx, dummy_result, parent_task_id="PT-123")

    # message_builder called with exact kwargs (value is the InternalMessage object)
    assert builder_mock.call_count == 1
    kwargs = builder_mock.call_args.kwargs
    assert kwargs["target_typename"] == "agent-assist/research-planner"
    assert kwargs["target_id"] == "PT-123"
    assert kwargs["value"] is dummy_result
    assert kwargs["value_type"] is IR_TYPE

    # send invoked once with builder's return value
    ctx.send.assert_called_once_with(sent_sentinel)

    # info log present
    assert any(
        rec.levelno == logging.INFO and "Research complete. Sending intermediate results" in rec.getMessage()
        for rec in caplog.records
    )

@pytest.mark.parametrize(
    "research_result,parent_task_id",
    [
        (None, "PT-1"),
        (DummyInternalMessage(), None),
        (None, None),
    ]
)
def test_send_result_early_return_logs_error_and_no_send(monkeypatch, caplog, research_result, parent_task_id):
    logger_name = _setup_logger(monkeypatch, level=logging.ERROR)
    caplog.set_level(logging.ERROR, logger=logger_name)

    # If builder is called in this branch, fail the test
    monkeypatch.setattr(m, "message_builder", MagicMock(side_effect=AssertionError("should not be called")), raising=False)

    ctx = types.SimpleNamespace(send=MagicMock())

    m._send_result_to_planner(ctx, research_result, parent_task_id)

    ctx.send.assert_not_called()
    assert any(
        rec.levelno == logging.ERROR and "CRITICAL: Could not send result to Research Planner" in rec.getMessage()
        for rec in caplog.records
    )

def test_send_result_logs_error_if_builder_raises(monkeypatch, caplog, dummy_result):
    logger_name = _setup_logger(monkeypatch, level=logging.ERROR)
    caplog.set_level(logging.ERROR, logger=logger_name)

    builder_mock = MagicMock(side_effect=RuntimeError("builder boom"))
    monkeypatch.setattr(m, "message_builder", builder_mock, raising=False)

    IR_TYPE = object()
    monkeypatch.setattr(m, "IntermediateResultType", IR_TYPE, raising=False)

    ctx = types.SimpleNamespace(send=MagicMock())

    m._send_result_to_planner(ctx, dummy_result, parent_task_id="PT-9")

    ctx.send.assert_not_called()
    assert any(
        rec.levelno == logging.ERROR
        and "Failed to send result to Research Planner: builder boom" in rec.getMessage()
        and rec.exc_info is not None
        for rec in caplog.records
    )

def test_send_result_logs_error_if_context_send_raises(monkeypatch, caplog, dummy_result):
    logger_name = _setup_logger(monkeypatch, level=logging.ERROR)
    caplog.set_level(logging.ERROR, logger=logger_name)

    sent_sentinel = object()
    builder_mock = MagicMock(return_value=sent_sentinel)
    monkeypatch.setattr(m, "message_builder", builder_mock, raising=False)

    IR_TYPE = object()
    monkeypatch.setattr(m, "IntermediateResultType", IR_TYPE, raising=False)

    send_mock = MagicMock(side_effect=RuntimeError("network down"))
    ctx = types.SimpleNamespace(send=send_mock)

    m._send_result_to_planner(ctx, dummy_result, parent_task_id="PT-42")

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
    def serialize(self, value) -> bytes:
        return b"x"
    def deserialize(self, data: bytes):
        raise NotImplementedError

class _ConstructedIntermediateResult:
    """Captures constructor args to assert on them."""
    def __init__(self, sub_task_id, parent_task_id, success, error_description, sub_task=None):
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
    def __init__(self):
        self.session_id = "S-err"
    def is_type(self, _t) -> bool:
        return True
    def as_type(self, _t):
        raise RuntimeError("decode fail")

class _SubTaskStub:
    def __init__(self, sub_task_id: str, parent_task_id: str, session_id: str):
        self.sub_task_id = sub_task_id
        self.parent_task_id = parent_task_id
        self.session_id = session_id

def test_extract_fallback_result_success(monkeypatch, caplog):
    logger_name = _setup_logger(monkeypatch, level=logging.ERROR)
    caplog.set_level(logging.ERROR, logger=logger_name)

    # Patch constructors used by the function
    monkeypatch.setattr(m, "IntermediateResult", _ConstructedIntermediateResult, raising=False)
    monkeypatch.setattr(m, "InternalMessage", _ConstructedInternalMessage, raising=False)

    # SubTaskType expected by function
    subtask_type = _TypeStub("sub_task/Acme.InternalMessage")
    monkeypatch.setattr(m, "SubTaskType", subtask_type, raising=False)

    # The decoded "internal_message" must expose a .sub_task with required fields
    fallback_sub_task = _SubTaskStub(sub_task_id="ST-88", parent_task_id="PT-77", session_id="S-22")
    decoded_internal = types.SimpleNamespace(sub_task=fallback_sub_task)

    msg = FakeMessage.from_value(decoded_internal, subtask_type)

    research_result, parent_id = m._extract_fallback_result(msg, error=RuntimeError("disk I/O"))

    # Validate wrapper & fields
    assert isinstance(research_result, _ConstructedInternalMessage)
    assert research_result.session_id == "S-22"  # taken from sub_task.session_id
    ir = research_result.intermediate_result
    assert isinstance(ir, _ConstructedIntermediateResult)
    assert ir.sub_task_id == "ST-88"
    assert ir.parent_task_id == "PT-77"
    assert ir.success is False
    assert ir.sub_task is fallback_sub_task
    assert ir.error_description == "Critical error in research_agent: disk I/O"
    assert parent_id == "PT-77"

    # No error logs on happy path
    assert not any(rec.levelno >= logging.ERROR for rec in caplog.records)

def test_extract_fallback_result_returns_none_when_wrong_type(monkeypatch, caplog):
    logger_name = _setup_logger(monkeypatch, level=logging.ERROR)
    caplog.set_level(logging.ERROR, logger=logger_name)

    expected = _TypeStub("sub_task/Expected")
    monkeypatch.setattr(m, "SubTaskType", expected, raising=False)

    # Message created with a different type -> is_type(...) False
    other_type = _TypeStub("other/Type")
    decoded_internal = types.SimpleNamespace(sub_task=_SubTaskStub("ST-x", "PT-x", "S-x"))
    msg = FakeMessage.from_value(decoded_internal, other_type)

    research_result, parent_id = m._extract_fallback_result(msg, error=RuntimeError("boom"))
    assert research_result is None and parent_id is None

    # Should not log error for this branch
    assert not any(
        rec.levelno >= logging.ERROR and "Failed to create fallback error result" in rec.getMessage()
        for rec in caplog.records
    )

def test_extract_fallback_result_logs_error_when_as_type_raises(monkeypatch, caplog):
    logger_name = _setup_logger(monkeypatch, level=logging.ERROR)
    caplog.set_level(logging.ERROR, logger=logger_name)

    monkeypatch.setattr(m, "SubTaskType", _TypeStub("sub_task/Any"), raising=False)

    msg = _BoomMessage()

    research_result, parent_id = m._extract_fallback_result(msg, error=ValueError("decode fail"))
    assert research_result is None and parent_id is None

    assert any(
        rec.levelno == logging.ERROR
        and "Failed to create fallback error result: decode fail" in rec.getMessage()
        and rec.exc_info is not None
        for rec in caplog.records
    )

def test_extract_fallback_result_logs_error_when_missing_sub_task(monkeypatch, caplog):
    logger_name = _setup_logger(monkeypatch, level=logging.ERROR)
    caplog.set_level(logging.ERROR, logger=logger_name)

    subtask_type = _TypeStub("sub_task/Acme.InternalMessage")
    monkeypatch.setattr(m, "SubTaskType", subtask_type, raising=False)

    # Decoded object missing 'sub_task' -> AttributeError inside function
    decoded_internal = types.SimpleNamespace()
    msg = FakeMessage.from_value(decoded_internal, subtask_type)

    research_result, parent_id = m._extract_fallback_result(msg, error=RuntimeError("oops"))
    assert research_result is None and parent_id is None

    assert any(
        rec.levelno == logging.ERROR
        and "Failed to create fallback error result:" in rec.getMessage()
        and rec.exc_info is not None
        for rec in caplog.records
    )


# =============================================================================
# Tests for: _create_error_result
# =============================================================================

def test_create_error_result_with_subtask(monkeypatch):
    # Patch constructors used by the function
    monkeypatch.setattr(m, "IntermediateResult", _ConstructedIntermediateResult, raising=False)
    monkeypatch.setattr(m, "InternalMessage", _ConstructedInternalMessage, raising=False)

    sub_task = _SubTaskStub("ST-101", "PT-202", "S-303")

    result = m._create_error_result(
        sub_task_message=sub_task,
        session_id="IGNORED",
        parent_task_id="IGNORED",
        sub_task_id="IGNORED",
        error=RuntimeError("boom"),
    )

    assert isinstance(result, _ConstructedInternalMessage)
    assert result.session_id == "S-303"  # taken from sub_task_message.session_id
    ir = result.intermediate_result
    assert isinstance(ir, _ConstructedIntermediateResult)
    assert ir.sub_task_id == "ST-101"
    assert ir.parent_task_id == "PT-202"
    assert ir.success is False
    assert ir.sub_task is sub_task
    assert ir.error_description == "An unexpected error occurred in research_agent: boom"

def test_create_error_result_ids_only_branch(monkeypatch):
    monkeypatch.setattr(m, "IntermediateResult", _ConstructedIntermediateResult, raising=False)
    monkeypatch.setattr(m, "InternalMessage", _ConstructedInternalMessage, raising=False)

    result = m._create_error_result(
        sub_task_message=None,
        session_id="S-1",
        parent_task_id="PT-1",
        sub_task_id="ST-1",
        error=ValueError("oops"),
    )

    assert isinstance(result, _ConstructedInternalMessage)
    assert result.session_id == "S-1"
    ir = result.intermediate_result
    assert isinstance(ir, _ConstructedIntermediateResult)
    assert ir.sub_task_id == "ST-1"
    assert ir.parent_task_id == "PT-1"
    assert ir.success is False
    assert ir.sub_task is None
    assert ir.error_description == "An unexpected error occurred in research_agent: oops"

@pytest.mark.parametrize(
    "session_id,parent_task_id,sub_task_id",
    [
        (None, "PT-1", "ST-1"),
        ("S-1", None, "ST-1"),
        ("S-1", "PT-1", None),
        (None, None, None),
    ]
)
def test_create_error_result_insufficient_data_returns_none(session_id, parent_task_id, sub_task_id):
    result = m._create_error_result(
        sub_task_message=None,
        session_id=session_id,
        parent_task_id=parent_task_id,
        sub_task_id=sub_task_id,
        error=RuntimeError("anything"),
    )
    assert result is None

