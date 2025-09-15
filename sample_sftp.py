# test_helpers/statefun_message_fake.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional

# A minimal interface for StateFun "Type" objects (e.g., from make_protobuf_type).
class _TypeLike:
    typename: str
    def serialize(self, value: Any) -> bytes: ...
    def deserialize(self, data: bytes) -> Any: ...

def _typename_of(t_or_name: Any) -> str:
    if isinstance(t_or_name, str):
        return t_or_name
    name = getattr(t_or_name, "typename", None)
    if not isinstance(name, str):
        raise TypeError(
            f"Expected a StateFun Type (with .typename) or a string typename, got: {type(t_or_name)}"
        )
    return name

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

    # ---------- API parity with StateFun.Message ----------

    def is_type(self, type_or_name: Any) -> bool:
        """True iff this message's typename matches the given StateFun Type or typename string."""
        return self._typename == _typename_of(type_or_name)

    def is_type_name(self, name: str) -> bool:
        return self._typename == name

    def value_type_name(self) -> str:
        return self._typename

    def as_type(self, type_obj: _TypeLike):
        """
        Decode the payload as the given StateFun Type.
        Mirrors Message.as_type semantics:
          - raises ValueError on type mismatch
          - uses Type.deserialize(...) to decode when needed
        """
        expected = _typename_of(type_obj)
        if self._typename != expected:
            raise ValueError(f"Expected message of type {expected!r}, got {self._typename!r}")

        if self._decoded is not None:
            return self._decoded

        if self._value_bytes is None:
            # Empty message, but the caller asked for a typed value.
            # StateFun would effectively error; we do the same.
            raise ValueError("Message has no value (empty), cannot decode as_type(...)")

        decoder = self._decoder or getattr(type_obj, "deserialize", None)
        if not callable(decoder):
            raise TypeError("No decoder available for this message/type pairing")

        self._decoded = decoder(self._value_bytes)
        return self._decoded

    # ---------- Convenience helpers for tests ----------

    def is_empty(self) -> bool:
        return self._decoded is None and (self._value_bytes is None or self._value_bytes == b"")

    def raw_value_bytes(self) -> Optional[bytes]:
        """Get the serialized bytes if present; will serialize from decoded if possible."""
        if self._value_bytes is not None:
            return self._value_bytes
        if self._decoded is not None and callable(self._encoder):
            self._value_bytes = self._encoder(self._decoded)
            return self._value_bytes
        return None

    # ---------- Constructors ----------

    @classmethod
    def from_value(cls, value: Any, value_type: _TypeLike) -> "FakeMessage":
        """
        Build a typed message from a Python object (e.g., a protobuf instance).
        Uses value_type.serialize(...) to populate bytes and value_type.deserialize(...) for decoding.
        """
        if not hasattr(value_type, "typename"):
            raise TypeError("value_type must be a StateFun Type with .typename/.serialize/.deserialize")
        try:
            value_bytes = value_type.serialize(value)
        except Exception as ex:
            raise TypeError(f"Failed to serialize value with provided type {value_type}: {ex}") from ex

        return cls(
            _typename=value_type.typename,
            _value_bytes=value_bytes,
            _decoded=value,
            _decoder=getattr(value_type, "deserialize", None),
            _encoder=getattr(value_type, "serialize", None),
        )

    @classmethod
    def from_bytes(
        cls,
        typename: str,
        payload_bytes: Optional[bytes],
        decoder: Optional[Callable[[bytes], Any]] = None,
        encoder: Optional[Callable[[Any], bytes]] = None,
    ) -> "FakeMessage":
        """
        Build a message from raw bytes. Provide decoder/encoder if you want as_type(...) to work.
        This is handy when you already have the serialized protobuf bytes.
        """
        return cls(
            _typename=typename,
            _value_bytes=payload_bytes,
            _decoded=None,
            _decoder=decoder,
            _encoder=encoder,
        )

    @classmethod
    def empty(cls, typename: str) -> "FakeMessage":
        """Build a message that carries no value (has_value == False in the wire protocol)."""
        return cls(_typename=typename, _value_bytes=None, _decoded=None)

