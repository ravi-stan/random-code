# tests/conftest.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Callable

import pytest

# Fakes (chat + text) from LangChain core
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage

# ---- Utilities to build fake tool-calling messages (optional) ----

def make_tool_call_ai_message(
    *,
    tool_name: str,
    args: Dict[str, Any],
    content: str = "",
    call_id: str = "call-1",
) -> AIMessage:
    """Return an AIMessage that asks the executor to run a tool."""
    # Tool calls are standardized: [{'name': ..., 'args': {...}, 'id': ..., 'type': 'tool_call'}]
    # Type is optional; name & args are the key fields.
    return AIMessage(
        content=content,
        tool_calls=[{"name": tool_name, "args": args, "id": call_id, "type": "tool_call"}],
    )


# ---- Stub classes that mimic ChatVertexAI / VertexAI constructors ----

class StubChatVertexAI(FakeListChatModel):
    """
    Drop-in replacement for langchain_google_vertexai.ChatVertexAI.
    Accepts any kwargs the real class would, but ignores them.

    Usage control:
      - responses: list[str]  -> will be the model.text content
      - response_messages: list[AIMessage] -> if provided, use message-based stub (tool calls, etc.)
    """
    def __init__(
        self,
        *,
        responses: Optional[List[str]] = None,
        response_messages: Optional[List[AIMessage]] = None,
        **kwargs: Any,
    ) -> None:
        # store init args for assertions if needed
        self._init_kwargs = kwargs
        if response_messages is not None:
            # Swap behavior to messages-based fake when you need tool-calls / metadata
            self._messages_model = FakeMessagesListChatModel(responses=response_messages)
            super().__init__(responses=["__unused__"])
        else:
            self._messages_model = None
            super().__init__(responses=responses or ["CHAT_OK"])

    # record calls if you want to assert on inputs later
    calls: List[Dict[str, Any]] = []

    def invoke(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        self.calls.append({"input": input, "config": config, "kwargs": kwargs, "mode": "invoke"})
        if self._messages_model is not None:
            return self._messages_model.invoke(input, config=config, **kwargs)
        return super().invoke(input, config=config, **kwargs)

    async def ainvoke(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        self.calls.append({"input": input, "config": config, "kwargs": kwargs, "mode": "ainvoke"})
        if self._messages_model is not None:
            return await self._messages_model.ainvoke(input, config=config, **kwargs)
        return await super().ainvoke(input, config=config, **kwargs)


class StubVertexAI(FakeListLLM):
    """
    Drop-in replacement for langchain_google_vertexai.VertexAI (text LLM).
    """
    def __init__(self, *, responses: Optional[List[str]] = None, **kwargs: Any) -> None:
        self._init_kwargs = kwargs
        super().__init__(responses=responses or ["LLM_OK"])

    calls: List[Dict[str, Any]] = []

    def invoke(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        self.calls.append({"input": input, "config": config, "kwargs": kwargs, "mode": "invoke"})
        return super().invoke(input, config=config, **kwargs)

    async def ainvoke(self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        self.calls.append({"input": input, "config": config, "kwargs": kwargs, "mode": "ainvoke"})
        return await super().ainvoke(input, config=config, **kwargs)


# ---- pytest fixtures ----

@pytest.fixture
def patch_vertexai(monkeypatch: pytest.MonkeyPatch):
    """
    Patch *your* module’s imports of ChatVertexAI and VertexAI to these stubs.

    Usage:
        patch_vertexai("myapp.llm", chat_responses=["hello"], text_responses=["ok"])
        # or: patch_vertexai("myapp.llm", response_messages=[make_tool_call_ai_message(...)])

    Always patch the symbol *where it is used*, not the provider package directly.
    """
    def _patch(
        where: str,
        *,
        chat_responses: Optional[List[str]] = None,
        text_responses: Optional[List[str]] = None,
        response_messages: Optional[List[AIMessage]] = None,
    ):
        # Build class factories that bake in the responses you want for this test
        def _chat_ctor(**kwargs):
            return StubChatVertexAI(
                responses=chat_responses,
                response_messages=response_messages,
                **kwargs,
            )

        def _text_ctor(**kwargs):
            return StubVertexAI(responses=text_responses, **kwargs)

        # Typical imports in user code:
        #   from langchain_google_vertexai import ChatVertexAI, VertexAI
        #   or: from myapp.llm import ChatVertexAI
        monkeypatch.setattr(f"{where}.ChatVertexAI", _chat_ctor, raising=True)
        monkeypatch.setattr(f"{where}.VertexAI", _text_ctor, raising=True)

        return {"chat": _chat_ctor, "text": _text_ctor}

    return _patch



from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import myapp.llm  # <- this module does: `from langchain_google_vertexai import ChatVertexAI`

def test_gemini_chain_invoke(patch_vertexai):
    # Make our stub ChatVertexAI return a known string
    patch_vertexai("myapp.llm", chat_responses=["Hello from fake Gemini!"])

    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are helpful."), ("human", "Say hi to {who}.")]
    )
    model = myapp.llm.ChatVertexAI(model="gemini-1.5-pro")  # gets stub
    chain = prompt | model | StrOutputParser()

    out = chain.invoke({"who": "pytest"})
    assert out == "Hello from fake Gemini!"
