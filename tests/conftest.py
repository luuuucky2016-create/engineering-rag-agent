"""Offline fakes: no API keys, no network, no model downloads."""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from app.rag import build_vector_store, load_documents, split_documents

DOCS_DIR = "data/docs"


class KeywordEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing embeddings: shared words -> similar vectors."""

    dim = 256

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class ScriptedChatModel(BaseChatModel):
    """Returns pre-scripted AIMessages in order and records every prompt it was sent."""

    responses: list[AIMessage]
    calls: list[list[BaseMessage]] = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        self.calls.append(list(messages))
        msg = self.responses[len(self.calls) - 1]
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs: Any):
        return self

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"


def tool_call(name: str, args: dict, call_id: str | None = None) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id or uuid.uuid4().hex}])


@pytest.fixture
def embeddings() -> KeywordEmbeddings:
    return KeywordEmbeddings()


@pytest.fixture
def chunks():
    return split_documents(load_documents(DOCS_DIR), chunk_size=800, chunk_overlap=120)


@pytest.fixture
def store(chunks, embeddings):
    return build_vector_store(chunks, embeddings, collection_name=f"test-{uuid.uuid4().hex}")


@pytest.fixture
def checkpointer():
    return InMemorySaver()
