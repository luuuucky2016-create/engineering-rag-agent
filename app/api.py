"""FastAPI service. The assistant is injected via a dependency, so tests swap in fakes."""

from __future__ import annotations

from functools import lru_cache
from uuid import uuid4

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from app import __version__
from app.agent import Assistant


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    thread_id: str | None = Field(None, description="Reuse to continue a conversation (memory).")


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    relevance: dict[str, float] = {}
    citations_removed: list[str] = []
    tools_used: list[str]
    thread_id: str


@lru_cache
def get_assistant() -> Assistant:  # built once, on first request
    from app.factory import build_assistant

    return build_assistant()


app = FastAPI(title="RAG + Tool-Calling Agent", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, assistant: Assistant = Depends(get_assistant)) -> ChatResponse:
    return ChatResponse(**assistant.ask(req.message, thread_id=req.thread_id or str(uuid4())))
