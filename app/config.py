"""Runtime settings, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "ollama"))  # ollama | openai
    chat_model: str = field(default_factory=lambda: _env("CHAT_MODEL", "qwen2.5:7b"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "nomic-embed-text"))
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    docs_dir: str = field(default_factory=lambda: _env("DOCS_DIR", "data/docs"))
    chroma_dir: str = field(default_factory=lambda: _env("CHROMA_DIR", ".chroma"))
    memory_db: str = field(default_factory=lambda: _env("MEMORY_DB", ".memory/checkpoints.sqlite"))
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "120")))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "3")))
    min_relevance: float = field(default_factory=lambda: float(_env("MIN_RELEVANCE", "0.6")))


def get_settings() -> Settings:
    return Settings()
