"""Model factories. Swap providers with LLM_PROVIDER without touching the agent code."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from app.config import Settings


def get_chat_model(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=settings.chat_model, base_url=settings.ollama_base_url, temperature=0)
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI  # needs OPENAI_API_KEY

        return ChatOpenAI(model=settings.chat_model, temperature=0)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.llm_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_base_url)
    if settings.llm_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embed_model)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")
