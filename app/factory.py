"""Wires real components together (models, vector store, memory) from Settings."""

from __future__ import annotations

from app.agent import Assistant, build_agent, sqlite_checkpointer
from app.config import Settings, get_settings
from app.llm import get_chat_model, get_embeddings
from app.rag import build_vector_store, load_documents, split_documents
from app.tools import calculator, make_kb_search_tool, voltage_drop


def build_assistant(settings: Settings | None = None) -> Assistant:
    s = settings or get_settings()
    chunks = split_documents(load_documents(s.docs_dir), s.chunk_size, s.chunk_overlap)
    store = build_vector_store(chunks, get_embeddings(s), persist_dir=s.chroma_dir, collection_name="knowledge_base_v2")
    kb = make_kb_search_tool(store, k=s.top_k, min_relevance=s.min_relevance)
    tools = [kb, voltage_drop, calculator]
    graph = build_agent(
        get_chat_model(s), tools, checkpointer=sqlite_checkpointer(s.memory_db),
        retrieve=lambda q: kb.invoke({"query": q}),  # ground every question first
    )
    return Assistant(graph)
