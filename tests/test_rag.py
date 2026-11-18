import uuid

from app.rag import build_vector_store, format_results
from app.tools import make_kb_search_tool


def test_chunks_carry_source_and_unique_ids(chunks):
    assert len(chunks) > 3
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert {c.metadata["source"] for c in chunks} == {"electrical_basics.md", "voltage_drop.md", "ip_ratings.md"}


def test_retrieval_finds_the_relevant_document(store):
    top = store.similarity_search("IP rating water jets dust tight", k=1)[0]
    assert top.metadata["source"] == "ip_ratings.md"


def test_kb_tool_returns_citable_sources(store):
    out = make_kb_search_tool(store, k=2).invoke({"query": "voltage drop formula three-phase cable"})
    assert "[source: voltage_drop.md#" in out


def test_empty_knowledge_base_is_explicit(embeddings):
    empty = build_vector_store([], embeddings, collection_name=f"empty-{uuid.uuid4().hex}")
    assert empty.similarity_search("anything", k=2) == []
    assert format_results([]).startswith("NO_RESULTS")


def test_low_relevance_results_are_filtered_out(store):
    out = make_kb_search_tool(store, k=3, min_relevance=0.99).invoke({"query": "boiling point of mercury"})
    assert out.startswith("NO_RESULTS")
