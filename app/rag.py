"""RAG pipeline: load -> chunk -> embed -> store (Chroma) -> retrieve -> cite."""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_SUFFIXES = {".md", ".txt"}


def load_documents(docs_dir: str | Path) -> list[Document]:
    """Read every .md/.txt file under docs_dir, tagging each with its source file name."""
    docs: list[Document] = []
    for path in sorted(Path(docs_dir).rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                docs.append(Document(page_content=text, metadata={"source": path.name}))
    return docs


def split_documents(docs: list[Document], chunk_size: int = 800, chunk_overlap: int = 120) -> list[Document]:
    """Chunk documents and give every chunk a stable id like 'file.md#2'."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    counters: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata["source"]
        counters[src] = counters.get(src, 0) + 1
        chunk.metadata["chunk_id"] = f"{src}#{counters[src]}"
    return chunks


def build_vector_store(
    chunks: list[Document],
    embeddings: Embeddings,
    persist_dir: str | None = None,
    collection_name: str = "knowledge_base",
) -> Chroma:
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},  # relevance score = cosine similarity
    )
    if chunks:
        ids = [c.metadata["chunk_id"] for c in chunks]
        store.add_documents(chunks, ids=ids)  # stable ids -> re-ingest is idempotent
    return store


def format_results(docs: list[Document], scores: list[float] | None = None) -> str:
    """Render retrieved chunks with [source] tags (and relevance, if known) so the model can cite them."""
    if not docs:
        return "NO_RESULTS: the knowledge base has nothing relevant to this query. Tell the user it is not covered."
    out = []
    for i, d in enumerate(docs):
        tag = f"[source: {d.metadata.get('chunk_id', d.metadata.get('source', '?'))}]"
        if scores is not None:
            tag += f" (relevance {scores[i]:.2f})"
        out.append(f"{tag}\n{d.page_content}")
    return "\n\n".join(out)
