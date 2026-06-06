"""Shared Pinecone client and Gemini embedding utility for Robomate.

ONE Pinecone index, THREE namespaces ("recordings", "tasks", "collectors").

Embeddings use Gemini ``gemini-embedding-001`` at 768 dimensions (Matryoshka
truncation) via the ``google-genai`` SDK — the same SDK the rest of the backend
uses. The legacy ``google-generativeai`` package and ``text-embedding-004`` are
NOT used: that model is not available for this project's API key, and 768-dim
``gemini-embedding-001`` is what the live index was created with.

All heavy clients are created lazily and memoised so importing this module never
makes a network call, and tests can patch the factories.
"""

from __future__ import annotations

import os
from typing import Any

from google import genai
from google.genai import types as genai_types
from pinecone import Pinecone, ServerlessSpec

# gemini-embedding-001 supports configurable output dims; 768 matches the index.
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768

# The three namespaces inside the single index.
RECORDINGS_NAMESPACE = "recordings"
TASKS_NAMESPACE = "tasks"
COLLECTORS_NAMESPACE = "collectors"

_genai_client: Any | None = None
_pc_instance: Any | None = None
_index_instance: Any | None = None


def _get_genai_client() -> Any:
    global _genai_client
    if _genai_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY in the environment.")
        _genai_client = genai.Client(api_key=api_key)
    return _genai_client


def get_pinecone_index() -> Any:
    """Return a singleton Pinecone index client.

    Prefers PINECONE_INDEX_HOST to skip the control-plane round-trip; falls back
    to a name lookup when the host isn't configured.
    """
    global _pc_instance, _index_instance
    if _index_instance is None:
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("Set PINECONE_API_KEY in the environment.")
        _pc_instance = Pinecone(api_key=api_key)
        host = os.environ.get("PINECONE_INDEX_HOST")
        if host:
            _index_instance = _pc_instance.Index(host=host)
        else:
            index_name = os.environ["PINECONE_INDEX_NAME"]
            _index_instance = _pc_instance.Index(index_name)
    return _index_instance


def ensure_index_exists() -> None:
    """Create the index if missing. Call at deploy time, not per request.

    Dimension 768 = Gemini gemini-embedding-001 (truncated) output size.
    """
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["PINECONE_INDEX_NAME"]
    existing = [getattr(i, "name", None) or i.get("name") for i in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"[pinecone_client] Created index '{index_name}'")
    else:
        print(f"[pinecone_client] Index '{index_name}' already exists")


def embed_document(text: str) -> list[float]:
    """Embed text for upsert (RETRIEVAL_DOCUMENT task type)."""
    return _embed(text, "RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> list[float]:
    """Embed text for query time (RETRIEVAL_QUERY task type)."""
    return _embed(text, "RETRIEVAL_QUERY")


def _embed(text: str, task_type: str) -> list[float]:
    client = _get_genai_client()
    config = genai_types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=EMBEDDING_DIMENSION,
    )
    result = client.models.embed_content(
        model=EMBEDDING_MODEL, contents=text, config=config
    )
    return _extract_embedding(result)


def _extract_embedding(result: Any) -> list[float]:
    """Pull the float vector out of the shapes embed_content can return."""
    embeddings = getattr(result, "embeddings", None)
    if embeddings:
        first = embeddings[0]
        values = getattr(first, "values", None)
        if values is not None:
            return list(values)
        if isinstance(first, dict) and "values" in first:
            return list(first["values"])
    embedding = getattr(result, "embedding", None)
    if embedding is not None:
        values = getattr(embedding, "values", None)
        if values is not None:
            return list(values)
    if isinstance(result, dict):
        if "embedding" in result:
            emb = result["embedding"]
            return list(emb["values"]) if isinstance(emb, dict) else list(emb)
        if "embeddings" in result and result["embeddings"]:
            emb = result["embeddings"][0]
            return list(emb["values"]) if isinstance(emb, dict) else list(emb)
    raise RuntimeError("Could not extract embedding vector from embed_content result")
