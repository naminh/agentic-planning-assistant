"""Local dense embeddings: a third, semantic ranking beside BM25 and the fuzzy index.

BM25 and the stem/trigram index match words; neither finds 'sustainable drains' for a
query about 'surface water attenuation'. A small local sentence-embedding model does,
with no API calls: it runs on CPU through `fastembed` (the optional `embeddings` extra)
and chunk vectors are cached on disk, so only the first run pays. If the extra or the
model is unavailable the index is simply absent and retrieval behaves as before.
"""

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from loguru import logger

from domain import PolicyChunk

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DEPTH = 50  # nearest chunks considered per query
BATCH = 64


class Embedder(Protocol):
    """Turns texts into L2-normalised row vectors."""

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class FastEmbedder:
    """`fastembed` ONNX model, loaded on first use."""

    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: Path | None = None):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(
            model, cache_dir=str(cache_dir) if cache_dir else None
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed `texts`; rows are unit length."""
        vectors = np.array(list(self._model.embed(list(texts), batch_size=BATCH)))
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1.0, norms)


class EmbeddingIndex:
    """Cosine nearest-neighbour search over chunk vectors."""

    def __init__(
        self,
        chunks: Sequence[PolicyChunk],
        embedder: Embedder,
        model_name: str,
        cache_dir: Path | None = None,
    ) -> None:
        self._embedder = embedder
        self._vectors = self._load_or_build(chunks, model_name, cache_dir)

    def _load_or_build(
        self, chunks: Sequence[PolicyChunk], model_name: str, cache_dir: Path | None
    ) -> np.ndarray:
        digest = hashlib.sha256(model_name.encode())
        for chunk in chunks:
            digest.update(chunk.text.encode())
        path = cache_dir / f"{digest.hexdigest()[:20]}.npy" if cache_dir else None
        if path is not None and path.exists():
            vectors: np.ndarray = np.load(path)
            return vectors
        logger.info(f"Embedding {len(chunks)} policy chunks with {model_name} (once)")
        vectors = self._embedder.embed([c.text for c in chunks])
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, vectors)
        return vectors

    def rank(self, query: str, allowed: Sequence[int]) -> list[int]:
        """Allowed chunk positions nearest to `query`, best first (at most EMBED_DEPTH)."""
        if not allowed or not len(self._vectors):
            return []
        q = self._embedder.embed([query])[0]
        idx = np.array(list(allowed))
        sims = self._vectors[idx] @ q
        order = np.argsort(-sims, kind="stable")[:EMBED_DEPTH]
        return [int(idx[i]) for i in order]


def load_embedding_index(
    chunks: Sequence[PolicyChunk], model_name: str, cache_root: Path
) -> EmbeddingIndex | None:
    """Build the index, or None (logged) if `fastembed` or the model is unavailable."""
    try:
        embedder = FastEmbedder(model_name, cache_root / "fastembed-models")
        return EmbeddingIndex(chunks, embedder, model_name, cache_root / "embeddings")
    except Exception as err:  # missing extra, no network for the first download, ...
        logger.warning(
            f"Local embeddings unavailable ({type(err).__name__}: {err}); "
            "retrieval continues with BM25 and the fuzzy index"
        )
        return None
