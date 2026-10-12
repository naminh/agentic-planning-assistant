"""Embedding ranking with a toy embedder (no model download)."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from config import load_settings
from domain import PolicyChunk
from policy.embeddings import EmbeddingIndex, load_embedding_index
from policy.index import PolicyIndex

CONCEPTS = [("water", "drain", "attenuation", "suds"), ("parking", "cars"), ("tree",)]


class Toy:
    """One axis per concept; synonyms share an axis, so meaning matches without words."""

    calls = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        Toy.calls += 1
        out = np.zeros((len(texts), len(CONCEPTS)))
        for r, text in enumerate(texts):
            for c, words in enumerate(CONCEPTS):
                out[r, c] = sum(w in text.lower() for w in words)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)


TEXTS = [
    "Sustainable drains (SuDS) must serve each dwelling.",
    "Parking for cars shall be provided.",
    "Trees are to be kept.",
]


def chunks() -> list[PolicyChunk]:
    return [PolicyChunk(source="S", page=i + 1, text=t) for i, t in enumerate(TEXTS)]


def test_rank_finds_a_synonym_that_shares_no_word(tmp_path: Path) -> None:
    index = EmbeddingIndex(chunks(), Toy(), "toy", tmp_path)
    assert index.rank("water attenuation", [0, 1, 2])[0] == 0
    assert index.rank("water attenuation", [1, 2])[0] in {1, 2}  # respects `allowed`
    assert index.rank("anything", []) == []


def test_vectors_are_cached_on_disk(tmp_path: Path) -> None:
    Toy.calls = 0
    EmbeddingIndex(chunks(), Toy(), "toy", tmp_path)
    EmbeddingIndex(chunks(), Toy(), "toy", tmp_path)
    assert Toy.calls == 1  # second build read the cache


def test_hybrid_search_adds_embedding_extras_after_the_lexical_hits(
    tmp_path: Path,
) -> None:
    index = PolicyIndex(chunks())
    assert [c.page for c in index.hybrid_search(["water attenuation"], k=1)] == []
    index.embeddings = EmbeddingIndex(chunks(), Toy(), "toy", tmp_path)
    assert [c.page for c in index.hybrid_search(["water attenuation"], k=1)] == [1]


def test_missing_model_degrades_to_no_embeddings(tmp_path: Path, monkeypatch) -> None:
    def boom(*a, **k):
        raise ImportError("no fastembed")

    monkeypatch.setattr("policy.embeddings.FastEmbedder", boom)
    assert load_embedding_index(chunks(), "x", tmp_path) is None


def test_with_embeddings_respects_the_setting() -> None:
    s = load_settings(require_key=False)
    index = PolicyIndex(chunks())
    assert (
        index.with_embeddings(
            s.__class__(**{**s.__dict__, "embeddings": False})
        ).embeddings
        is None
    )
