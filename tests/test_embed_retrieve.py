"""Tests for embed_retrieve.py — written BEFORE implementation (TDD Red phase)."""
import pytest
import numpy as np
from src.embed_retrieve import build_index, retrieve


SAMPLE_CHUNKS = [
    "Orbital mechanics is the study of spacecraft motion under gravitational forces.",
    "Chemical propulsion uses fuel combustion to generate thrust.",
    "Deep space missions travel beyond the Earth-Moon system.",
    "The Hohmann transfer orbit minimises fuel usage between planetary orbits.",
    "Radioisotope thermoelectric generators power long-duration space probes.",
]


@pytest.fixture(scope="module")
def built_index():
    model, index, embeddings = build_index(SAMPLE_CHUNKS)
    return model, index, embeddings, SAMPLE_CHUNKS


# --- build_index ---

def test_build_index_returns_three_objects(built_index):
    model, index, embeddings, _ = built_index
    assert model is not None
    assert index is not None
    assert embeddings is not None


def test_embeddings_shape_matches_chunks(built_index):
    _, _, embeddings, chunks = built_index
    assert embeddings.shape[0] == len(chunks)


def test_embeddings_are_float32(built_index):
    _, _, embeddings, _ = built_index
    assert embeddings.dtype == np.float32


def test_index_contains_correct_number_of_vectors(built_index):
    _, index, _, chunks = built_index
    assert index.ntotal == len(chunks)


def test_build_index_single_chunk():
    model, index, embeddings = build_index(["Single chunk text."])
    assert index.ntotal == 1
    assert embeddings.shape[0] == 1


def test_build_index_empty_raises():
    """Empty chunk list should raise a ValueError."""
    with pytest.raises((ValueError, Exception)):
        build_index([])


# --- retrieve ---

def test_retrieve_returns_list(built_index):
    model, index, _, chunks = built_index
    results = retrieve("propulsion", model, index, chunks)
    assert isinstance(results, list)


def test_retrieve_default_top_k(built_index):
    model, index, _, chunks = built_index
    results = retrieve("propulsion", model, index, chunks)
    assert len(results) == 3


def test_retrieve_custom_top_k(built_index):
    model, index, _, chunks = built_index
    for k in [1, 2, 4]:
        results = retrieve("space probe power", model, index, chunks, top_k=k)
        assert len(results) == k


def test_retrieve_results_are_strings(built_index):
    model, index, _, chunks = built_index
    results = retrieve("orbital mechanics", model, index, chunks)
    assert all(isinstance(r, str) for r in results)


def test_retrieve_results_are_subset_of_chunks(built_index):
    model, index, _, chunks = built_index
    results = retrieve("deep space", model, index, chunks)
    assert all(r in chunks for r in results)


def test_retrieve_semantic_relevance(built_index):
    """The most semantically relevant chunk should appear in results."""
    model, index, _, chunks = built_index
    results = retrieve("Hohmann transfer orbit fuel efficiency", model, index, chunks, top_k=2)
    assert any("Hohmann" in r for r in results)


def test_retrieve_top_k_capped_at_chunk_count(built_index):
    """top_k larger than available chunks should return all chunks without error."""
    model, index, _, chunks = built_index
    results = retrieve("space", model, index, chunks, top_k=100)
    assert len(results) == len(chunks)


def test_retrieve_no_duplicates(built_index):
    model, index, _, chunks = built_index
    results = retrieve("space travel propulsion", model, index, chunks, top_k=3)
    assert len(results) == len(set(results))
