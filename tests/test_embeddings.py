"""Tests for embeddings module."""

import pytest

from backend.rag.embeddings import EmbeddingGenerator, VectorDatabase


@pytest.fixture
def embedding_generator():
    """Create embedding generator with local model."""
    return EmbeddingGenerator(provider="local", model="all-MiniLM-L6-v2")


@pytest.fixture
def vector_db(tmp_path):
    """Create temporary vector database."""
    return VectorDatabase(persist_dir=str(tmp_path / "test_chroma"))


def test_embedding_generation(embedding_generator):
    """Test embedding generation."""
    text = "Float 2902123: 2023-03-15, location (2.1N, 45.2E), max depth 2000m"
    embedding = embedding_generator.generate(text)

    assert embedding is not None
    assert len(embedding) == embedding_generator.dimension
    assert all(isinstance(x, float) for x in embedding)


def test_batch_embedding_generation(embedding_generator):
    """Test batch embedding generation."""
    texts = [
        "Profile 1: Temperature 28.3°C",
        "Profile 2: Salinity 34.1 psu",
        "Profile 3: Dissolved oxygen 220 µmol/kg",
    ]

    embeddings = embedding_generator.generate_batch(texts)

    assert len(embeddings) == len(texts)
    assert all(len(emb) == embedding_generator.dimension for emb in embeddings)


def test_vector_db_add_and_query(embedding_generator, vector_db):
    """Test adding and querying vector database."""
    texts = [
        "Temperature profile in Indian Ocean",
        "Salinity measurements near equator",
        "Oxygen levels in Arabian Sea",
    ]

    embeddings = embedding_generator.generate_batch(texts)
    ids = [f"test_{i}" for i in range(len(texts))]
    metadatas = [{"index": i} for i in range(len(texts))]

    vector_db.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    query_text = "temperature data in ocean"
    query_embedding = embedding_generator.generate(query_text)

    results = vector_db.query(query_embeddings=[query_embedding], n_results=2)

    assert results is not None
    assert len(results["ids"][0]) == 2
    assert "temperature" in results["documents"][0][0].lower()


def test_vector_db_count(vector_db, embedding_generator):
    """Test vector database count."""
    initial_count = vector_db.count()

    texts = ["Test 1", "Test 2"]
    embeddings = embedding_generator.generate_batch(texts)
    ids = ["test_1", "test_2"]

    vector_db.add(ids=ids, embeddings=embeddings, documents=texts)

    assert vector_db.count() == initial_count + 2


def test_vector_db_delete(vector_db, embedding_generator):
    """Test deleting from vector database."""
    texts = ["Test 1", "Test 2", "Test 3"]
    embeddings = embedding_generator.generate_batch(texts)
    ids = ["del_1", "del_2", "del_3"]

    vector_db.add(ids=ids, embeddings=embeddings, documents=texts)
    count_before = vector_db.count()

    vector_db.delete(ids=["del_1", "del_2"])

    assert vector_db.count() == count_before - 2
