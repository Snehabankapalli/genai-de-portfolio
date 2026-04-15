"""Tests for OpenAIEmbedder and EmbeddingCache — mocks OpenAI client."""

import pytest
from unittest.mock import MagicMock, patch

from src.embedder import OpenAIEmbedder, EmbeddingCache, EmbeddingResult
from src.chunker import DocumentChunk


def _make_chunk(content: str, index: int = 0) -> DocumentChunk:
    return DocumentChunk(
        content=content,
        index=index,
        source="test.pdf",
        total_chunks=1,
        metadata={"char_count": len(content), "word_count": len(content.split()), "has_code": False},
    )


def _mock_embedding_response(texts: list[str], dim: int = 1536):
    """Build a mock OpenAI embeddings response."""
    response = MagicMock()
    response.data = [
        MagicMock(embedding=[0.1] * dim) for _ in texts
    ]
    return response


class TestOpenAIEmbedderInit:
    def test_default_model(self):
        with patch("openai.OpenAI"):
            embedder = OpenAIEmbedder(api_key="test-key")
        assert embedder.model == "text-embedding-3-small"

    def test_custom_model(self):
        with patch("openai.OpenAI"):
            embedder = OpenAIEmbedder(model="text-embedding-3-large", api_key="test-key")
        assert embedder.model == "text-embedding-3-large"

    def test_default_batch_size(self):
        with patch("openai.OpenAI"):
            embedder = OpenAIEmbedder(api_key="test-key")
        assert embedder.batch_size == 100

    def test_known_model_dimension(self):
        with patch("openai.OpenAI"):
            embedder = OpenAIEmbedder(model="text-embedding-3-large", api_key="test-key")
        assert embedder._embedding_dim == 3072

    def test_unknown_model_defaults_dimension(self):
        with patch("openai.OpenAI"):
            embedder = OpenAIEmbedder(model="unknown-model", api_key="test-key")
        assert embedder._embedding_dim == 1536


class TestOpenAIEmbedderEmbedChunks:
    def setup_method(self):
        with patch("openai.OpenAI"):
            self.embedder = OpenAIEmbedder(api_key="test-key", batch_size=3)

    def test_returns_embedding_results(self):
        chunks = [_make_chunk(f"text {i}", i) for i in range(3)]
        mock_response = _mock_embedding_response([c.content for c in chunks])

        with patch.object(self.embedder, "_embed_batch", return_value=[[0.1] * 1536] * 3):
            results = self.embedder.embed_chunks(chunks, source="test.pdf")

        assert len(results) == 3
        for r in results:
            assert isinstance(r, EmbeddingResult)

    def test_result_has_correct_source(self):
        chunks = [_make_chunk("hello world")]
        with patch.object(self.embedder, "_embed_batch", return_value=[[0.1] * 1536]):
            results = self.embedder.embed_chunks(chunks, source="my_doc.pdf")
        assert results[0].source == "my_doc.pdf"

    def test_result_has_correct_model(self):
        chunks = [_make_chunk("hello world")]
        with patch.object(self.embedder, "_embed_batch", return_value=[[0.1] * 1536]):
            results = self.embedder.embed_chunks(chunks, source="doc.pdf")
        assert results[0].model == self.embedder.model

    def test_result_chunk_index_matches(self):
        chunks = [_make_chunk(f"text {i}", i) for i in range(2)]
        with patch.object(self.embedder, "_embed_batch", return_value=[[0.1] * 1536] * 2):
            results = self.embedder.embed_chunks(chunks)
        assert results[0].chunk_index == 0
        assert results[1].chunk_index == 1

    def test_batches_large_input(self):
        chunks = [_make_chunk(f"text {i}", i) for i in range(10)]
        call_count = []

        def fake_embed_batch(texts):
            call_count.append(len(texts))
            return [[0.1] * 1536] * len(texts)

        with patch.object(self.embedder, "_embed_batch", side_effect=fake_embed_batch):
            results = self.embedder.embed_chunks(chunks)

        assert len(results) == 10
        # With batch_size=3, should call _embed_batch 4 times (3+3+3+1)
        assert len(call_count) == 4

    def test_empty_chunks_returns_empty(self):
        results = self.embedder.embed_chunks([])
        assert results == []


class TestTokenEstimation:
    def test_estimate_tokens_roughly_char_over_four(self):
        with patch("openai.OpenAI"):
            embedder = OpenAIEmbedder(api_key="test-key")
        text = "a" * 400
        assert embedder._estimate_tokens(text) == 100

    def test_empty_string_returns_zero(self):
        with patch("openai.OpenAI"):
            embedder = OpenAIEmbedder(api_key="test-key")
        assert embedder._estimate_tokens("") == 0


class TestCalculateSimilarity:
    def setup_method(self):
        with patch("openai.OpenAI"):
            self.embedder = OpenAIEmbedder(api_key="test-key")

    def test_identical_embeddings_return_one(self):
        vec = [1.0, 0.0, 0.0]
        similarity = self.embedder.calculate_similarity(vec, vec)
        assert abs(similarity - 1.0) < 1e-6

    def test_orthogonal_embeddings_return_zero(self):
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = self.embedder.calculate_similarity(vec1, vec2)
        assert abs(similarity) < 1e-6

    def test_opposite_embeddings_return_negative_one(self):
        vec1 = [1.0, 0.0]
        vec2 = [-1.0, 0.0]
        similarity = self.embedder.calculate_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 1e-6

    def test_similarity_bounded(self):
        import random
        random.seed(42)
        vec1 = [random.random() for _ in range(128)]
        vec2 = [random.random() for _ in range(128)]
        similarity = self.embedder.calculate_similarity(vec1, vec2)
        assert -1.0 <= similarity <= 1.0


class TestEmbeddingCache:
    def setup_method(self):
        self.cache = EmbeddingCache()

    def test_miss_returns_none(self):
        result = self.cache.get("hello", "model-v1")
        assert result is None

    def test_set_then_get_returns_embedding(self):
        embedding = [0.1, 0.2, 0.3]
        self.cache.set("hello", "model-v1", embedding)
        result = self.cache.get("hello", "model-v1")
        assert result == embedding

    def test_different_model_is_cache_miss(self):
        self.cache.set("hello", "model-v1", [0.1, 0.2])
        result = self.cache.get("hello", "model-v2")
        assert result is None

    def test_different_text_is_cache_miss(self):
        self.cache.set("hello", "model-v1", [0.1, 0.2])
        result = self.cache.get("world", "model-v1")
        assert result is None

    def test_hit_rate_calculated_correctly(self):
        self.cache.set("a", "m", [1.0])
        self.cache.get("a", "m")   # hit
        self.cache.get("b", "m")   # miss

        stats = self.cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert abs(stats["hit_rate"] - 0.5) < 1e-6

    def test_size_increments_on_set(self):
        self.cache.set("a", "m", [1.0])
        self.cache.set("b", "m", [2.0])
        assert self.cache.get_stats()["size"] == 2

    def test_zero_requests_hit_rate_zero(self):
        stats = self.cache.get_stats()
        assert stats["hit_rate"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
