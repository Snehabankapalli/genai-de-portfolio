"""Tests for RecursiveTextChunker and CodeAwareChunker."""

import pytest
from src.chunker import RecursiveTextChunker, CodeAwareChunker, DocumentChunk


class TestRecursiveTextChunkerBasic:
    def test_empty_string_returns_empty_list(self, chunker):
        assert chunker.chunk_text("") == []

    def test_whitespace_only_returns_empty_list(self, chunker):
        assert chunker.chunk_text("   \n\n   ") == []

    def test_short_text_produces_one_chunk(self, chunker, short_text):
        chunks = chunker.chunk_text(short_text, source="test.pdf")
        assert len(chunks) == 1
        assert chunks[0].source == "test.pdf"

    def test_chunk_has_correct_source(self, chunker, short_text):
        chunks = chunker.chunk_text(short_text, source="my_doc.txt")
        for chunk in chunks:
            assert chunk.source == "my_doc.txt"

    def test_chunk_indices_are_sequential(self, chunker, long_text):
        chunks = chunker.chunk_text(long_text)
        indices = [c.index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_total_chunks_consistent(self, chunker, long_text):
        chunks = chunker.chunk_text(long_text)
        total = len(chunks)
        for chunk in chunks:
            assert chunk.total_chunks == total

    def test_no_empty_chunks(self, chunker, long_text):
        chunks = chunker.chunk_text(long_text)
        for chunk in chunks:
            assert chunk.content.strip() != ""

    def test_chunk_size_respected(self):
        chunker = RecursiveTextChunker(chunk_size=50, chunk_overlap=5)
        text = "word " * 200  # 1000 chars of "word " repeated
        chunks = chunker.chunk_text(text)
        for chunk in chunks:
            # Allow some tolerance for separator handling
            assert len(chunk.content) <= 55

    def test_returns_document_chunk_instances(self, chunker, short_text):
        chunks = chunker.chunk_text(short_text)
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)


class TestDocumentChunkMetadata:
    def test_metadata_initialized_if_none(self):
        chunk = DocumentChunk(
            content="hello",
            index=0,
            source="test",
            total_chunks=1,
            metadata=None,
        )
        assert chunk.metadata == {}

    def test_metadata_has_char_count(self, chunker, short_text):
        chunks = chunker.chunk_text(short_text)
        for chunk in chunks:
            assert "char_count" in chunk.metadata
            assert chunk.metadata["char_count"] > 0

    def test_metadata_has_word_count(self, chunker, short_text):
        chunks = chunker.chunk_text(short_text)
        for chunk in chunks:
            assert "word_count" in chunk.metadata
            assert chunk.metadata["word_count"] > 0

    def test_metadata_has_code_flag(self, chunker, short_text):
        chunks = chunker.chunk_text(short_text)
        for chunk in chunks:
            assert "has_code" in chunk.metadata

    def test_code_flag_true_for_backtick_content(self, chunker):
        text = "Here is code: `SELECT * FROM users`"
        chunks = chunker.chunk_text(text)
        assert any(c.metadata["has_code"] for c in chunks)

    def test_code_flag_false_for_plain_text(self, chunker, short_text):
        chunks = chunker.chunk_text(short_text)
        assert all(not c.metadata["has_code"] for c in chunks)


class TestCharacterSplit:
    def test_very_long_single_word_handled(self):
        chunker = RecursiveTextChunker(chunk_size=10, chunk_overlap=2)
        text = "a" * 100
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= 10


class TestCodeAwareChunker:
    def test_is_subclass_of_recursive_chunker(self):
        assert issubclass(CodeAwareChunker, RecursiveTextChunker)

    def test_chunking_without_code_blocks(self):
        chunker = CodeAwareChunker(chunk_size=200)
        text = "Plain text without code. " * 20
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.content.strip() != ""

    def test_empty_text_returns_empty(self):
        chunker = CodeAwareChunker()
        assert chunker.chunk_text("") == []


class TestRecursiveTextChunkerCustomSeparators:
    def test_custom_separators_used(self):
        chunker = RecursiveTextChunker(
            chunk_size=20,
            chunk_overlap=0,
            separators=[".", " "]
        )
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 1
