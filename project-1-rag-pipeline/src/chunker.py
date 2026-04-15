"""
Document chunking service for RAG pipeline.
Implements intelligent text splitting with semantic preservation.
"""

import re
from typing import List, Iterator
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()


@dataclass
class DocumentChunk:
    """Represents a chunk of text with metadata."""
    content: str
    index: int
    source: str
    total_chunks: int
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RecursiveTextChunker:
    """
    Recursively splits text into chunks while preserving semantic boundaries.

    Uses hierarchical separators:
    1. Paragraph breaks (\\n\\n)
    2. Sentence boundaries (.!?)
    3. Word boundaries (space)

    Includes configurable overlap for context preservation.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\\n\\n", "\\n", ". ", "! ", "? ", " "]

    def chunk_text(self, text: str, source: str = "unknown") -> List[DocumentChunk]:
        """
        Split text into overlapping chunks.

        Args:
            text: Raw text content
            source: Document identifier

        Returns:
            List of DocumentChunk objects
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking", source=source)
            return []

        chunks = []
        current_chunk = ""
        chunk_index = 0

        # Clean text
        text = self._clean_text(text)

        # Split by top-level separator first
        sections = self._split_by_separator(text, self.separators[0])

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # If section fits in chunk, add it
            if len(current_chunk) + len(section) + 1 <= self.chunk_size:
                current_chunk += (" " if current_chunk else "") + section
            else:
                # Store current chunk if exists
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk, chunk_index, source, len(chunks) + 1
                    ))
                    chunk_index += 1

                # Handle oversized sections by recursive splitting
                if len(section) > self.chunk_size:
                    sub_chunks = self._recursive_split(section, source, chunk_index)
                    chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = section

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk, chunk_index, source, len(chunks) + 1
            ))

        # Update total_chunks for all
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total

        logger.info(
            "Text chunked successfully",
            source=source,
            total_chunks=total,
            avg_chunk_size=sum(len(c.content) for c in chunks) // max(total, 1)
        )

        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\\s+', ' ', text)
        # Remove non-printable characters
        text = re.sub(r'[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f]', '', text)
        return text.strip()

    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """Split text by given separator."""
        if separator == " ":
            # Word splitting
            return text.split()
        return text.split(separator)

    def _recursive_split(
        self,
        text: str,
        source: str,
        start_index: int,
        separator_idx: int = 1
    ) -> List[DocumentChunk]:
        """Recursively split oversized text with next separator."""
        chunks = []

        if separator_idx >= len(self.separators):
            # Final fallback: character-level splitting
            return self._character_split(text, source, start_index)

        separator = self.separators[separator_idx]
        parts = self._split_by_separator(text, separator)

        current = ""
        idx = start_index

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if len(current) + len(part) + len(separator) <= self.chunk_size:
                current += (separator if current else "") + part
            else:
                if current:
                    chunks.append(self._create_chunk(current, idx, source, 0))
                    idx += 1

                # If part still too big, recurse deeper
                if len(part) > self.chunk_size:
                    deep_chunks = self._recursive_split(part, source, idx, separator_idx + 1)
                    chunks.extend(deep_chunks)
                    idx += len(deep_chunks)
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(self._create_chunk(current, idx, source, 0))

        return chunks

    def _character_split(self, text: str, source: str, start_index: int) -> List[DocumentChunk]:
        """Final fallback: split by characters with overlap."""
        chunks = []
        step = self.chunk_size - self.chunk_overlap

        for i in range(0, len(text), step):
            chunk_text = text[i:i + self.chunk_size]
            chunks.append(self._create_chunk(chunk_text, start_index + len(chunks), source, 0))

        return chunks

    def _create_chunk(
        self,
        content: str,
        index: int,
        source: str,
        total: int
    ) -> DocumentChunk:
        """Create a DocumentChunk with metadata."""
        return DocumentChunk(
            content=content.strip(),
            index=index,
            source=source,
            total_chunks=total,
            metadata={
                "char_count": len(content),
                "word_count": len(content.split()),
                "has_code": "`" in content or "```" in content,
            }
        )


class CodeAwareChunker(RecursiveTextChunker):
    """
    Chunker that preserves code blocks and structured content.
    Extends RecursiveTextChunker with code detection.
    """

    CODE_BLOCK_PATTERN = re.compile(r'```[\\s\\S]*?```')

    def chunk_text(self, text: str, source: str = "unknown") -> List[DocumentChunk]:
        """Chunk text while preserving code blocks intact."""
        # Extract code blocks first
        code_blocks = self.CODE_BLOCK_PATTERN.findall(text)
        text_without_code = self.CODE_BLOCK_PATTERN.sub('<<<CODE_BLOCK>>>', text)

        # Chunk the non-code text
        chunks = super().chunk_text(text_without_code, source)

        # Re-insert code blocks into appropriate chunks
        result = []
        code_idx = 0

        for chunk in chunks:
            content = chunk.content
            while '<<<CODE_BLOCK>>>' in content and code_idx < len(code_blocks):
                content = content.replace(
                    '<<<CODE_BLOCK>>>',
                    code_blocks[code_idx],
                    1
                )
                code_idx += 1

            chunk.content = content
            result.append(chunk)

        return result
