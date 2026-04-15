"""
Embedding service using OpenAI's embedding models.
Handles batching, rate limiting, and error retries.
"""

import os
from typing import List, Optional
from dataclasses import dataclass
import time
import asyncio
from functools import wraps

import openai
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger()


@dataclass
class EmbeddingResult:
    """Represents an embedding result for a text chunk."""
    text: str
    embedding: List[float]
    model: str
    token_count: int
    source: str
    chunk_index: int


class OpenAIEmbedder:
    """
    OpenAI embedding client with batching and retry logic.

    Features:
    - Batch processing for cost efficiency
    - Automatic rate limit handling
    - Token count tracking
    - Async support for high throughput
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        batch_size: int = 100,
        max_retries: int = 3
    ):
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries

        # Model-specific dimensions
        self.dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }

        self._embedding_dim = self.dimensions.get(model, 1536)
        logger.info(
            "OpenAI embedder initialized",
            model=model,
            batch_size=batch_size
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts with retry logic."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except openai.RateLimitError as e:
            logger.warning("Rate limit hit, retrying", error=str(e))
            raise
        except Exception as e:
            logger.error("Embedding failed", error=str(e))
            raise

    def embed_chunks(
        self,
        chunks: List,
        source: str = "unknown"
    ) -> List[EmbeddingResult]:
        """
        Embed document chunks in batches.

        Args:
            chunks: List of DocumentChunk objects
            source: Document source identifier

        Returns:
            List of EmbeddingResult objects
        """
        results = []
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(0, len(chunks), self.batch_size):
            batch = chunks[batch_idx:batch_idx + self.batch_size]
            texts = [chunk.content for chunk in batch]

            logger.info(
                "Processing batch",
                batch=batch_idx // self.batch_size + 1,
                total=total_batches,
                size=len(batch)
            )

            # Get embeddings
            embeddings = self._embed_batch(texts)

            # Create results
            for chunk, embedding in zip(batch, embeddings):
                results.append(EmbeddingResult(
                    text=chunk.content,
                    embedding=embedding,
                    model=self.model,
                    token_count=self._estimate_tokens(chunk.content),
                    source=source,
                    chunk_index=chunk.index
                ))

            # Rate limit buffer
            if batch_idx + self.batch_size < len(chunks):
                time.sleep(0.1)

        logger.info(
            "Embedding complete",
            total_chunks=len(results),
            model=self.model
        )

        return results

    async def embed_chunks_async(
        self,
        chunks: List,
        source: str = "unknown"
    ) -> List[EmbeddingResult]:
        """Async version for high-throughput scenarios."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.embed_chunks, chunks, source
        )

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 chars for English)."""
        return len(text) // 4

    def calculate_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings."""
        import numpy as np

        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


class EmbeddingCache:
    """
    Simple in-memory cache for embeddings to avoid redundant API calls.
    Uses content hash as key.
    """

    def __init__(self):
        self._cache = {}
        self._hits = 0
        self._misses = 0

    def _get_key(self, text: str, model: str) -> str:
        """Generate cache key from text and model."""
        import hashlib
        content = f"{text}:{model}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Get cached embedding if exists."""
        key = self._get_key(text, model)
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def set(self, text: str, model: str, embedding: List[float]):
        """Cache an embedding."""
        key = self._get_key(text, model)
        self._cache[key] = embedding

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "size": len(self._cache)
        }
