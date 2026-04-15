"""Shared fixtures for RAG pipeline tests."""

import pytest
from unittest.mock import MagicMock

from src.chunker import DocumentChunk, RecursiveTextChunker


@pytest.fixture
def chunker():
    return RecursiveTextChunker(chunk_size=100, chunk_overlap=10)


@pytest.fixture
def short_text():
    return "This is a short document about Snowflake data pipelines."


@pytest.fixture
def long_text():
    paragraphs = [
        "Apache Kafka is a distributed event streaming platform.",
        "Snowflake is a cloud data warehouse built for the cloud.",
        "PySpark enables large-scale data processing in Python.",
        "dbt transforms data in the warehouse using SQL.",
        "AWS Glue is a serverless ETL service from Amazon.",
    ]
    return "\n\n".join(paragraphs * 10)


@pytest.fixture
def sample_chunk():
    return DocumentChunk(
        content="Snowflake is a cloud data warehouse.",
        index=0,
        source="test_doc.pdf",
        total_chunks=3,
        metadata={"char_count": 35, "word_count": 7, "has_code": False},
    )
