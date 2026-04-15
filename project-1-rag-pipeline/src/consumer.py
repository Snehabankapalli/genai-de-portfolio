"""
Kafka consumer that processes documents, generates embeddings,
and stores in ChromaDB.
"""

import json
import signal
from typing import Optional
from dataclasses import asdict

from kafka import KafkaConsumer, TopicPartition
import chromadb
from chromadb.config import Settings
import structlog

from chunker import RecursiveTextChunker
from embedder import OpenAIEmbedder

logger = structlog.get_logger()


class DocumentProcessor:
    """
    Consumer that processes documents from Kafka,
    chunks them, generates embeddings, and stores in ChromaDB.
    """

    def __init__(
        self,
        kafka_servers: str = "localhost:9092",
        topic: str = "documents.ingestion",
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        collection_name: str = "documents"
    ):
        self.kafka_servers = kafka_servers
        self.topic = topic
        self.running = False

        # Initialize chunker and embedder
        self.chunker = RecursiveTextChunker(chunk_size=512, chunk_overlap=50)
        self.embedder = OpenAIEmbedder(batch_size=100)

        # Initialize ChromaDB
        self.chroma_client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port
        )

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize Kafka consumer
        self.consumer = KafkaConsumer(
            bootstrap_servers=kafka_servers,
            group_id="document-processors",
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            auto_commit_interval_ms=5000
        )

        logger.info(
            "Document processor initialized",
            kafka=kafka_servers,
            topic=topic,
            chroma=f"{chroma_host}:{chroma_port}"
        )

    def process_message(self, message) -> bool:
        """Process a single Kafka message."""
        try:
            document = message.value
            source = document.get("filename", "unknown")
            content = document.get("content", "")

            logger.info("Processing document", source=source, size=len(content))

            # Step 1: Chunk the document
            chunks = self.chunker.chunk_text(content, source=source)
            if not chunks:
                logger.warning("No chunks generated", source=source)
                return False

            logger.info("Document chunked", source=source, chunks=len(chunks))

            # Step 2: Generate embeddings
            embeddings = self.embedder.embed_chunks(chunks, source=source)

            logger.info("Embeddings generated", source=source, count=len(embeddings))

            # Step 3: Store in ChromaDB
            self._store_in_chromadb(embeddings, source)

            logger.info("Document processed successfully", source=source)
            return True

        except Exception as e:
            logger.error("Failed to process message", error=str(e))
            return False

    def _store_in_chromadb(self, embeddings: list, source: str):
        """Store embeddings in ChromaDB."""
        ids = [f"{source}_{e.chunk_index}" for e in embeddings]
        texts = [e.text for e in embeddings]
        vectors = [e.embedding for e in embeddings]
        metadatas = [{
            "source": source,
            "chunk_index": e.chunk_index,
            "model": e.model,
            "token_count": e.token_count
        } for e in embeddings]

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(embeddings), batch_size):
            self.collection.upsert(
                ids=ids[i:i+batch_size],
                documents=texts[i:i+batch_size],
                embeddings=vectors[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )

        logger.info("Stored in ChromaDB", source=source, count=len(embeddings))

    def start(self):
        """Start consuming messages."""
        self.running = True
        self.consumer.subscribe([self.topic])

        logger.info("Starting document processor", topic=self.topic)

        try:
            while self.running:
                # Poll for messages
                messages = self.consumer.poll(timeout_ms=1000)

                for topic_partition, msgs in messages.items():
                    for message in msgs:
                        if not self.running:
                            break
                        self.process_message(message)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.stop()

    def stop(self):
        """Stop the consumer."""
        self.running = False
        self.consumer.close()
        logger.info("Document processor stopped")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Document processing consumer")
    parser.add_argument("--kafka", default="localhost:9092", help="Kafka servers")
    parser.add_argument("--topic", default="documents.ingestion", help="Kafka topic")
    parser.add_argument("--chroma-host", default="localhost", help="ChromaDB host")
    parser.add_argument("--chroma-port", type=int, default=8000, help="ChromaDB port")

    args = parser.parse_args()

    processor = DocumentProcessor(
        kafka_servers=args.kafka,
        topic=args.topic,
        chroma_host=args.chroma_host,
        chroma_port=args.chroma_port
    )

    processor.start()


if __name__ == "__main__":
    main()
