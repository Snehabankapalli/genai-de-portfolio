"""
Kafka producer for document ingestion.
Handles file parsing and message publishing.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from kafka import KafkaProducer
import structlog

logger = structlog.get_logger()


class DocumentProducer:
    """
    Producer that reads documents and publishes to Kafka.

    Supports: PDF, DOCX, TXT, MD
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "documents.ingestion"
    ):
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda v: v.encode('utf-8') if v else None,
            acks='all',
            retries=3,
            max_in_flight_requests=5
        )
        logger.info(
            "Document producer initialized",
            bootstrap=bootstrap_servers,
            topic=topic
        )

    def parse_document(self, file_path: Path) -> Optional[dict]:
        """Parse a document file and extract content."""
        try:
            suffix = file_path.suffix.lower()

            if suffix == '.pdf':
                content = self._parse_pdf(file_path)
            elif suffix == '.docx':
                content = self._parse_docx(file_path)
            elif suffix in ['.txt', '.md']:
                content = file_path.read_text(encoding='utf-8')
            else:
                logger.warning("Unsupported file type", file=str(file_path))
                return None

            return {
                "content": content,
                "filename": file_path.name,
                "source": str(file_path),
                "file_type": suffix.lstrip('.'),
                "parsed_at": datetime.utcnow().isoformat(),
                "size_bytes": file_path.stat().st_size
            }
        except Exception as e:
            logger.error("Failed to parse document", file=str(file_path), error=str(e))
            return None

    def _parse_pdf(self, file_path: Path) -> str:
        """Parse PDF content."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except ImportError:
            logger.error("pypdf not installed")
            return ""

    def _parse_docx(self, file_path: Path) -> str:
        """Parse DOCX content."""
        try:
            from docx import Document
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except ImportError:
            logger.error("python-docx not installed")
            return ""

    def publish_document(self, document: dict) -> bool:
        """Publish a document to Kafka."""
        try:
            key = document.get("filename", "unknown")
            future = self.producer.send(
                self.topic,
                key=key,
                value=document
            )
            future.get(timeout=10)  # Wait for confirmation
            logger.info(
                "Document published",
                filename=document["filename"],
                size=document["size_bytes"]
            )
            return True
        except Exception as e:
            logger.error("Failed to publish", error=str(e))
            return False

    def process_directory(self, dir_path: str, recursive: bool = True):
        """Process all documents in a directory."""
        path = Path(dir_path)

        if not path.exists():
            logger.error("Directory not found", path=dir_path)
            return

        pattern = "**/*" if recursive else "*"
        supported = ['.pdf', '.docx', '.txt', '.md']

        files = [f for f in path.glob(pattern) if f.suffix.lower() in supported]

        logger.info("Processing directory", path=dir_path, files_found=len(files))

        success_count = 0
        for file_path in files:
            doc = self.parse_document(file_path)
            if doc and self.publish_document(doc):
                success_count += 1

        logger.info(
            "Directory processing complete",
            total=len(files),
            successful=success_count
        )

        # Flush and close
        self.producer.flush()

    def close(self):
        """Close the producer."""
        self.producer.close()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Document ingestion producer")
    parser.add_argument("path", help="File or directory to process")
    parser.add_argument("--kafka", default="localhost:9092", help="Kafka bootstrap servers")
    parser.add_argument("--topic", default="documents.ingestion", help="Kafka topic")
    parser.add_argument("--recursive", action="store_true", help="Process subdirectories")

    args = parser.parse_args()

    producer = DocumentProducer(
        bootstrap_servers=args.kafka,
        topic=args.topic
    )

    try:
        path = Path(args.path)
        if path.is_file():
            doc = producer.parse_document(path)
            if doc:
                producer.publish_document(doc)
        else:
            producer.process_directory(args.path, args.recursive)
    finally:
        producer.close()


if __name__ == "__main__":
    main()
