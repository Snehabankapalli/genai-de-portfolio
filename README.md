# GenAI Data Engineering Portfolio

> Production-grade AI and LLM integrations for data engineering use cases — RAG pipelines, LLM-powered agents, and intelligent data tooling.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Claude API](https://img.shields.io/badge/Claude_API-6B48FF?style=flat)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white)
![Apache Kafka](https://img.shields.io/badge/Kafka-231F20?style=flat&logo=apachekafka&logoColor=white)
![Snowflake](https://img.shields.io/badge/Snowflake-29B5E8?style=flat&logo=snowflake&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-FF9900?style=flat&logo=amazonaws&logoColor=white)
[![CI](https://github.com/Snehabankapalli/genai-de-portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/Snehabankapalli/genai-de-portfolio/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Projects

### Project 1: Real-Time RAG Document Pipeline

Production-grade document ingestion pipeline for Retrieval-Augmented Generation using Kafka, vector databases, and LLMs.

- **Throughput:** 10,000 documents/minute via Kafka streaming
- **Search latency:** <100ms semantic vector search
- **Stack:** Kafka, OpenAI embeddings, ChromaDB, FastAPI, PySpark

[View Project](project-1-rag-pipeline/)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  GenAI Data Engineering Stack                           │
│                                                         │
│  Data Sources → Kafka → Embedding Service → Vector DB   │
│                              │                          │
│                         LLM (Claude/GPT-4)              │
│                              │                          │
│                         RAG API → Consumers             │
└─────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Streaming | Apache Kafka |
| Embeddings | OpenAI text-embedding-3-small |
| Vector DB | ChromaDB / Pinecone |
| LLM | Claude API (Anthropic), OpenAI GPT-4 |
| Processing | PySpark, AWS EMR Serverless |
| API | FastAPI |
| Warehouse | Snowflake |
| Monitoring | Prometheus, Grafana |

---

## Quick Start

```bash
git clone https://github.com/Snehabankapalli/genai-de-portfolio
cd genai-de-portfolio

# Project 1: RAG Pipeline
cd project-1-rag-pipeline
pip install -r requirements.txt
cp .env.example .env  # Add your API keys — never hardcode
docker-compose up -d
python src/producer.py sample-docs/
python src/api.py
```

---

## Contributing

See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for setup and workflow.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sneha2095/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/Snehabankapalli)
