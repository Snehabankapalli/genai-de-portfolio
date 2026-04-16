# GenAI Data Engineering Portfolio — Architecture

Production-grade GenAI + data engineering projects.

## Project 1: Real-Time RAG Pipeline

### End-to-End Flow

```mermaid
graph LR
    A["📄 Document Source<br/>S3 / API / Files"]
    B["🚀 Kafka Producer<br/>batch ingest"]
    C["📬 Kafka Topic<br/>documents"]
    D["⚡ Consumer<br/>chunker + embedder"]
    E["🧠 Embedding API<br/>text-embedding-3"]
    F["🗄 Vector Store<br/>Pinecone / pgvector"]
    G["🔎 Query API<br/>FastAPI"]
    H["📊 Observability<br/>Prometheus + rolling KPIs"]

    A -->|events| B
    B -->|publish| C
    C -->|consume| D
    D -->|embed| E
    E -->|vectors| F
    G -->|top-k search| F
    D -.->|metrics| H
    G -.->|metrics| H

    style E fill:#f3e5f5
    style F fill:#e8f5e9
    style H fill:#fff3e0
```

### Chunking + Embedding Pipeline

```mermaid
graph TB
    subgraph "Chunker"
        C1["recursive split<br/>max 512 tokens<br/>50 token overlap"]
    end
    subgraph "Embedder"
        E1["batch size 32<br/>retry w/ backoff<br/>cost tracking"]
    end
    subgraph "Upsert"
        U1["Pinecone namespace<br/>metadata: source, ts, chunk_id"]
    end

    C1 -->|chunks| E1
    E1 -->|vectors + meta| U1
```

### Observability

```mermaid
graph LR
    M["observability.py<br/>PipelineMetrics"]
    K["Kafka consumer lag"]
    L["embedding latency p50/p99"]
    C["API cost $/hr"]
    E["error rate"]
    P["GET /metrics<br/>Prometheus format"]

    K --> M
    L --> M
    C --> M
    E --> M
    M --> P
```

## Request Flow (Query API)

```mermaid
sequenceDiagram
    participant U as User
    participant A as FastAPI
    participant E as Embedder
    participant V as Vector DB
    participant L as Claude / GPT
    participant O as Observability

    U->>A: POST /query {"q": "..."}
    A->>E: embed(query)
    E-->>A: vector
    A->>V: similarity_search(vector, k=5)
    V-->>A: top-5 chunks
    A->>L: synthesize(query, chunks)
    L-->>A: answer + citations
    A-->>U: {answer, sources}
    A->>O: record_query(latency, cost)
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| Ingestion | Apache Kafka, Python |
| Processing | PySpark (streaming), asyncio |
| Embedding | OpenAI text-embedding-3-large |
| LLM | Claude Sonnet 4.6, GPT-4 |
| Vector DB | Pinecone, pgvector |
| API | FastAPI, Pydantic |
| Observability | Prometheus, structlog |
| Infra | Docker, AWS (EKS / Lambda) |

## SLOs

| Metric | Target |
|--------|--------|
| Query latency p95 | <2s |
| Embedding cost / 1M tokens | <$0.13 |
| Kafka consumer lag | <100 messages |
| Error rate | <0.1% |
| Retrieval precision@5 | >85% |
