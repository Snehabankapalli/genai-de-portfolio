"""
Production observability for the RAG document pipeline.

Tracks embedding latency, throughput, ChromaDB query performance,
token costs, error rates, and queue depth. Exposes Prometheus metrics
and structured logs for Grafana dashboards and Slack alerting.

Usage:
    metrics = PipelineMetrics()
    metrics.record_embedding_batch(count=100, latency_ms=240, tokens=48000)
    metrics.record_query(latency_ms=85, chunks_retrieved=5)
    metrics.record_error("embedding_timeout")
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

import structlog

logger = structlog.get_logger()

# Cost model (update when OpenAI changes pricing)
OPENAI_EMBEDDING_COST_PER_1M_TOKENS = 0.02    # text-embedding-3-small
OPENAI_COMPLETION_COST_PER_1K_TOKENS = 0.005   # gpt-4o-mini input
ALERT_THRESHOLDS = {
    "embedding_latency_p99_ms": 2000,
    "query_latency_p99_ms":     500,
    "error_rate_pct":           5.0,
    "queue_depth_lag":          10_000,
    "hourly_cost_usd":          10.0,
}


@dataclass
class BatchMetrics:
    """Metrics for a single embedding batch."""
    timestamp: float
    document_count: int
    chunk_count: int
    token_count: int
    latency_ms: float
    cost_usd: float
    errors: int = 0


@dataclass
class QueryMetrics:
    """Metrics for a single RAG query."""
    timestamp: float
    latency_ms: float
    chunks_retrieved: int
    reranked: bool
    completion_tokens: int
    cost_usd: float


@dataclass
class PipelineHealthSnapshot:
    """Point-in-time health summary for dashboards and alerting."""
    throughput_docs_per_min: float
    embedding_latency_p50_ms: float
    embedding_latency_p99_ms: float
    query_latency_p50_ms: float
    query_latency_p99_ms: float
    error_rate_pct: float
    total_cost_usd: float
    hourly_cost_usd: float
    kafka_lag: int
    chroma_doc_count: int
    alerts: list[str] = field(default_factory=list)


class PipelineMetrics:
    """
    Thread-safe metrics collector for the RAG pipeline.

    Maintains rolling windows for latency percentiles and rate calculations.
    Call export_prometheus() to integrate with Prometheus scraping.
    """

    _ROLLING_WINDOW_SECONDS = 3600   # 1-hour rolling window

    def __init__(self):
        self._lock = Lock()
        self._embedding_batches: list[BatchMetrics] = []
        self._queries: list[QueryMetrics] = []
        self._error_counts: dict[str, int] = {}
        self._total_cost_usd: float = 0.0
        self._kafka_lag: int = 0
        self._chroma_doc_count: int = 0

    def record_embedding_batch(
        self,
        document_count: int,
        chunk_count: int,
        token_count: int,
        latency_ms: float,
        errors: int = 0,
    ) -> None:
        """Record metrics for one embedding batch."""
        cost = token_count / 1_000_000 * OPENAI_EMBEDDING_COST_PER_1M_TOKENS
        metric = BatchMetrics(
            timestamp=time.time(),
            document_count=document_count,
            chunk_count=chunk_count,
            token_count=token_count,
            latency_ms=latency_ms,
            cost_usd=cost,
            errors=errors,
        )

        with self._lock:
            self._embedding_batches.append(metric)
            self._total_cost_usd += cost
            self._prune_old_records()

        logger.info(
            "embedding_batch",
            docs=document_count,
            chunks=chunk_count,
            tokens=token_count,
            latency_ms=round(latency_ms, 1),
            cost_usd=round(cost, 6),
        )

    def record_query(
        self,
        latency_ms: float,
        chunks_retrieved: int,
        completion_tokens: int,
        reranked: bool = False,
    ) -> None:
        """Record metrics for one RAG query."""
        cost = completion_tokens / 1000 * OPENAI_COMPLETION_COST_PER_1K_TOKENS
        metric = QueryMetrics(
            timestamp=time.time(),
            latency_ms=latency_ms,
            chunks_retrieved=chunks_retrieved,
            reranked=reranked,
            completion_tokens=completion_tokens,
            cost_usd=cost,
        )

        with self._lock:
            self._queries.append(metric)
            self._total_cost_usd += cost
            self._prune_old_records()

        logger.info(
            "rag_query",
            latency_ms=round(latency_ms, 1),
            chunks=chunks_retrieved,
            tokens=completion_tokens,
        )

    def record_error(self, error_type: str) -> None:
        """Increment error counter for a given error type."""
        with self._lock:
            self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1

        logger.warning("pipeline_error", error_type=error_type)

    def update_kafka_lag(self, lag: int) -> None:
        """Update latest Kafka consumer lag reading."""
        with self._lock:
            self._kafka_lag = lag

        if lag > ALERT_THRESHOLDS["queue_depth_lag"]:
            logger.warning("kafka_lag_high", lag=lag, threshold=ALERT_THRESHOLDS["queue_depth_lag"])

    def update_chroma_count(self, count: int) -> None:
        """Update total document count in ChromaDB."""
        with self._lock:
            self._chroma_doc_count = count

    @contextmanager
    def time_embedding_batch(self, document_count: int, chunk_count: int, token_count: int):
        """Context manager: auto-record embedding batch latency."""
        start = time.monotonic()
        errors = 0
        try:
            yield
        except Exception:
            errors = 1
            self.record_error("embedding_error")
            raise
        finally:
            latency_ms = (time.monotonic() - start) * 1000
            self.record_embedding_batch(document_count, chunk_count, token_count, latency_ms, errors)

    @contextmanager
    def time_query(self, chunks_retrieved: int, completion_tokens: int):
        """Context manager: auto-record query latency."""
        start = time.monotonic()
        try:
            yield
        except Exception:
            self.record_error("query_error")
            raise
        finally:
            latency_ms = (time.monotonic() - start) * 1000
            self.record_query(latency_ms, chunks_retrieved, completion_tokens)

    def get_health(self) -> PipelineHealthSnapshot:
        """Compute health snapshot for the current rolling window."""
        with self._lock:
            self._prune_old_records()
            batches = list(self._embedding_batches)
            queries = list(self._queries)
            kafka_lag = self._kafka_lag
            chroma_count = self._chroma_doc_count
            total_errors = sum(self._error_counts.values())

        now = time.time()
        window_start = now - self._ROLLING_WINDOW_SECONDS

        # Throughput
        recent_batches = [b for b in batches if b.timestamp >= window_start]
        total_docs = sum(b.document_count for b in recent_batches)
        throughput = total_docs / (self._ROLLING_WINDOW_SECONDS / 60)

        # Embedding latencies
        embed_latencies = sorted(b.latency_ms for b in recent_batches)
        embed_p50 = self._percentile(embed_latencies, 50)
        embed_p99 = self._percentile(embed_latencies, 99)

        # Query latencies
        recent_queries = [q for q in queries if q.timestamp >= window_start]
        query_latencies = sorted(q.latency_ms for q in recent_queries)
        query_p50 = self._percentile(query_latencies, 50)
        query_p99 = self._percentile(query_latencies, 99)

        # Error rate
        total_ops = len(recent_batches) + len(recent_queries) + 1
        error_rate = total_errors / total_ops * 100

        # Cost
        hourly_cost = sum(b.cost_usd for b in recent_batches) + sum(q.cost_usd for q in recent_queries)

        # Alerts
        alerts = self._check_alerts(embed_p99, query_p99, error_rate, hourly_cost, kafka_lag)

        return PipelineHealthSnapshot(
            throughput_docs_per_min=round(throughput, 1),
            embedding_latency_p50_ms=round(embed_p50, 1),
            embedding_latency_p99_ms=round(embed_p99, 1),
            query_latency_p50_ms=round(query_p50, 1),
            query_latency_p99_ms=round(query_p99, 1),
            error_rate_pct=round(error_rate, 2),
            total_cost_usd=round(self._total_cost_usd, 4),
            hourly_cost_usd=round(hourly_cost, 4),
            kafka_lag=kafka_lag,
            chroma_doc_count=chroma_count,
            alerts=alerts,
        )

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format for scraping."""
        health = self.get_health()
        lines = [
            "# HELP rag_throughput_docs_per_min Documents processed per minute",
            "# TYPE rag_throughput_docs_per_min gauge",
            f"rag_throughput_docs_per_min {health.throughput_docs_per_min}",
            "",
            "# HELP rag_embedding_latency_ms Embedding batch latency",
            "# TYPE rag_embedding_latency_ms summary",
            f'rag_embedding_latency_ms{{quantile="0.5"}} {health.embedding_latency_p50_ms}',
            f'rag_embedding_latency_ms{{quantile="0.99"}} {health.embedding_latency_p99_ms}',
            "",
            "# HELP rag_query_latency_ms RAG query latency",
            "# TYPE rag_query_latency_ms summary",
            f'rag_query_latency_ms{{quantile="0.5"}} {health.query_latency_p50_ms}',
            f'rag_query_latency_ms{{quantile="0.99"}} {health.query_latency_p99_ms}',
            "",
            "# HELP rag_error_rate_pct Error rate percentage",
            "# TYPE rag_error_rate_pct gauge",
            f"rag_error_rate_pct {health.error_rate_pct}",
            "",
            "# HELP rag_hourly_cost_usd Hourly LLM cost in USD",
            "# TYPE rag_hourly_cost_usd gauge",
            f"rag_hourly_cost_usd {health.hourly_cost_usd}",
            "",
            "# HELP rag_kafka_consumer_lag Kafka consumer lag",
            "# TYPE rag_kafka_consumer_lag gauge",
            f"rag_kafka_consumer_lag {health.kafka_lag}",
            "",
            "# HELP rag_chroma_doc_count Total documents in ChromaDB",
            "# TYPE rag_chroma_doc_count gauge",
            f"rag_chroma_doc_count {health.chroma_doc_count}",
        ]
        return "\n".join(lines)

    def _percentile(self, sorted_values: list[float], p: int) -> float:
        """Compute p-th percentile from a sorted list."""
        if not sorted_values:
            return 0.0
        idx = int(len(sorted_values) * p / 100)
        return sorted_values[min(idx, len(sorted_values) - 1)]

    def _check_alerts(
        self,
        embed_p99: float,
        query_p99: float,
        error_rate: float,
        hourly_cost: float,
        kafka_lag: int,
    ) -> list[str]:
        """Return list of active alert strings."""
        alerts = []
        t = ALERT_THRESHOLDS

        if embed_p99 > t["embedding_latency_p99_ms"]:
            alerts.append(f"EMBEDDING_SLOW: p99={embed_p99:.0f}ms (threshold={t['embedding_latency_p99_ms']}ms)")
        if query_p99 > t["query_latency_p99_ms"]:
            alerts.append(f"QUERY_SLOW: p99={query_p99:.0f}ms (threshold={t['query_latency_p99_ms']}ms)")
        if error_rate > t["error_rate_pct"]:
            alerts.append(f"HIGH_ERROR_RATE: {error_rate:.1f}% (threshold={t['error_rate_pct']}%)")
        if hourly_cost > t["hourly_cost_usd"]:
            alerts.append(f"COST_SPIKE: ${hourly_cost:.2f}/hr (threshold=${t['hourly_cost_usd']}/hr)")
        if kafka_lag > t["queue_depth_lag"]:
            alerts.append(f"KAFKA_LAG: {kafka_lag:,} messages (threshold={t['queue_depth_lag']:,})")

        return alerts

    def _prune_old_records(self) -> None:
        """Remove records older than the rolling window. Must be called under lock."""
        cutoff = time.time() - self._ROLLING_WINDOW_SECONDS
        self._embedding_batches = [b for b in self._embedding_batches if b.timestamp >= cutoff]
        self._queries = [q for q in self._queries if q.timestamp >= cutoff]
