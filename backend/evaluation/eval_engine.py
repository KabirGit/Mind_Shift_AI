from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)


class EvalEngine:
    """System-quality metrics. Deterministic, local, developer-facing."""

    def __init__(self, vector_store, db, latency_log_path: str = "data/latency_log.jsonl") -> None:
        self.vector_store = vector_store
        self.db = db
        self.latency_log_path = latency_log_path

    def retrieval_precision_at_k(self, k: int = 3, n_samples: int = 10) -> dict:
        note = "Approximation using topic overlap as relevance proxy."
        try:
            records = self.db.get_all()
            if not records:
                return {"precision_at_k": 0.0, "k": k, "n_samples_used": 0, "note": note}

            sample = records[:n_samples]
            relevant = 0
            for rec in sample:
                query_topics = set(rec.topics or [])
                results = self.vector_store.query(rec.text, top_k=k)
                for res in results:
                    meta = res.get("metadata", {})
                    if query_topics & set(meta.get("topics", []) or []):
                        relevant += 1
            denom = k * len(sample)
            precision = relevant / denom if denom else 0.0
            return {
                "precision_at_k": round(precision, 4),
                "k": k,
                "n_samples_used": len(sample),
                "note": note,
            }
        except Exception as exc:
            logger.exception("retrieval_precision_at_k failed: %s", exc)
            return {"precision_at_k": 0.0, "k": k, "n_samples_used": 0, "note": note}

    def emotion_confidence_stats(self) -> dict:
        default = {"mean_confidence": 0.0, "min": 0.0, "max": 0.0, "low_confidence_ratio": 0.0}
        try:
            records = self.db.get_all()
            if not records:
                return default
            confs = [r.emotion_confidence for r in records]
            low = sum(1 for c in confs if c < 0.5)
            return {
                "mean_confidence": round(sum(confs) / len(confs), 4),
                "min": round(min(confs), 4),
                "max": round(max(confs), 4),
                "low_confidence_ratio": round(low / len(confs), 4),
            }
        except Exception as exc:
            logger.exception("emotion_confidence_stats failed: %s", exc)
            return default

    def latency_summary(self) -> dict:
        default = {"avg_ms": 0.0, "p95_ms": 0.0, "sample_count": 0}
        try:
            if not os.path.exists(self.latency_log_path):
                return default
            values: list[float] = []
            with open(self.latency_log_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        values.append(float(json.loads(line)["elapsed_ms"]))
                    except Exception:
                        continue
            if not values:
                return default
            values.sort()
            idx = max(0, int(round(0.95 * (len(values) - 1))))
            return {
                "avg_ms": round(sum(values) / len(values), 2),
                "p95_ms": round(values[idx], 2),
                "sample_count": len(values),
            }
        except Exception as exc:
            logger.exception("latency_summary failed: %s", exc)
            return default
