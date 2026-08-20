"""Ingest a provided 30-day demo persona through the real backend pipeline.

This script intentionally does not interpret persona_brief and does not hand-write
analytics outputs. It:

1. Runs each journal entry through RAGService.run_pipeline in chronological order.
2. Backdates the stored SQLite record and FAISS metadata to the entry's date.
3. Calls the actual FastAPI dashboard/graph/diagnostics endpoints via TestClient.
4. Writes raw endpoint JSON under debug/demo_persona_raw_output/.
5. Publishes the same raw endpoint JSON to backend/demo_data/ for demo-mode UI.

Usage:
    python scripts/ingest_demo_persona.py C:/Users/Kabir/Downloads/demo_persona_journals.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from backend.api.main import app, get_service  # noqa: E402
from backend.api.rag_service import RAGService  # noqa: E402
from backend.config.settings import get_settings  # noqa: E402
from backend.embedding.pipeline import EmbeddingPipeline  # noqa: E402
from backend.retrieval.vector_store import FaissVectorStore  # noqa: E402
from backend.storage.db import JournalDB  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEMO_PERSONA_DIR = ROOT / "data" / "demo_persona"
RAW_OUTPUT_DIR = ROOT / "debug" / "demo_persona_raw_output"
DEMO_DATA_DIR = ROOT / "backend" / "demo_data"
INGESTION_LOG_NAME = "ingestion_log.json"
LLM_FALLBACK_RESPONSE = "I'm here with you. Tell me more about what you're feeling."


def _load_dataset(path: Path) -> tuple[str, list[dict[str, Any]]]:
    payload = json.JSONDecoder(strict=False).decode(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return "", payload
    if not isinstance(payload, dict):
        raise ValueError("Dataset must be a JSON object or an array of entries.")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Dataset object must contain an 'entries' array.")
    return str(payload.get("persona_brief", "")), entries


def _entry_timestamp(entry: dict[str, Any]) -> str:
    date = str(entry.get("date", "")).strip()
    if not date:
        raise ValueError(f"Entry day {entry.get('day')} has no date.")
    return f"{date}T12:00:00Z"


def _entry_sort_key(entry: dict[str, Any]) -> tuple[int, str]:
    return (int(entry.get("day", 0)), str(entry.get("date", "")))


def _reset_generated_dirs() -> None:
    for path in (DEMO_PERSONA_DIR, RAW_OUTPUT_DIR):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def _build_service() -> RAGService:
    latency_log_path = DEMO_PERSONA_DIR / "latency_log.jsonl"
    os.environ["SQLITE_PATH"] = str(DEMO_PERSONA_DIR / "journal.db")
    os.environ["VECTOR_STORE_DIR"] = str(DEMO_PERSONA_DIR / "faiss_store")
    os.environ["LATENCY_LOG_PATH"] = str(latency_log_path)

    settings = get_settings()
    embedding_pipeline = EmbeddingPipeline(
        model_name=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    vector_store = FaissVectorStore(
        persist_dir=str(DEMO_PERSONA_DIR / "faiss_store"),
        embedding_pipeline=embedding_pipeline,
        ltm_max_entries=settings.ltm_max_entries,
    )
    journal_db = JournalDB(str(DEMO_PERSONA_DIR / "journal.db"))
    service = RAGService(vector_store=vector_store, journal_db=journal_db)
    # Settings are evaluated when the module is imported, so force diagnostics
    # isolation even if another import already read the default latency path.
    service._latency_log_path = str(latency_log_path)
    service.eval_engine.latency_log_path = str(latency_log_path)
    return service


def _backdate_last_entry(service: RAGService, *, text: str, timestamp: str) -> Any:
    rid = service.vector_store._hash_entry(text, None)
    record = None
    for candidate in service.journal_db.get_all():
        if candidate.id == rid:
            candidate.timestamp = timestamp
            service.journal_db.insert(candidate)
            record = candidate
            break
    if record is None:
        raise RuntimeError("Pipeline did not persist the expected JournalRecord.")

    matched_meta = False
    for meta in service.vector_store.metadata:
        if meta.get("entry_hash") == rid or meta.get("text", "").strip() == text.strip():
            meta["timestamp"] = timestamp
            matched_meta = True
            break
    if not matched_meta:
        raise RuntimeError("Pipeline did not persist the expected FAISS metadata.")

    service.vector_store.save()
    return record


def _record_log(
    *,
    entry: dict[str, Any],
    timestamp: str,
    output: dict[str, Any],
    record: Any,
) -> dict[str, Any]:
    emotion = output.get("emotion", {})
    crisis = output.get("crisis", {})
    return {
        "day": entry.get("day"),
        "date": entry.get("date"),
        "stored_timestamp": timestamp,
        "record_id": record.id,
        "emotion": emotion.get("emotion"),
        "emotion_confidence": emotion.get("confidence"),
        "all_emotions": emotion.get("all_emotions", []),
        "topics": record.topics,
        "habits": record.habits,
        "entities_people": record.entities_people,
        "entities_places": record.entities_places,
        "entities_orgs": record.entities_orgs,
        "keywords": record.keywords,
        "person_relationship_types": record.person_relationship_types,
        "sentiment_compound": record.sentiment_compound,
        "sentiment_valence": record.sentiment_valence,
        "crisis_flagged": bool(crisis.get("flagged")),
        "crisis_matched_terms": crisis.get("matched_terms", []),
        "stored_entry": output.get("stored_entry", {}),
        "response": output.get("response", ""),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _fetch_json(client: TestClient, path: str) -> Any:
    response = client.get(path)
    response.raise_for_status()
    return response.json()


def _graph_query_nodes(service: RAGService) -> list[str]:
    records = service.journal_db.get_all()
    people: dict[str, int] = {}
    topics: dict[str, int] = {}
    habits: dict[str, int] = {}
    for record in records:
        for person in record.entities_people:
            people[person] = people.get(person, 0) + 1
        for topic in record.topics:
            topics[topic] = topics.get(topic, 0) + 1
        for habit in record.habits:
            habits[habit] = habits.get(habit, 0) + 1

    nodes = ["User"]
    for bucket in (people, topics, habits):
        nodes.extend(
            key for key, _ in sorted(bucket.items(), key=lambda item: item[1], reverse=True)[:5]
        )
    deduped = []
    for node in nodes:
        if node and node.lower() not in {item.lower() for item in deduped}:
            deduped.append(node)
    return deduped


def _export_endpoint_outputs(service: RAGService) -> dict[str, Any]:
    app.dependency_overrides[get_service] = lambda: service
    client = TestClient(app)
    try:
        outputs = {
            "dashboard_summary.json": _fetch_json(
                client, "/api/dashboard/summary?range=Last%2030%20days"
            ),
            "dashboard_story.json": _fetch_json(
                client, "/api/dashboard/story?range=Last%2030%20days"
            ),
            "goals.json": _fetch_json(client, "/api/dashboard/goals"),
            "predictions.json": _fetch_json(client, "/api/dashboard/predictions"),
            "timeline.json": _fetch_json(client, "/api/dashboard/timeline"),
            "growth.json": _fetch_json(client, "/api/dashboard/growth"),
            "diagnostics.json": _fetch_json(client, "/api/diagnostics"),
            "graph_people.json": _fetch_json(client, "/api/graph/people"),
        }
        graph_queries = {}
        for node in _graph_query_nodes(service):
            graph_queries[node.lower()] = _fetch_json(
                client, f"/api/graph/query?node={quote(node)}"
            )
        outputs["graph_queries.json"] = graph_queries
        return outputs
    finally:
        app.dependency_overrides.clear()


def _chat_transcript(log_entries: list[dict[str, Any]]) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    for item in log_entries:
        messages.append({"role": "user", "content": item.get("stored_entry", {}).get("text", "")})
        messages.append(
            {
                "role": "assistant",
                "content": item["response"],
                "emotion": {
                    "emotion": item["emotion"],
                    "confidence": item["emotion_confidence"],
                    "all_emotions": item["all_emotions"],
                },
                "memory_replay": None,
                "crisis": {
                    "flagged": item["crisis_flagged"],
                    "matched_terms": item["crisis_matched_terms"],
                },
                "retrieved_memories": [],
                "prompt": None,
            }
        )
    return {
        "mode": "demo",
        "persona": "30-day demo persona journal dataset ingested through RAGService.run_pipeline.",
        "generated_with_live_llm": any(
            item["response"] and item["response"] != LLM_FALLBACK_RESPONSE
            for item in log_entries
        ),
        "messages": messages,
    }


def _publish_outputs(outputs: dict[str, Any]) -> None:
    DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, payload in outputs.items():
        _write_json(DEMO_DATA_DIR / filename, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        nargs="?",
        default=str(Path.home() / "Downloads" / "demo_persona_journals.json"),
        help="Path to demo_persona_journals.json.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    persona_brief, entries = _load_dataset(dataset_path)
    ordered_entries = sorted(entries, key=_entry_sort_key)
    if len(ordered_entries) != 30:
        raise ValueError(f"Expected 30 entries, found {len(ordered_entries)}.")

    _reset_generated_dirs()
    service = _build_service()
    log_entries: list[dict[str, Any]] = []

    for entry in ordered_entries:
        text = str(entry.get("text", "")).strip()
        if not text:
            raise ValueError(f"Entry day {entry.get('day')} has empty text.")
        timestamp = _entry_timestamp(entry)
        output = service.run_pipeline(text=text, chat_history=[], top_k=3)
        record = _backdate_last_entry(service, text=text, timestamp=timestamp)
        log_entries.append(
            _record_log(
                entry=entry,
                timestamp=timestamp,
                output=output,
                record=record,
            )
        )
        print(
            "Ingested day "
            f"{entry.get('day')} ({entry.get('date')}): "
            f"{log_entries[-1]['emotion']} "
            f"{log_entries[-1]['emotion_confidence']}"
        )

    outputs = _export_endpoint_outputs(service)
    outputs["chat_transcript.json"] = _chat_transcript(log_entries)
    manifest = {
        "source_file": str(dataset_path),
        "entry_count": len(log_entries),
        "persona_brief_present": bool(persona_brief),
        "storage": {
            "sqlite": str(DEMO_PERSONA_DIR / "journal.db"),
            "faiss": str(DEMO_PERSONA_DIR / "faiss_store"),
            "latency_log": str(DEMO_PERSONA_DIR / "latency_log.jsonl"),
        },
        "raw_output_dir": str(RAW_OUTPUT_DIR),
        "published_demo_data_dir": str(DEMO_DATA_DIR),
    }

    _write_json(RAW_OUTPUT_DIR / INGESTION_LOG_NAME, {"entries": log_entries})
    for filename, payload in outputs.items():
        _write_json(RAW_OUTPUT_DIR / filename, payload)
    _write_json(RAW_OUTPUT_DIR / "manifest.json", manifest)
    _publish_outputs(outputs)

    print(f"Ingested {len(log_entries)} entries into {DEMO_PERSONA_DIR}")
    print(f"Wrote raw outputs to {RAW_OUTPUT_DIR}")
    print(f"Published demo-mode JSON to {DEMO_DATA_DIR}")


if __name__ == "__main__":
    main()
