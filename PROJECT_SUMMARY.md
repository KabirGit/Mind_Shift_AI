# AI Reflection Intelligence Platform — Project Summary

> Comprehensive, self-contained reference describing the architecture, components, data flow, configuration, and runtime behavior of the project as it currently stands (Phases 1–18 complete). Intended to be fed to another LLM as context for Q&A.

---

## 1. High-Level Overview

**Project name:** AI Reflection Intelligence Platform (folder: `RAG`)

**Purpose:** A local-first Streamlit journaling companion that turns entries into explainable, deterministic self-reflection intelligence. It detects emotion, enriches text locally, stores structured records, runs a large suite of deterministic analytics engines (patterns, triggers, habits, relationships, temporal/causal reasoning, predictions, goals, timeline, growth, insights), surfaces proactive alerts and memory replays, and uses a single hosted LLM call only to phrase the empathetic reply.

**Core architecture claim:** The backend does all the thinking with deterministic Python (stats, rules, counters). The LLM is called in **exactly one place** (`PromptBuilder` → `llm.generate`) as the communication layer — never for analysis, pattern detection, prediction, or insight generation.

**Per-message pipeline:**

```
crisis check (rules, first) → emotion detection (28-label GoEmotions)
  → text enrichment (spaCy NER + keywords + topics + habits + VADER)
  → memory store (STM deque + FAISS LTM) → SQLite structured record
  → hybrid retrieval → memory replay → deterministic engines → IntelligencePacket
  → prompt build (insights + reflection + profile snapshot + replay)
  → single LLM call (Mistral) → response (+ crisis resources if flagged)
  → latency logged
```

**Runtime model:** Single Streamlit process. Heavy components cached once per process via `st.cache_resource`; STM + chat history per-session in `st.session_state`. Two tabs: **Chat** and **Insights Dashboard**.

**Stack:** Python 3.11 · Streamlit · sentence-transformers `all-MiniLM-L6-v2` (embeddings, local) · FAISS `IndexFlatL2` (local) · HF transformers `SamLowe/roberta-base-go_emotions` (emotion, 28 labels, local) · spaCy `en_core_web_sm` (local) · vaderSentiment (local) · SQLite (stdlib) · numpy · networkx · fpdf2 · Mistral free tier (the single external call) · LangChain loaders/splitters.

**Status:** 99 tests passing; ruff + mypy clean; CI runs lint + typecheck + test; one-command demo via `make seed && make run`.

---

## 2. Repository Layout

```
RAG/
├── app.py                       # Streamlit UI (Chat + Insights Dashboard tabs)
├── Makefile                     # install / install-dev / run / seed / test / lint / typecheck / clean
├── requirements.txt             # runtime deps only
├── requirements-dev.txt         # -r requirements.txt + pytest, ruff, mypy, types-requests
├── ruff.toml                    # lint config (E,F,I,UP,B,SIM; line-length 100)
├── mypy.ini                     # type-check config (+ narrow fpdf2 arg-type override)
├── pytest.ini
├── .env.example                 # copy to .env; only MISTRAL_API_KEY required
├── .gitignore                   # ignores .env, data/, faiss_store/, caches; keeps .env.example
├── .dockerignore / Dockerfile
├── .github/workflows/tests.yml  # CI: install-dev → spaCy model → ruff → mypy → pytest
├── docs/README.md               # screenshot placeholder instructions
├── scripts/seed_demo_data.py    # ~18 backdated synthetic entries through the real pipeline
├── data/text_files/             # demo journals (incl. demo_feature_check.txt)
├── faiss_store/                 # faiss.index + metadata.pkl (runtime, gitignored)
├── backend/
│   ├── api/rag_service.py       # RAGService — full pipeline orchestration (single LLM call site)
│   ├── config/{settings,logger,debug}.py
│   ├── emotion/detector.py      # GoEmotions 28-label detector w/ ranked spread
│   ├── ingestion/loaders.py     # pdf/txt/csv/docx/xlsx/json loaders
│   ├── embedding/pipeline.py    # sentence-transformers + chunking
│   ├── retrieval/{retriever,vector_store}.py
│   ├── memory/{manager,schema,replay_engine}.py
│   ├── storage/{db,models}.py   # JournalDB (SQLite + idempotent migrations) + JournalRecord
│   ├── nlp/text_processor.py    # spaCy + VADER + topic/habit buckets
│   ├── analytics/
│   │   ├── models.py            # TriggerStat, PatternSummary, compute_confidence
│   │   ├── _stats_utils.py      # window filter / trend / timestamp helpers
│   │   ├── pattern_engine.py    · habit_engine.py · relationship_engine.py
│   │   ├── temporal_engine.py   · causal_engine.py · alert_engine.py
│   │   ├── prediction_engine.py · goal_engine.py
│   │   ├── timeline_engine.py   · growth_tracker.py
│   │   ├── insight_engine.py    · reflection_engine.py
│   ├── profile/{models,profile_manager}.py   # UserProfile + ProfileManager
│   ├── orchestrator/{packet,orchestrator}.py # IntelligencePacket + Orchestrator
│   ├── graph/knowledge_graph.py # networkx knowledge graph
│   ├── safety/crisis_detector.py
│   ├── reports/report_generator.py           # weekly PDF (fpdf2)
│   └── evaluation/eval_engine.py             # precision@k, confidence stats, latency
└── tests/                       # 99 tests: emotion, retrieval, storage, nlp, analytics,
                                 #   profile, orchestrator, memory, graph, reports,
                                 #   evaluation, llm, safety, integration
```

---

## 3. Component Breakdown

### 3.1 `app.py` — Streamlit UI
- `_get_shared_components()` (`@st.cache_resource`): loads embedding pipeline, FAISS store, emotion detector, retriever, LLM client, prompt builder, JournalDB, TextProcessor once per process.
- Per-session `MemoryManager` (session-local STM deque) + `RAGService`.
- **Chat tab** (`_render_chat`): emotion + secondary-emotions display, response, 🔁 Memory Replay expander, show-retrieved / debug toggles, reset, chat input, file uploader; watch-level proactive-alert banner once per session.
- **Insights Dashboard** (`_render_dashboard`, no LLM): proactive alerts at top, time-range selector, emotion-over-time area chart, top topics bar, triggers/habits/people tables (with confidence), Goals progress bars, Stress Pattern (disclaimered), insights, Knowledge Graph search, "My Journey" timeline, "Growth Over Time" chart, weekly PDF download, ⚙ System Diagnostics expander.

### 3.2 `backend/config/settings.py` — `Settings`
Frozen dataclass from env. Notable: `sqlite_path` (`data/journal.db`), `latency_log_path` (`data/latency_log.jsonl`), `emotion_model` (`SamLowe/roberta-base-go_emotions`), retrieval weights (0.6/0.25/0.15, half-life 72, pool 20), `hf_api_token` (Mistral key). `.env.example` documents all keys; only `MISTRAL_API_KEY`/`HF_API_TOKEN` required.

### 3.3 `backend/emotion/detector.py` — `EmotionDetector`
GoEmotions (28 labels), lazy HF pipeline with `top_k=None`, no label collapsing. Output: `{emotion, confidence, all_emotions:[{emotion,score}...] (top-5)}`. Thread-safe, `truncation=True`, empty/error → neutral fallback. Env-swappable.

### 3.4 `backend/embedding/pipeline.py` — `EmbeddingPipeline`
sentence-transformers + LangChain recursive splitter; 384-dim float32.

### 3.5 `backend/retrieval/vector_store.py` — `FaissVectorStore`
Thread-safe `IndexFlatL2` + pickled metadata. Dedup by `sha256(normalized_text)`. Metadata now carries `topics` (Phase 18a) so retrieval evaluation is meaningful.

### 3.6 `backend/retrieval/retriever.py` — `Retriever`
Hybrid `0.6*semantic + 0.25*emotion + 0.15*recency`. Emotion is **valence-aware**: 1.0 exact, 0.5 either-neutral, 0.6 same valence family, else 0.0.

### 3.7 `backend/memory/` — `MemoryManager`, `MemoryEntry`, `ReplayEngine`
- STM bounded deque + FAISS LTM. `store_entry(..., topics=[])` stamps topics into `MemoryEntry.to_metadata()` (Phase 18a).
- **`ReplayEngine.find_replay`**: semantic query top-5, skips last 48h, picks best older entry, finds the entry 1–3 days after it as "what happened next", builds a recovery hint, confidence from FAISS score. Returns `None` if index < 5 entries.

### 3.8 `backend/storage/` — `JournalDB` + `JournalRecord`
SQLite `journal_records`, upsert by id (= FAISS entry_hash), list fields JSON-encoded, fail-safe try/except, idempotent `ALTER TABLE` migration (adds `habits`). `JournalRecord` includes entities/keywords/topics/habits/sentiment.

### 3.9 `backend/nlp/text_processor.py` — `TextProcessor`
Local spaCy NER + noun-chunk keywords + fixed topic buckets (career/health/relationship/money/education) + habit buckets (exercise/sleep/reading/meditation/social_media/coffee/cooking/coding) + VADER. Zero network.

### 3.10 `backend/analytics/` — deterministic engines (no LLM)
All output models carry `confidence = min(1, count/10)` + a one-sentence `explanation`.
- **`models.py`**: `TriggerStat`, `PatternSummary`, `compute_confidence`.
- **`_stats_utils.py`**: `parse_ts`, `sort_key`, `filter_window`, `half_split_trend`.
- **`pattern_engine.py`**: recurring emotions/topics/people + per-topic `TriggerStat` (freq ≥ 2, avg sentiment, mode emotion, first/second-half trend).
- **`habit_engine.py`**: `HabitCorrelation` — habit-mention vs not-mentioned sentiment delta → positive/negative/neutral.
- **`relationship_engine.py`**: `RelationshipProfile` — per-person (≥2 mentions) avg sentiment, mode emotion, last mention, improving/declining/stable trend.
- **`temporal_engine.py`**: `TemporalPattern` — per-topic (≥5 records) day-of-week peak deviation (>0.15).
- **`causal_engine.py`**: `CausalLink` — P(positive mood | habit same/next day) vs base rate; P(negative | stressor topic); lift, sample ≥ 3, |lift| > 0.1.
- **`alert_engine.py`**: `ProactiveAlert` — consecutive stress (watch), trigger spike (watch), positive streak (info), habit absence (info); severity-sorted.
- **`prediction_engine.py`**: `SentimentForecast` (numpy linear fit + R² confidence, improving/declining/stable) and `BurnoutRisk` (weighted rule score; **always** appends "This is a statistical pattern only, not a clinical assessment.").
- **`goal_engine.py`**: `GoalProgress` — goal-keyword matches (≥2), sentiment-slope progress proxy.
- **`timeline_engine.py`**: `TimelineEvent` — significance-ranked events; peaks always kept.
- **`growth_tracker.py`**: `GrowthSnapshot` per YYYY-MM + deltas + template `narrative()`.
- **`insight_engine.py`**: template insight strings from pattern + optional habit/relationship engines (freq/mention thresholds ≥ 3).
- **`reflection_engine.py`**: per-message rule-based reflective questions (max 2) + one personalized question derived from a `MemoryReplay` with a better past outcome.

### 3.11 `backend/profile/` — `UserProfile` + `ProfileManager`
Single-row `user_profiles` table in the same SQLite file. `update()` recomputes baseline vs current sentiment, dominant emotion, recovery speed, top triggers/habits/people, entry count, growth score, communication style (brief/reflective/detailed by avg length). Fail-safe.

### 3.12 `backend/orchestrator/` — `IntelligencePacket` + `Orchestrator`
`IntelligencePacket` (Pydantic) holds current emotion/sentiment, insights, reflection prompts, triggers, habits, relationships, user_profile, proactive_alerts, temporal_patterns, causal_links, predictions, goals, memory_replay. `Orchestrator.assemble()` collects already-computed outputs, degrades gracefully on failure.

### 3.13 `backend/graph/knowledge_graph.py` — `KnowledgeGraph`
In-memory networkx graph: User→topic/person edges (weight, sentiment) + topic↔person / topic↔habit co-occurrence. `query()` and `summarize_node()` for dashboard search.

### 3.14 `backend/safety/crisis_detector.py` — `CrisisDetector`
Rule-based regex crisis flag; runs first; on flag a calm non-diagnostic resource message is prepended (pipeline still runs).

### 3.15 `backend/reports/report_generator.py` — `ReportGenerator`
fpdf2 weekly PDF (uncompressed for greppability): "Your Story This Week" narrative → Emotional Summary → Triggers → Habits → People → Insights → Predictions → disclaimer. Zero-data safe.

### 3.16 `backend/evaluation/eval_engine.py` — `EvalEngine`
`retrieval_precision_at_k` (topic-overlap proxy), `emotion_confidence_stats`, `latency_summary` (avg/p95 from JSONL). Developer-facing (System Diagnostics).

### 3.17 `backend/llm/` — generation + prompt
- `HuggingFaceInferenceClient.generate()` calls **Mistral** (`mistral-small`) — the only network call; fixed empathetic fallback on failure.
- `PromptBuilder.build(...)`: role + safety + emotion + patterns + optional insights + Intelligence Context (profile snapshot) + Memory Replay + reflection questions + past entries + history + message. All optional sections omitted when empty (byte-identical for legacy callers).

### 3.18 `backend/api/rag_service.py` — `RAGService`
Wires every component; `run_pipeline()` times itself, runs crisis→emotion→enrich→store(+topics)→persist→retrieve→replay→reflection→assemble packet→prompt→LLM→(crisis prepend)→log latency. Returns `{emotion, stored_entry, retrieved_memories, prompt, response, crisis, packet}`. Helpers `_safe_insights`, `_safe_reflection`, `_assemble_packet`, `_persist_journal_record`, `_log_latency` all swallow exceptions.

---

## 4. Hybrid Retrieval Math
`s = 1-(d-d_min)/(d_max-d_min)`; emotion = 1.0/0.6/0.5/0.0 (exact/same-valence/either-neutral/else); recency = `0.5**(age_h/72)`; combined = `0.6*s + 0.25*emotion + 0.15*recency`.

---

## 5. Testing, CI, Tooling
- **99 tests** (`pytest -v` / `make test`); LLM always mocked → no API key needed in CI.
- **ruff** clean (`make lint`); **mypy** clean (`make typecheck`, one narrow fpdf2 stub override in `mypy.ini`).
- **CI** (`.github/workflows/tests.yml`): install-dev → spaCy model → ruff → mypy → pytest.
- **Makefile** targets: install, install-dev, run, seed, test, lint, typecheck, clean, help.
- **Seed** (`make seed`): ~18 backdated synthetic entries through the real pipeline (LLM stubbed locally, no key needed); backdates both SQLite and FAISS timestamps so replay/trends work; prints Precision@3 + growth narrative. Includes a crisis false-positive sanity assertion.

---

## 6. Known Quirks
- `HuggingFaceInferenceClient` actually calls Mistral (historical name).
- `HF_MODEL` read but unused by the active Mistral path.
- Uploaded files land in `data/text_files/` but aren't auto-indexed; delete `faiss_store/` and restart to rebuild.
- Dedup ignores timestamps; identical text updates STM but not LTM.
- GoEmotions model (~500MB) downloads once on first emotion detection, then cached.
- Dashboard tables use `use_container_width=True` (Streamlit deprecation warning; harmless until removed after 2026-12-31).

---

## 7. Q&A Hints for Downstream LLMs
- **Generation model?** `mistral-small` (Mistral API) — the only network call.
- **Embeddings?** `all-MiniLM-L6-v2` (384-dim, local).
- **Emotion?** `SamLowe/roberta-base-go_emotions` — 28 labels + ranked spread; env-swappable.
- **Stores?** FAISS `IndexFlatL2` (semantic) + SQLite `journal_records` (structured, migratable).
- **Deterministic vs LLM?** Crisis, emotion, enrichment, replay, and all ~12 analytics engines + profile + graph + reports are local/deterministic. LLM only phrases the reply — one call site.
- **Wow features?** Memory Replay, Proactive Alerts, Psychological Twin (UserProfile), Knowledge Graph search.
- **Explainability?** Every engine output carries confidence + one-sentence explanation.
- **Prediction/burnout?** numpy linear fit forecast; rule-based burnout score always disclaimered.
- **Retrieval scoring?** `0.6*semantic + 0.25*emotion + 0.15*recency`, valence-aware emotion.
- **Tests?** 99 passing; ruff + mypy clean; CI key-independent.
- **Precision@3?** ~57% on seed data (topic-overlap proxy) after the Phase 18a topics-sync fix.
- **Medical?** No — a reflection tool, not diagnosis; disclaimers throughout.
