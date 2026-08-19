# AI Reflection Intelligence Platform

A local-first journaling companion that turns your entries into explainable, deterministic self-reflection intelligence — with a single free-tier LLM call as the only external dependency.

![CI](https://img.shields.io/badge/tests-passing-brightgreen)

[Live App](https://mind-shift-ai.pages.dev) | [API Health](https://ai-reflection-intelligence-platform-eei6.onrender.com/api/health)

![Dashboard Screenshot](docs/screenshot_placeholder.png)

## What makes this different

- **Memory Replay:** surfaces similar past experiences and their outcomes from your own history.
- **Deterministic intelligence:** pattern/trigger/habit/insight logic is pure Python analytics — the LLM is only the final communication layer, not the reasoning engine.
- **Explainable outputs:** every engine output carries a confidence score and a one-sentence explanation of its evidence.
- **Proactive alerts:** the system surfaces meaningful changes without waiting to be asked.

## Architecture

```
journal entry
  → CrisisDetector (rules, first)          [local]
  → EmotionDetector (GoEmotions, 28 labels) [local, HF transformers]
  → TextProcessor (spaCy NER + keywords + topics + habits + VADER) [local]
  → MemoryManager (STM deque + FAISS LTM)  → JournalDB (SQLite record) [local]
  → Retriever (semantic + valence emotion + recency)
  → ReplayEngine (similar past entry + what happened next) [local]
  → engines feeding one IntelligencePacket (all local, deterministic):
        PatternEngine · HabitEngine · RelationshipEngine · TemporalEngine
        CausalEngine · AlertEngine · PredictionEngine · GoalEngine
        TimelineEngine · GrowthTracker · InsightEngine · ReflectionEngine
        ProfileManager (UserProfile) · KnowledgeGraph
  → Orchestrator.assemble() → IntelligencePacket
  → PromptBuilder.build(insights, reflection, profile snapshot, memory replay)
  → llm.generate() ◄── THE SINGLE LLM CALL (free-tier Mistral)
  → response (+ crisis resources prepended if flagged)
  → latency logged to data/latency_log.jsonl

Dashboard (no LLM): proactive alerts · emotion timeline · triggers/habits/people
  (with confidence) · goals · stress pattern · insights · knowledge-graph search
  · My Journey timeline · Growth Over Time · weekly PDF · System Diagnostics
```

## What's local vs. API

| Component | Where it runs |
|---|---|
| Emotion detection | Local (HuggingFace transformers) |
| Text/entity extraction | Local (spaCy) |
| Sentiment scoring | Local (VADER) |
| All analytics engines | Local (pure Python + numpy) |
| Knowledge graph | Local (networkx) |
| PDF export | Local (fpdf2) |
| Vector search | Local (FAISS) |
| Metadata storage | Local (SQLite) |
| LLM response generation | Free-tier Mistral API (1 call/entry) |

## Quick Start

```bash
git clone <repo> && cd <repo>
cp .env.example .env    # add your Mistral API key
make install-dev
make seed               # optional: populate 30 days of demo data
make run                # opens http://localhost:8501
```

Render uses lightweight deterministic embeddings/emotion detection so the app
fits small containers. For the full local transformer-backed experience, also run:

```bash
pip install -r requirements-ml.txt
```

## Deployment

- Frontend: [Cloudflare Pages](https://mind-shift-ai.pages.dev)
- Backend: [Render FastAPI service](https://ai-reflection-intelligence-platform-eei6.onrender.com/api/health)
- Frontend build root: `frontend`
- Cloudflare build command: `npm run pages:build`
- Cloudflare output directory: `out`
- Public frontend env var: `NEXT_PUBLIC_API_URL=https://ai-reflection-intelligence-platform-eei6.onrender.com`
- Backend CORS env var: `ALLOWED_ORIGIN=https://mind-shift-ai.pages.dev`

## Run Tests

```bash
make test
```

**99 tests passing** across emotion, retrieval, storage, NLP, analytics, profile, orchestrator, memory, graph, reports, evaluation, and integration. The LLM is mocked, so tests need no API key.

## Project Structure

```
RAG/
├── app.py                      # Streamlit UI (Chat + Insights Dashboard tabs)
├── Makefile                    # install / run / seed / test / lint / typecheck / clean
├── requirements.txt            # runtime deps
├── requirements-dev.txt        # + pytest, ruff, mypy
├── ruff.toml / mypy.ini        # lint + type-check config
├── .env.example                # copy to .env and add your key
├── .github/workflows/tests.yml # CI: lint + typecheck + test
├── scripts/
│   └── seed_demo_data.py       # populate demo data through the real pipeline
├── docs/                       # screenshots
└── backend/
    ├── api/rag_service.py      # pipeline orchestration (single LLM call site)
    ├── config/                 # settings, logging, debug
    ├── emotion/detector.py     # GoEmotions 28-label detector
    ├── ingestion/loaders.py    # multi-format document loaders
    ├── embedding/pipeline.py   # sentence-transformers + chunking
    ├── retrieval/              # retriever + FAISS vector store
    ├── memory/                 # manager, schema, replay_engine
    ├── storage/                # JournalDB (SQLite) + JournalRecord
    ├── nlp/text_processor.py   # spaCy + VADER + topic/habit buckets
    ├── analytics/              # pattern, habit, relationship, temporal, causal,
    │                           #   alert, prediction, goal, timeline, growth,
    │                           #   insight, reflection engines + models
    ├── profile/                # UserProfile + ProfileManager
    ├── orchestrator/           # IntelligencePacket + Orchestrator
    ├── graph/knowledge_graph.py# networkx knowledge graph
    ├── safety/crisis_detector.py
    ├── reports/report_generator.py  # weekly PDF (fpdf2)
    └── evaluation/eval_engine.py    # precision@k, confidence, latency
```

## Roadmap

Intentionally out of scope for v1:

- Multi-user authentication and per-user data isolation
- Knowledge graph persistence to disk
- Cognitive-distortion ML classifier (reflection is currently rule-based)
- Mobile app
- Real-time collaboration

## Architecture Decisions

- **Deterministic analytics, not LLM-as-reasoner:** patterns and predictions are computed in testable Python so results are reproducible, explainable, and free — the LLM only phrases the reply.
- **FAISS + SQLite dual storage:** FAISS gives fast semantic recall while SQLite holds the structured, queryable metadata that powers the analytics engines and dashboard.
- **Free-tier Mistral over a fully local model:** keeps setup lightweight and fast to run on any machine while still limiting the cloud dependency to a single call per entry.

> This project is a reflection tool, not a medical or psychological diagnosis.
