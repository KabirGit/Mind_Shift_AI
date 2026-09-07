# Mind Shift AI — Detailed Project Summary

This document is the implementation-oriented reference for the repository. It
describes the active FastAPI/Next.js product, the retained Streamlit entry point, the
data flow, deterministic services, model boundary, route contracts, configuration,
demo generation, deployment, testing, and known limitations.

## 1. Product intent and current status

Mind Shift AI is a longitudinal journaling and decision-support system. Its intended
progression is:

```text
journal or conversation
  -> understand the current situation
  -> retrieve relevant personal history
  -> identify evidence-backed patterns
  -> understand a decision and its constraints
  -> compare realistic alternatives
  -> score trade-offs and reversibility
  -> give a recommendation only when evidence supports one
  -> produce concrete next actions
```

Normal journaling remains a reflection experience. Decision support is an additional
bounded route, not a replacement for the existing memory and analytics pipeline.
Safety rules always take precedence over model output. The system is not a medical,
psychological, or emergency authority.

The deployed product is split into:

- `backend.api.main:app`: FastAPI service and route contracts.
- `frontend/`: static-export Next.js application.
- `backend/demo_data/`: committed recruiter/demo snapshots.
- `app.py`: retained legacy Streamlit surface.

## 2. End-to-end request flows

### 2.1 Reflection route

For a normal journal entry, `RAGService.run_pipeline()` performs:

```text
request
  -> CrisisDetector
  -> EmotionDetector
  -> TextProcessor
  -> MemoryManager + JournalDB persistence
  -> hybrid Retriever
  -> memory replay
  -> deterministic analytics and UserProfile update
  -> IntelligencePacket via Orchestrator
  -> PromptBuilder.build()
  -> one typed reflection completion
  -> crisis prefix when applicable
  -> redacted RequestTrace and latency JSONL
```

The response contains emotion, stored entry metadata, retrieved memories, prompt,
response text, crisis metadata, an optional `IntelligencePacket`, route mode, and a
request trace ID.

### 2.2 Guidance route

Decision-shaped requests are identified conservatively by cues such as “should I,”
“help me choose,” “deciding between,” “whether to,” or “thinking of quitting.” Normal
journal entries do not trigger the decision parser.

The guidance route has exactly two logical model calls when successful:

1. `decision_parse-v2`: JSON-mode extraction into a bounded `DecisionState`.
2. `guidance-response-v2`: JSON-mode phrasing of the validated deterministic result.

Between the calls, `GuidanceAgent` may make at most three read-only tool calls:

```text
search_similar_memories
get_emotional_patterns
get_user_profile
get_recent_decision_context
```

The agent chooses the first two based on the decision and current emotion, then
optionally makes a third call when evidence strength is below the early-stop threshold.
It records tool observations, result counts, failures, termination reason, and remaining
budget. It cannot mutate journal storage.

`DecisionContextBuilder` adapts existing retriever, SQLite, pattern, profile, and goal
services into a small evidence packet. `PromptContextPacker`:

- allows at most three memories, three patterns, two goals, two prior decisions, and
  six conversation turns;
- allowlists profile fields;
- suppresses near duplicates;
- excludes the current message from retrieved evidence;
- applies a global estimated 1,500-token budget;
- reports included, truncated, deduplicated, and dropped context.

`DecisionScorer` uses these weights:

| Dimension | Weight |
| --- | ---: |
| Goal alignment | 30% |
| Historical evidence | 25% |
| Feasibility | 20% |
| Reversibility | 15% |
| Downside containment | 10% |

`ReversibilityChecker` identifies difficult-to-undo actions such as quitting,
resigning, ending a relationship, buying, selling, relocating, signing, or publishing.
When such an option coincides with a high-confidence negative emotion, it requires a
cooling-off period and cannot receive a strong recommendation.

Recommendations are intentionally conservative. The service declines to choose when
evidence strength is below `0.35`, the leading score is below `0.55`, the lead over the
runner-up is under `0.08`, or the leading action requires cooling off. The fallback
output explains uncertainty and supplies clarification, information-gathering, trial,
or review-point actions.

### 2.3 Safety routes

`CrisisDetector` runs before other reasoning. Crisis language produces a calm,
non-diagnostic resource message and does not allow the normal recommendation behavior.

`GuidanceRiskClassifier` blocks decision guidance for:

- medication, dosage, treatment, diagnosis, or related medical changes;
- immediate danger, abuse, stalking, or domestic violence;
- severe mental-health signals such as psychosis, hallucinations, mania, or prolonged
  sleep deprivation.

Protected-risk requests use zero guidance model calls and zero guidance tools. The
response directs the user to appropriate qualified, trusted, urgent, or emergency
support.

## 3. Backend component map

### API and orchestration

- `backend/api/main.py`: creates the FastAPI app, CORS middleware, dependency-injected
  shared `RAGService`, live routes, and demo router.
- `backend/api/rag_service.py`: wires storage, retrieval, emotion, NLP, analytics,
  guidance, safety, profile, graph, reports, evaluation, tracing, and LLM clients.
- `backend/api/schemas.py`: Pydantic request and response contracts.
- `backend/api/demo.py`: reads committed JSON snapshots and serves them under
  `/api/demo/*`.

### Configuration and operational logging

- `backend/config/settings.py`: frozen environment-backed settings object.
- `backend/config/logger.py`: application logging setup.
- `backend/config/debug.py`: local stage logging helpers.
- `backend/evaluation/tracing.py`: redacted request traces with model attempts,
  prompt versions, tools, counts, status, failures, and stage latency.
- `backend/evaluation/eval_engine.py`: retrieval precision proxy, emotion confidence,
  latency summary, and trace health.
- `backend/evaluation/observability.py`: recruiter-facing content-free capability
  proof snapshot.

### Input processing and memory

- `backend/safety/crisis_detector.py`: first-pass crisis regex gate.
- `backend/safety/guidance_risk.py`: protected high-risk guidance gate.
- `backend/emotion/detector.py`: lazy emotion model loading and ranked emotion output.
- `backend/nlp/text_processor.py`: spaCy entities, keywords, fixed topic buckets,
  habit buckets, and VADER sentiment.
- `backend/ingestion/loaders.py`: document loading helpers for supported local files.
- `backend/embedding/pipeline.py`: chunking and embedding abstraction.
- `backend/retrieval/vector_store.py`: persistent FAISS `IndexFlatL2`, metadata,
  normalized-text hash deduplication, and optional bounded LTM.
- `backend/retrieval/retriever.py`: semantic, emotion, and recency ranking.
- `backend/memory/manager.py`: bounded short-term memory plus long-term vector memory.
- `backend/memory/replay_engine.py`: similar older entry, what happened next, recovery
  hint, and replay confidence.
- `backend/storage/db.py`: SQLite schema, upsert, migrations, and record reads.
- `backend/storage/models.py`: structured journal record fields.

### Deterministic analytics

All analytics run locally and are designed to expose confidence and explanation
metadata rather than opaque claims.

- `pattern_engine.py`: recurring emotions, topics, people, triggers, sentiment, and
  first/second-half trend.
- `habit_engine.py`: sentiment when a habit is mentioned versus other days.
- `relationship_engine.py`: person-level sentiment, emotion, mention count, type,
  closeness, co-occurrence, and trend.
- `temporal_engine.py`: day/time rhythm patterns with minimum samples and deviation
  thresholds.
- `causal_engine.py`: bounded lift-style comparisons for habits and stressor topics.
- `alert_engine.py`: consecutive-stress, trigger-spike, positive-streak, and habit
  absence alerts.
- `prediction_engine.py`: NumPy sentiment forecast and rule-based burnout signal.
- `goal_engine.py`: goal-keyword detection and sentiment-slope progress proxy.
- `timeline_engine.py`: significance-ranked journal events.
- `growth_tracker.py`: period snapshots, deltas, and narrative.
- `insight_engine.py`: evidence-thresholded template insights.
- `reflection_engine.py`: at most two reflective questions, optionally personalized
  from memory replay.
- `presentation.py`: user-facing labels and summary formatting.
- `dashboard_story.py`: composes the dashboard headline and sections from backend data.

### Profile, graph, reports, and orchestration

- `backend/profile/profile_manager.py`: maintains a single SQLite user profile with
  entry count, baseline/current sentiment, dominant emotion, recovery speed, top
  triggers/habits/people, growth score, and communication style.
- `backend/orchestrator/packet.py`: Pydantic `IntelligencePacket` contract.
- `backend/orchestrator/orchestrator.py`: combines already-computed analytics without
  making an LLM call.
- `backend/graph/knowledge_graph.py`: builds an in-memory NetworkX graph connecting
  user, topics, people, habits, and co-occurrences.
- `backend/reports/report_generator.py`: creates the weekly PDF with summaries,
  triggers, habits, people, insights, predictions, and disclaimers.

### Guidance package

- `guidance/models.py`: bounded decision, evidence, option, score, action, agent, and
  response-draft models.
- `guidance/decision_parser.py`: conservative router, JSON extraction, Pydantic
  validation, explicit-option fallback, and goal/constraint extraction.
- `guidance/agent.py`: bounded read-only tool selector.
- `guidance/option_generator.py`: deduplicates stated and generated choices and adds
  safe information/trial alternatives.
- `guidance/decision_scorer.py`: weighted deterministic option scoring.
- `guidance/reversibility.py`: reversibility and cooling-off assessment.
- `guidance/action_planner.py`: concrete next-action generation.
- `guidance/service.py`: coordinates parser, option generation, scoring, checks, and
  fallback rendering.
- `context/decision_context.py`: adapts existing services into decision evidence.
- `context/prompt_context.py`: global context caps, allowlists, deduplication, and
  token-budget accounting.

## 4. LLM boundary

`backend/llm/models.py` defines `LLMRequest`, `LLMCallResult`, usage metadata, failure
categories, and the common invocation boundary.

`HuggingFaceInferenceClient` is the default client. It sends OpenAI-compatible
chat-completion requests to:

```text
https://router.huggingface.co/v1/chat/completions
```

The default requested model is:

```text
arsoban/ocd-therapist-27b-v0.3:featherless-ai
```

The provider suffix is applied by the client from `HF_INFERENCE_PROVIDER`. The
compatibility `MistralInferenceClient` remains available with `LLM_BACKEND=mistral`.
Both clients share bounded timeout, retry, response validation, token usage, and
failure categorization behavior.

The model may:

- extract a decision from a message into a validated schema;
- phrase a deterministic reflection or guidance result;
- provide natural-language validation and explanation.

The model may not be relied on for:

- crisis or protected-risk classification;
- memory selection guarantees;
- context limits;
- option scoring;
- reversibility or cooling-off rules;
- data persistence;
- evaluation or trace redaction.

## 5. Storage and retrieval details

SQLite stores structured records with content-hash IDs, timestamps, emotion confidence,
sentiment, entities, keywords, topics, habits, and relationship metadata. The profile
is stored in the same database. Schema changes use idempotent migrations.

FAISS stores embeddings and metadata for long-term retrieval. The default retriever
combines:

```text
combined = 0.60 * semantic
         + 0.25 * emotion compatibility
         + 0.15 * recency decay
```

The default recency half-life is 72 hours and the candidate pool is 20. Emotion
compatibility is valence-aware rather than requiring exact emotion labels. Guidance
retrieval can opt into diversity and exclusion of the current content hash without
changing the legacy retrieval path.

Short-term memory is a bounded deque for current-session continuity. Long-term memory
is persisted in FAISS. Identical normalized text is deduplicated in long-term memory;
the structured journal record remains the source for analytics.

## 6. Frontend

The frontend is Next.js 16, React 19, TypeScript, Tailwind, Recharts, D3 force, and
Lucide icons. `next.config.mjs` uses `output: "export"` for Cloudflare Pages.

Important files:

- `src/app/page.tsx`: redirects `/` to `/dashboard`.
- `src/app/chat/page.tsx`: demo transcript or live chat, emotion display, route labels,
  structured decision evidence, memory replay, prompt/context debugging, and crisis
  banner.
- `src/app/dashboard/page.tsx`: narrative dashboard, thresholds, charts, people graph,
  graph search, timeline, and PDF link.
- `src/app/observability/page.tsx`: system proof and redacted operational evidence.
- `src/components/AppShell.tsx`: navigation and demo/live switch.
- `src/components/DemoModeProvider.tsx`: browser-local mode persistence; server and
  first load default to demo.
- `src/components/RelationshipGraph.tsx`: people graph visualization.
- `src/lib/api.ts`: typed route clients and shared response models.
- `src/lib/presentation.ts` and `src/lib/format.ts`: presentation-only formatting.

The frontend never receives LLM secrets. It receives only the public backend origin
through `NEXT_PUBLIC_API_URL`.

## 7. API route contracts

Live routes:

```text
GET  /api/health
POST /api/chat
GET  /api/dashboard/summary?range=Last%2030%20days
GET  /api/dashboard/story?range=Last%2030%20days
GET  /api/dashboard/goals
GET  /api/dashboard/predictions
GET  /api/dashboard/timeline
GET  /api/dashboard/growth
GET  /api/graph/query?node=...
GET  /api/graph/people
GET  /api/report/weekly
GET  /api/diagnostics
GET  /api/observability
```

`POST /api/chat` accepts `text`, optional `chat_history`, `top_k` from 1 to 20, and
optional tags. Its response includes:

- `mode`: `reflection`, `guidance`, or `safety`;
- emotion and crisis payloads;
- response text;
- retrieved memories and optional memory replay;
- optional `decision_state` and structured `guidance`;
- optional packet and prompt;
- `trace_id`.

Demo routes under `/api/demo` provide read-only snapshot equivalents for dashboard,
graph, diagnostics, observability, journal evidence, and chat history.

## 8. Demo and data generation

`scripts/generate_demo_snapshot.py` creates the committed synthetic recruiter dataset.
It builds 30 detailed consecutive entries, runs the real local analytics, calls the
same endpoint composers through TestClient, and writes:

```text
backend/demo_data/chat_transcript.json
backend/demo_data/dashboard_story.json
backend/demo_data/dashboard_summary.json
backend/demo_data/diagnostics.json
backend/demo_data/goals.json
backend/demo_data/graph_people.json
backend/demo_data/graph_queries.json
backend/demo_data/growth.json
backend/demo_data/predictions.json
backend/demo_data/timeline.json
```

Live provider access is optional during generation. If no valid credential is
available, the snapshot records fallback transcript text and remains useful for
analytics and UI demonstration.

`scripts/ingest_demo_persona.py` accepts a JSON object with an `entries` array or a
plain array. It requires exactly 30 entries, runs them chronologically through
`RAGService.run_pipeline()`, backdates SQLite and FAISS metadata, exports endpoint
outputs, and publishes them as demo snapshots.

`scripts/seed_demo_data.py` is the smaller local-development seed path. It clears
generated DB/vector state, inserts synthetic records through the real service, and
prints basic diagnostics.

## 9. Configuration and deployment

`backend/config/settings.py` reads environment variables once per settings call. Key
defaults are:

```text
DATA_DIR=data
VECTOR_STORE_DIR=faiss_store
SQLITE_PATH=data/journal.db
LATENCY_LOG_PATH=data/latency_log.jsonl
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMOTION_MODEL=SamLowe/roberta-base-go_emotions
LLM_BACKEND=huggingface
HF_MODEL=arsoban/ocd-therapist-27b-v0.3
HF_INFERENCE_PROVIDER=featherless-ai
GUIDANCE_CONTEXT_TOKEN_BUDGET=1500
```

`render.yaml` changes the deployment defaults to:

```text
EMBEDDING_MODEL=hashing
EMOTION_MODEL=rule-based
DATA_DIR=/tmp/mind-shift-ai/data
VECTOR_STORE_DIR=/tmp/mind-shift-ai/faiss_store
SQLITE_PATH=/tmp/mind-shift-ai/data/journal.db
LATENCY_LOG_PATH=/tmp/mind-shift-ai/data/latency_log.jsonl
MISTRAL_MODEL=mistral-small
```

The deployment checklist is in `DEPLOYMENT.md`. Cloudflare Pages builds from
`frontend/` with `npm run pages:build` and publishes `out`. Render runs the Docker
image and exposes `/api/health`. CORS accepts configured origins and Pages preview
origins through `ALLOWED_ORIGIN_REGEX`.

## 10. Evaluation, observability, and privacy

The system writes operational traces as JSONL beside latency records. A trace can
include route, status, outcome, requested and actual models, prompt versions, logical
model calls, provider attempts, retries, token usage when returned, tools, tool
observations, memory counts, agent termination, context selection, stage timings, and
failure category.

Traces intentionally exclude:

- raw journal text;
- raw prompts;
- raw retrieved-memory text;
- raw model responses.

`EvalEngine` reports:

- retrieval precision@k using topic overlap as an explicit relevance proxy;
- mean/min/max emotion confidence and low-confidence ratio;
- average and p95 latency from JSONL;
- trace health metrics.

`backend/evaluation/guidance_eval.py` validates production outputs against deterministic
cases. It can assert route, required/forbidden tools and memories, tool limits,
structured guidance fields, model calls, provider attempts, retries, trace status,
termination reason, and safety precedence. It does not use an LLM judge.

## 11. Testing and CI

Python tests are under `tests/` and cover:

```text
analytics, API, context, emotion, evaluation, graph, guidance, integration,
LLM clients, memory, NLP, orchestrator, profile, reports, retrieval, safety, storage
```

The GitHub Actions workflow `.github/workflows/tests.yml` runs backend and frontend
jobs. Backend uses Python 3.11, installs the spaCy model, runs Ruff, MyPy, and Pytest.
Frontend uses the repository Node version, runs `npm ci`, ESLint, the static export
build, and `npm audit --audit-level=high`.

Provider calls are mocked in tests. Local tests therefore do not require an HF or
Mistral credential, although model-dependent local development may still need model
downloads.

## 12. Repository layout

```text
.
├── backend/
│   ├── analytics/       deterministic analytics and dashboard story composition
│   ├── api/             FastAPI app, schemas, demo router, RAGService
│   ├── config/          settings, logging, debug helpers
│   ├── context/         bounded decision context and prompt packing
│   ├── demo_data/       committed demo JSON snapshots
│   ├── embedding/       embedding abstraction
│   ├── emotion/         emotion detector
│   ├── evaluation/      metrics, tracing, observability, guidance evaluation
│   ├── graph/           NetworkX knowledge graph
│   ├── guidance/        parser, tools, scoring, reversibility, actions
│   ├── ingestion/       local document loaders
│   ├── llm/             typed model clients and prompt builder
│   ├── memory/          short/long-term memory and replay
│   ├── nlp/             local enrichment
│   ├── orchestrator/    IntelligencePacket assembly
│   ├── profile/         persisted user profile
│   ├── reports/         PDF report generation
│   ├── retrieval/       FAISS storage and ranking
│   ├── safety/          crisis and protected-risk gates
│   └── storage/         SQLite records and migrations
├── frontend/
│   └── src/             Next.js routes, components, API types, presentation helpers
├── scripts/             seed, snapshot generation, persona ingestion
├── tests/               unit, API, integration, and contract tests
├── docs/                deployment and manual QA/architecture notes
├── app.py               legacy Streamlit entry point
├── Dockerfile           Render-compatible FastAPI image
├── Makefile             development commands
├── render.yaml          Render service configuration
└── Intent.md            product intent and architectural constraints
```

## 13. Known limitations and intentional non-goals

- There is no authentication, account system, or multi-user data isolation.
- The current service dependency is a shared single-user `RAGService`.
- Render free-tier files are ephemeral; persistent deployment needs a disk or external
  storage design.
- The graph is in memory and is rebuilt from SQLite records.
- Statistical predictions, burnout signals, emotion labels, and correlations are not
  diagnoses or causal proof.
- Crisis and risk regexes are conservative gates, not complete safety detection.
- Uploaded text is inserted into the live composer and is not automatically indexed as
  a general document collection.
- Long-term vector memory deduplicates identical normalized text.
- Lightweight Render backends are intentionally less semantically rich than the full
  local transformer stack.
- The model provider remains an external runtime dependency for live natural-language
  responses; deterministic fallback text is available when it fails.
- `app.py`/`make run` is retained for compatibility, while the deployed product path is
  FastAPI plus static Next.js.

For deployment procedures, see `DEPLOYMENT.md`. For the hardening rationale behind
the model boundary, bounded tools, context policy, tracing, and evaluation, see
`docs/production-genai-architecture.md`.
