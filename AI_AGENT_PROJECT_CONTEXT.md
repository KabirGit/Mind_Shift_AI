# Mind Shift AI - Complete Project Context for AI Agents

Last updated: 2026-08-20

This file is a high-detail, self-contained project briefing intended for another AI agent, coding assistant, or LLM. Read this before making changes. It captures the current architecture, runtime flow, deployment setup, key files, data stores, environment variables, known quirks, and debugging context.

## 1. Project Identity

Project name: Mind Shift AI / AI Reflection Intelligence Platform

Repository: https://github.com/KabirGit/Mind_Shift_AI

Local workspace path:

```text
C:\Users\Kabir\Desktop\CAMPUSX\RAG
```

Current Git branch used for deployment:

```text
main
```

Live frontend:

```text
https://mind-shift-ai.pages.dev
```

Live backend health endpoint:

```text
https://ai-reflection-intelligence-platform-eei6.onrender.com/api/health
```

Expected backend health response:

```json
{"status":"ok"}
```

## 2. One-Sentence Summary

Mind Shift AI is a local-first AI journaling and reflection platform that analyzes journal entries with local or deterministic components, stores semantic and structured memory, runs explainable analytics, and uses a single Mistral LLM call only to phrase the final empathetic response.

## 3. Core Architectural Principle

The LLM is not the reasoning engine.

The project intentionally separates deterministic reasoning from language generation:

```text
Python analytics, retrieval, memory, safety, profile, reports = reasoning
Mistral LLM call = final communication layer
```

This keeps the system:

- More explainable
- Easier to test
- Cheaper to run
- More deterministic
- Easier to debug
- Safer than a plain chatbot that delegates all reasoning to the model

## 4. Current Product Surfaces

There are two user-facing surfaces in the repo.

### 4.1 Deployed Frontend

Location:

```text
frontend/
```

Framework:

```text
Next.js static export
React
TypeScript
Tailwind CSS
Lucide icons
Recharts
```

Hosted on:

```text
Cloudflare Pages
```

Build settings:

```text
Root directory: frontend
Build command: npm run pages:build
Build output directory: out
Framework preset: Next.js (Static HTML Export)
Node version: frontend/.node-version, currently 22.16.0
```

Important public env var:

```text
NEXT_PUBLIC_API_URL=https://ai-reflection-intelligence-platform-eei6.onrender.com
```

Important detail: because this is a static Next export, `NEXT_PUBLIC_API_URL` is baked into the generated frontend bundle at build time. If the value changes in Cloudflare, the site must be rebuilt/redeployed.

### 4.2 Legacy/Local Streamlit App

Location:

```text
app.py
```

Run command:

```bash
make run
```

Equivalent:

```bash
streamlit run app.py
```

The Streamlit app has:

- Chat tab
- Insights Dashboard tab
- File upload
- Debug toggles
- Memory replay expander
- PDF report download

This Streamlit app is still valuable for local development and demos, but the deployed production app is the Next.js frontend plus FastAPI backend.

## 5. Backend Surface

Location:

```text
backend/api/main.py
```

Framework:

```text
FastAPI
```

Hosted on:

```text
Render
```

Runtime:

```text
Docker
uvicorn backend.api.main:app
```

Docker entry:

```text
CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8501}"]
```

Render service URL:

```text
https://ai-reflection-intelligence-platform-eei6.onrender.com
```

## 6. High-Level Data Flow

For one journal entry, the complete flow is:

```text
User journal text
-> FastAPI POST /api/chat or Streamlit chat input
-> RAGService.run_pipeline()
-> CrisisDetector.check()
-> EmotionDetector.detect()
-> TextProcessor.extract()
-> MemoryManager.store_entry()
-> FaissVectorStore.add_entries()
-> JournalDB.insert()
-> Retriever.retrieve()
-> ReplayEngine.find_replay()
-> deterministic analytics engines
-> ProfileManager.update()
-> Orchestrator.assemble()
-> IntelligencePacket
-> PromptBuilder.build()
-> HuggingFaceInferenceClient.generate()
-> Mistral chat completions API
-> response
-> crisis resource prepend if needed
-> latency logged
-> response returned to frontend
```

The returned response object includes:

```text
emotion
response
memory_replay
crisis
retrieved_memories
stored_entry
packet
prompt
```

## 7. FastAPI Endpoints

Defined in:

```text
backend/api/main.py
backend/api/schemas.py
```

### Health

```http
GET /api/health
```

Returns:

```json
{"status":"ok"}
```

Used by Render health checks and manual deployment verification.

### Chat

```http
POST /api/chat
```

Request model:

```text
ChatRequest
```

Fields:

```text
text: non-empty string
chat_history: list of role/content messages
top_k: integer, default 3, min 1, max 20
tags: optional list of strings
```

Response model:

```text
ChatResponse
```

Fields:

```text
emotion
response
memory_replay
crisis
retrieved_memories
stored_entry
packet
prompt
```

### Dashboard Summary

```http
GET /api/dashboard/summary?range=Last%2030%20days
```

Supported ranges are normalized from:

```text
7
7 days
last 7 days
30
30 days
last 30 days
all
all time
```

Returns:

```text
range
lookback_days
emotion_over_time
pattern_summary
recurring_topics
triggers
habits
relationships
insights
```

### Goals

```http
GET /api/dashboard/goals
```

Returns goal progress estimates from `GoalEngine`.

### Predictions

```http
GET /api/dashboard/predictions
```

Returns:

```text
sentiment_forecast
burnout_risk
```

The burnout/stress output is statistical only and not clinical.

### Timeline

```http
GET /api/dashboard/timeline
```

Returns significant journal timeline events.

### Growth

```http
GET /api/dashboard/growth
```

Returns monthly growth snapshots and a narrative.

### Knowledge Graph Query

```http
GET /api/graph/query?node=User
```

Builds an in-memory NetworkX graph and returns neighbors and edge data for the requested node.

### Weekly PDF Report

```http
GET /api/report/weekly
```

Returns a PDF stream generated by `ReportGenerator`.

### Diagnostics

```http
GET /api/diagnostics
```

Returns:

```text
retrieval_precision
emotion_confidence
latency
```

### CORS Preflight

Browser preflight is handled by FastAPI `CORSMiddleware`.

There is also an explicit fallback:

```http
OPTIONS /{path:path}
```

A regression test now verifies that Cloudflare Pages origins can preflight `OPTIONS /api/chat`.

## 8. CORS and Deployment Wiring

This was a real debugging area. Preserve these facts.

The frontend must call the Render backend, not localhost and not a Cloudflare-relative `/api/chat` path.

Correct frontend env var:

```text
NEXT_PUBLIC_API_URL=https://ai-reflection-intelligence-platform-eei6.onrender.com
```

Wrong examples:

```text
NEXT_PUBLIC_API_URL=https://ai-reflection-intelligence-platform-eei6.onrender.com/api
NEXT_PUBLIC_API_URL=http://127.0.0.1:8501
```

Backend CORS vars on Render:

```text
ALLOWED_ORIGIN=https://mind-shift-ai.pages.dev
ALLOWED_ORIGIN_REGEX=https://.*\.pages\.dev
```

`ALLOWED_ORIGIN` is the frontend origin, not the Render URL.

`NEXT_PUBLIC_API_URL` is the backend origin, not the Cloudflare URL.

The production frontend fallback in `frontend/src/lib/api.ts` is intentionally set to the Render backend URL so a missing Cloudflare env var does not fall back to localhost.

Current frontend API base logic:

```text
configured API URL = process.env.NEXT_PUBLIC_API_URL trimmed and trailing slash removed
production fallback = https://ai-reflection-intelligence-platform-eei6.onrender.com
development fallback = http://127.0.0.1:8501
```

If DevTools shows requests going to `127.0.0.1:8501`, the deployed bundle is stale or the build did not include the recent frontend code.

If DevTools shows requests going to `https://*.pages.dev/api/chat`, the frontend is not using the backend URL and is likely doing a relative call or serving an old bundle.

If DevTools shows requests going to Render but preflight fails, check Render CORS env vars and redeploy/restart Render.

Live preflight was verified from this workspace against Render for both:

```text
https://mind-shift-ai.pages.dev
https://abc123.mind-shift-ai.pages.dev
```

Both returned `200` for `OPTIONS /api/chat` with allowed origin/method/header values.

## 9. Core Backend Orchestrator

Primary file:

```text
backend/api/rag_service.py
```

Main class:

```text
RAGService
```

Primary method:

```text
run_pipeline()
```

`RAGService` wires together:

- `FaissVectorStore`
- `EmbeddingPipeline`
- `EmotionDetector`
- `Retriever`
- `PromptBuilder`
- `HuggingFaceInferenceClient`
- `MemoryManager`
- `JournalDB`
- `TextProcessor`
- `PatternEngine`
- `HabitEngine`
- `RelationshipEngine`
- `InsightEngine`
- `CrisisDetector`
- `ReflectionEngine`
- `ReportGenerator`
- `ProfileManager`
- `Orchestrator`
- `TemporalEngine`
- `CausalEngine`
- `AlertEngine`
- `PredictionEngine`
- `GoalEngine`
- `ReplayEngine`
- `KnowledgeGraph`
- `TimelineEngine`
- `GrowthTracker`
- `EvalEngine`

Design detail: `RAGService` accepts injected components. This is important for Streamlit caching and tests. If dependencies are not injected, it creates default components from settings.

Defensive behavior: many helper methods swallow/log exceptions rather than crashing the chat pipeline. Examples:

- `_persist_journal_record`
- `_safe_insights`
- `_safe_reflection`
- `_assemble_packet`
- `_log_latency`

## 10. Configuration

Primary file:

```text
backend/config/settings.py
```

The settings dataclass reads environment variables through `dotenv`.

Important variables:

```text
DATA_DIR
VECTOR_STORE_DIR
SQLITE_PATH
LATENCY_LOG_PATH
EMBEDDING_MODEL
MISTRAL_MODEL
HF_MODEL
MISTRAL_API_KEY
HF_API_TOKEN
HF_MAX_NEW_TOKENS
HF_TIMEOUT_S
HF_TEMPERATURE
CHUNK_SIZE
CHUNK_OVERLAP
STM_SIZE
SESSION_STM_SIZE
EMOTION_MODEL
RETRIEVAL_SEMANTIC_WEIGHT
RETRIEVAL_EMOTION_WEIGHT
RETRIEVAL_RECENCY_WEIGHT
RETRIEVAL_HALF_LIFE_HOURS
RETRIEVAL_CANDIDATE_POOL
LTM_MAX_ENTRIES
```

Defaults:

```text
DATA_DIR=data
VECTOR_STORE_DIR=faiss_store
SQLITE_PATH=data/journal.db
LATENCY_LOG_PATH=data/latency_log.jsonl
EMBEDDING_MODEL=all-MiniLM-L6-v2
MISTRAL_MODEL/HF_MODEL fallback=mistral-small
HF_MAX_NEW_TOKENS=220
HF_TIMEOUT_S=30
HF_TEMPERATURE=0.2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
STM_SIZE=10
SESSION_STM_SIZE=8
EMOTION_MODEL=SamLowe/roberta-base-go_emotions
RETRIEVAL_SEMANTIC_WEIGHT=0.6
RETRIEVAL_EMOTION_WEIGHT=0.25
RETRIEVAL_RECENCY_WEIGHT=0.15
RETRIEVAL_HALF_LIFE_HOURS=72
RETRIEVAL_CANDIDATE_POOL=20
LTM_MAX_ENTRIES=0
```

Token selection:

```text
hf_api_token = MISTRAL_API_KEY or HF_API_TOKEN
hf_model = MISTRAL_MODEL or HF_MODEL or mistral-small
```

Historical naming quirk: the LLM client class is still named `HuggingFaceInferenceClient`, but it currently calls Mistral.

## 11. Production Versus Local Runtime Modes

The project supports two broad runtime modes.

### Full Local ML Mode

Used for richer local demos/dev:

```text
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMOTION_MODEL=SamLowe/roberta-base-go_emotions
```

Requires:

```bash
pip install -r requirements-ml.txt
python -m spacy download en_core_web_sm
```

Characteristics:

- sentence-transformers embeddings
- HuggingFace transformer emotion detection
- heavier model downloads
- larger memory footprint

### Lightweight Production/Demo Mode

Used on Render free tier:

```text
EMBEDDING_MODEL=hashing
EMOTION_MODEL=rule-based
```

Characteristics:

- deterministic hashing embeddings
- rule-based emotion detector
- lower memory footprint
- faster cold start
- less semantically rich than transformer-backed mode

Production Render config in `render.yaml` uses `/tmp/mind-shift-ai` storage because free-tier Render does not provide persistent disks.

## 12. Embedding Pipeline

File:

```text
backend/embedding/pipeline.py
```

Class:

```text
EmbeddingPipeline
```

Default model:

```text
all-MiniLM-L6-v2
```

Lightweight fallback model names:

```text
hashing
hashing-384
lightweight
```

If a lightweight model name is used, `_HashingEmbeddingModel` creates deterministic 384-dimensional vectors from token hashes.

If sentence-transformers is unavailable, the pipeline logs a warning and uses hashing embeddings.

Chunking:

```text
RecursiveCharacterTextSplitter
chunk_size from settings
chunk_overlap from settings
separators = ["\n\n", "\n", " ", ""]
```

Embeddings are returned as `float32` numpy arrays, compatible with FAISS.

## 13. Emotion Detection

File:

```text
backend/emotion/detector.py
```

Class:

```text
EmotionDetector
```

Default full model:

```text
SamLowe/roberta-base-go_emotions
```

Supports GoEmotions labels such as:

```text
admiration
amusement
anger
annoyance
approval
caring
confusion
curiosity
desire
disappointment
disapproval
disgust
embarrassment
excitement
fear
gratitude
grief
joy
love
nervousness
optimism
pride
realization
relief
remorse
sadness
surprise
neutral
```

Output contract:

```json
{
  "emotion": "joy",
  "confidence": 0.9,
  "all_emotions": [
    {"emotion": "joy", "score": 0.9}
  ]
}
```

Rule-based mode is used when:

```text
EMOTION_MODEL=rule-based
EMOTION_MODEL=rules
EMOTION_MODEL=lightweight
```

Rule-based labels include:

```text
joy
optimism
fear
sadness
anger
stress
neutral
```

If transformer import or inference fails, the detector logs the issue and falls back to rule-based detection.

## 14. NLP Text Processing

File:

```text
backend/nlp/text_processor.py
```

Class:

```text
TextProcessor
```

Uses:

- spaCy `en_core_web_sm`
- VADER sentiment
- deterministic topic keyword buckets
- deterministic habit keyword buckets

Extracted fields:

```text
entities_people
entities_places
entities_orgs
keywords
topics
habits
sentiment_compound
sentiment_valence
```

Topic buckets:

```text
career
health
relationship
money
education
```

Habit buckets:

```text
exercise
sleep
reading
meditation
social_media
coffee
cooking
coding
```

If extraction fails, the processor returns empty lists and zero sentiment. It should not crash the pipeline.

## 15. Memory System

Memory is split into short-term memory and long-term memory.

### Short-Term Memory

File:

```text
backend/memory/manager.py
```

Short-term memory is a bounded `deque`.

In Streamlit, the deque is session-local via `st.session_state`.

Default sizes:

```text
STM_SIZE=10
SESSION_STM_SIZE=8
```

Short-term memory gives the current session conversational continuity.

### Memory Entry Schema

File:

```text
backend/memory/schema.py
```

Memory entries include:

```text
text
emotion
emotion_intensity
tags
topics
timestamp
```

### Long-Term Memory

File:

```text
backend/retrieval/vector_store.py
```

Class:

```text
FaissVectorStore
```

Uses:

```text
FAISS IndexFlatL2
metadata.pkl
thread lock for add/query/save/load
```

Default local paths:

```text
faiss_store/faiss.index
faiss_store/metadata.pkl
```

Render paths:

```text
/tmp/mind-shift-ai/faiss_store/faiss.index
/tmp/mind-shift-ai/faiss_store/metadata.pkl
```

Deduplication:

```text
sha256(normalized lowercased text)
```

Timestamps are stored in metadata but intentionally ignored for deduplication. Identical text will update short-term memory but not create another long-term vector entry.

`LTM_MAX_ENTRIES=0` means unlimited.

## 16. Structured Storage

Files:

```text
backend/storage/db.py
backend/storage/models.py
```

Database:

```text
SQLite
```

Default local path:

```text
data/journal.db
```

Render path:

```text
/tmp/mind-shift-ai/data/journal.db
```

Main table:

```text
journal_records
```

Columns:

```text
id
text
timestamp
emotion
emotion_confidence
entities_people
entities_places
entities_orgs
keywords
topics
habits
sentiment_compound
sentiment_valence
```

List fields are JSON-encoded strings.

Insert behavior:

```text
upsert by id
```

The id matches the FAISS entry hash. This keeps structured DB records aligned with memory deduplication.

Migrations:

- `habits` is added idempotently if missing.

All DB calls are defensive: storage failure logs the error but should not crash the chat pipeline.

## 17. Retrieval

File:

```text
backend/retrieval/retriever.py
```

Class:

```text
Retriever
```

Retrieval uses FAISS candidates plus hybrid reranking.

Formula:

```text
combined = 0.60 * semantic + 0.25 * emotion + 0.15 * recency
```

Defaults are configurable:

```text
RETRIEVAL_SEMANTIC_WEIGHT=0.6
RETRIEVAL_EMOTION_WEIGHT=0.25
RETRIEVAL_RECENCY_WEIGHT=0.15
RETRIEVAL_HALF_LIFE_HOURS=72
RETRIEVAL_CANDIDATE_POOL=20
```

Semantic score:

```text
normalized from FAISS L2 distance:
s = 1 - ((d - d_min) / (d_max - d_min))
```

Emotion similarity:

```text
1.0 exact match
0.6 same valence family
0.5 if either side is neutral
0.0 otherwise
```

Recency:

```text
0.5 ** (age_hours / half_life_hours)
```

The final retrieved payload includes `scores.semantic`, `scores.emotion`, `scores.recency`, and `scores.combined`.

## 18. Memory Replay

File:

```text
backend/memory/replay_engine.py
```

Purpose:

Find a similar older experience and, when possible, describe what happened next.

Behavior:

- searches FAISS for similar memories
- skips entries from the last 48 hours
- chooses a good older match
- checks entries 1 to 3 days after the match
- creates a recovery hint
- returns `None` if there is not enough memory

The UI exposes this as a "Memory replay" detail/expander.

## 19. Safety System

File:

```text
backend/safety/crisis_detector.py
```

Class:

```text
CrisisDetector
```

Runs first in the pipeline.

It uses conservative local regex patterns for high-risk crisis/self-harm language.

If flagged:

- the pipeline still continues
- the journal entry is still stored
- response includes crisis resources
- the output `crisis.flagged` is true
- `crisis.matched_terms` contains matched phrases

The crisis message references emergency services and 988 for the US. The code notes that real production deployments should localize/review this professionally.

Important product disclaimer:

```text
This project is a reflection tool, not a medical or psychological diagnosis.
```

## 20. Analytics Engines

Folder:

```text
backend/analytics/
```

All analytics are deterministic and local. They read from SQLite journal records.

Most output models include:

```text
confidence
explanation
```

Confidence generally uses simple evidence-count style scoring, for example `min(1, count / 10)`.

### Shared Helpers

File:

```text
backend/analytics/_stats_utils.py
```

Provides:

```text
parse_ts
sort_key
filter_window
half_split_trend
```

### Shared Models

File:

```text
backend/analytics/models.py
```

Includes:

```text
TriggerStat
PatternSummary
compute_confidence
```

### PatternEngine

File:

```text
backend/analytics/pattern_engine.py
```

Finds:

```text
recurring emotions
recurring topics
recurring people
per-topic triggers
period entry count
```

Triggers include:

```text
topic
frequency
avg_sentiment
dominant_emotion
trend
confidence
explanation
```

### HabitEngine

File:

```text
backend/analytics/habit_engine.py
```

Finds correlations between habit mentions and sentiment.

Output model:

```text
HabitCorrelation
```

Fields include:

```text
habit
mention_count
avg_sentiment_when_mentioned
avg_sentiment_other_days
delta
correlation_label
confidence
explanation
```

### RelationshipEngine

File:

```text
backend/analytics/relationship_engine.py
```

Tracks emotional patterns around named people.

Output model:

```text
RelationshipProfile
```

Fields include:

```text
person
mention_count
avg_sentiment
dominant_emotion
last_mentioned
trend
confidence
explanation
```

### TemporalEngine

File:

```text
backend/analytics/temporal_engine.py
```

Finds time/day patterns for topics when enough data exists.

### CausalEngine

File:

```text
backend/analytics/causal_engine.py
```

Computes simple lift-style associations.

Examples:

- positive mood given a habit same/next day
- negative mood given a stressor topic

This is statistical patterning, not true causal proof.

### AlertEngine

File:

```text
backend/analytics/alert_engine.py
```

Produces proactive alerts such as:

- consecutive stress
- trigger spike
- positive streak
- habit absence

Severity examples:

```text
watch
info
```

### PredictionEngine

File:

```text
backend/analytics/prediction_engine.py
```

Produces:

```text
SentimentForecast
BurnoutRisk
```

Uses deterministic/statistical logic, including numpy linear fitting.

Burnout/stress risk always carries a disclaimer:

```text
This is a statistical pattern only, not a clinical assessment.
```

### GoalEngine

File:

```text
backend/analytics/goal_engine.py
```

Tracks goal-like keywords and estimates progress based on mention frequency and sentiment trend.

Output:

```text
GoalProgress
```

### TimelineEngine

File:

```text
backend/analytics/timeline_engine.py
```

Builds significant journal events for "My Journey".

### GrowthTracker

File:

```text
backend/analytics/growth_tracker.py
```

Creates monthly snapshots:

```text
period_label
entry_count
avg_sentiment
dominant_emotion
top_topic
snapshot_date
```

Also produces a narrative.

### InsightEngine

File:

```text
backend/analytics/insight_engine.py
```

Generates template-based human-readable insights from patterns, habits, and relationships.

### ReflectionEngine

File:

```text
backend/analytics/reflection_engine.py
```

Generates up to a small number of rule-based reflective questions.

It can use `MemoryReplay` to personalize one question from a similar prior experience.

## 21. User Profile System

Files:

```text
backend/profile/models.py
backend/profile/profile_manager.py
```

The profile is persisted in the same SQLite DB in a `user_profiles` table.

Single-row profile id:

```text
default
```

Profile fields include:

```text
user_id
baseline_sentiment
current_sentiment
dominant_emotion
recovery_speed_days
top_triggers
top_habits
top_people
entry_count
last_updated
growth_score
communication_style
```

Communication style is inferred from average entry length:

```text
brief
reflective
detailed
```

Recovery speed is estimated from negative sentiment entries followed by positive entries.

Profile update is defensive and returns last known profile on failure.

## 22. Intelligence Packet and Orchestrator

Files:

```text
backend/orchestrator/packet.py
backend/orchestrator/orchestrator.py
```

Main model:

```text
IntelligencePacket
```

Fields:

```text
current_entry_emotion
current_entry_sentiment
insights
reflection_prompts
triggers
habits
relationships
user_profile
proactive_alerts
temporal_patterns
causal_links
predictions
goals
memory_replay
```

The orchestrator does not run engines. It only packages already-computed values. If assembly fails, it returns a minimal valid packet.

## 23. Prompt Builder

File:

```text
backend/llm/prompt_builder.py
```

Class:

```text
PromptBuilder
```

The prompt includes:

- assistant role
- safety and style rules
- current emotion
- detected memory patterns
- long-term insights
- user profile snapshot
- memory replay
- optional reflection questions
- relevant past entries
- recent chat history
- current user message

Final instruction asks the LLM to respond in 4 to 7 sentences, validate emotion, include one suggestion, and avoid encouraging extreme actions.

The prompt builder is also where retrieved memories are turned into readable context.

## 24. LLM Client

File:

```text
backend/llm/huggingface_client.py
```

Class:

```text
HuggingFaceInferenceClient
```

Historical name: despite the class name, the active implementation calls Mistral.

Endpoint:

```text
https://api.mistral.ai/v1/chat/completions
```

Model:

```text
settings.hf_model
```

Default:

```text
mistral-small
```

Auth:

```text
Authorization: Bearer <MISTRAL_API_KEY or HF_API_TOKEN>
```

Payload:

```json
{
  "model": "mistral-small",
  "messages": [
    {"role": "system", "content": "You are an empathetic journaling assistant."},
    {"role": "user", "content": "<constructed prompt>"}
  ],
  "temperature": 0.2,
  "max_tokens": 220
}
```

On failure, the client logs the exception and returns:

```text
I'm here with you. Tell me more about what you're feeling.
```

## 25. Knowledge Graph

File:

```text
backend/graph/knowledge_graph.py
```

Graph library:

```text
networkx
```

Graph is built in memory from SQLite journal records.

Nodes can include:

```text
User
topics
people
habits
places
organizations
```

Edges include:

```text
User -> topic
User -> person
topic -> person
topic -> habit
```

Edge data includes:

```text
weight
sentiment
type
neighbor
```

Used by the dashboard's knowledge graph search.

## 26. Reports

File:

```text
backend/reports/report_generator.py
```

Library:

```text
fpdf2
```

Output:

```text
weekly_report.pdf bytes
```

Sections:

- Weekly Reflection Report title
- date range and generation date
- Your Story This Week
- Emotional Summary
- Top Triggers
- Habits & Mood
- People
- Insights
- Predictions
- medical/diagnostic disclaimer

The PDF generator uses defensive ASCII/Latin-1 conversion because fpdf2 core fonts are limited.

## 27. Evaluation and Diagnostics

File:

```text
backend/evaluation/eval_engine.py
```

Used by the dashboard diagnostics endpoint.

Metrics:

```text
retrieval_precision_at_k
emotion_confidence_stats
latency_summary
```

Latency log path:

```text
data/latency_log.jsonl
```

Render:

```text
/tmp/mind-shift-ai/data/latency_log.jsonl
```

Retrieval precision is a topic-overlap proxy. It is not a formal human-labeled benchmark.

## 28. Frontend Structure

Folder:

```text
frontend/
```

Important files:

```text
frontend/package.json
frontend/next.config.mjs
frontend/wrangler.toml
frontend/.node-version
frontend/src/lib/api.ts
frontend/src/lib/format.ts
frontend/src/app/layout.tsx
frontend/src/app/page.tsx
frontend/src/app/chat/page.tsx
frontend/src/app/dashboard/page.tsx
frontend/src/app/globals.css
frontend/src/components/AppShell.tsx
frontend/src/components/Card.tsx
frontend/src/components/EmptyState.tsx
frontend/src/components/ProgressBar.tsx
frontend/tailwind.config.ts
```

`next.config.mjs`:

```text
output: "export"
reactStrictMode: true
agentRules: false
```

`package.json` scripts:

```text
dev: next dev
build: next build
pages:build: next build
lint: eslint .
```

API wrapper:

```text
frontend/src/lib/api.ts
```

Exports:

```text
sendChat
getDashboardSummary
getGoals
getPredictions
getTimeline
getGrowth
queryGraph
getDiagnostics
weeklyReportUrl
```

Types mirror backend schemas:

```text
ChatMessage
EmotionResult
ChatResponse
DashboardSummary
GoalProgress
SentimentForecast
BurnoutRisk
TimelineEvent
GrowthSnapshot
GraphQuery
Diagnostics
```

Chat page:

```text
frontend/src/app/chat/page.tsx
```

Features:

- user journal textarea
- send button
- reset
- attach text file
- debug context toggle
- displays assistant/user messages
- displays emotion signal and secondary emotions
- displays memory replay details
- displays retrieved context and prompt when debug is enabled

Dashboard page:

```text
frontend/src/app/dashboard/page.tsx
```

Features:

- range selector
- mood direction
- stress pattern
- entry window stats
- emotional rhythm chart
- top topics
- triggers
- habits
- people
- goal progress
- knowledge graph search
- journey timeline
- growth over time
- insights list
- diagnostics
- weekly PDF link

## 29. Legacy Streamlit Structure

File:

```text
app.py
```

Important functions:

```text
_get_shared_components()
_init_session_state()
_append_chat()
_format_secondary_emotions()
_render_memory_replay()
_save_uploaded_file()
_emotion_over_time_df()
_render_dashboard()
_render_chat()
```

Key Streamlit choices:

- Heavy resources cached by `st.cache_resource`
- STM and chat history are session state
- RAGService is per session
- vector store loaded or built once per process
- uploaded files saved to `data/text_files`

Known limitation: uploaded files are saved but not automatically reindexed into FAISS unless the vector store is rebuilt.

## 30. Ingestion

File:

```text
backend/ingestion/loaders.py
```

Purpose:

Load documents from `DATA_DIR` for initial vector store build.

Supported formats are described by README/project summary as including:

```text
pdf
txt
csv
docx
xlsx
json
```

The RAG service uses `load_all_documents(settings.data_dir)` when a vector store does not yet exist.

## 31. Docker and Render Deployment

Dockerfile:

```text
Dockerfile
```

Important details:

- base image: `python:3.11-slim`
- working directory: `/app`
- installs `libgomp1` for FAISS/numeric wheels
- installs `requirements.txt`
- downloads `en_core_web_sm`
- copies repo
- exposes `8501`
- starts uvicorn for FastAPI

Render config:

```text
render.yaml
```

Service:

```text
type: web
name: ai-reflection-intelligence-platform
runtime: docker
plan: free
dockerfilePath: ./Dockerfile
dockerContext: .
autoDeployTrigger: commit
healthCheckPath: /api/health
```

Render env vars from `render.yaml`:

```text
ALLOWED_ORIGIN_REGEX=https://.*\.pages\.dev
MISTRAL_MODEL=mistral-small
EMBEDDING_MODEL=hashing
EMOTION_MODEL=rule-based
DATA_DIR=/tmp/mind-shift-ai/data
VECTOR_STORE_DIR=/tmp/mind-shift-ai/faiss_store
SQLITE_PATH=/tmp/mind-shift-ai/data/journal.db
LATENCY_LOG_PATH=/tmp/mind-shift-ai/data/latency_log.jsonl
PYTHONUNBUFFERED=1
```

Manual Render env vars:

```text
MISTRAL_API_KEY=<secret>
ALLOWED_ORIGIN=https://mind-shift-ai.pages.dev
```

Do not put the Mistral key in Cloudflare. Browser code must never receive the secret.

Render free tier storage is ephemeral. Journal history and FAISS memory can reset on restart/redeploy.

For persistent storage, use paid Render disk and adjust:

```text
DATA_DIR=/var/data/data
VECTOR_STORE_DIR=/var/data/faiss_store
SQLITE_PATH=/var/data/data/journal.db
LATENCY_LOG_PATH=/var/data/data/latency_log.jsonl
```

## 32. Cloudflare Pages Deployment

Cloudflare project:

```text
mind-shift-ai
```

Config file:

```text
frontend/wrangler.toml
```

Important values:

```text
name = "mind-shift-ai"
pages_build_output_dir = "out"
```

Cloudflare settings:

```text
Production branch: main
Root directory: frontend
Framework preset: Next.js (Static HTML Export)
Build command: npm run pages:build
Build output directory: out
```

Required Cloudflare env var:

```text
NEXT_PUBLIC_API_URL=https://ai-reflection-intelligence-platform-eei6.onrender.com
```

This variable must be configured for the environment being deployed:

- Production
- Preview, if testing a preview deployment

After changing it, rebuild/redeploy Cloudflare.

## 33. Makefile Commands

File:

```text
Makefile
```

Commands:

```bash
make help
make install
make install-dev
make run
make seed
make test
make lint
make typecheck
make clean
```

`make install`:

- installs runtime requirements
- downloads spaCy model

`make install-dev`:

- installs dev requirements
- downloads spaCy model

`make run`:

- starts Streamlit app

`make seed`:

- runs `scripts/seed_demo_data.py`

`make test`:

- runs `pytest -v`

`make clean`:

- removes generated DB and FAISS store

Note: `make clean` uses Unix-style `rm`; on Windows PowerShell this may depend on shell compatibility.

## 34. Seed Data

File:

```text
scripts/seed_demo_data.py
```

Purpose:

Populate demo data through the real pipeline while avoiding a real LLM call.

Behavior described by docs:

- creates backdated synthetic journal entries
- resets existing DB and FAISS store
- backdates SQLite and FAISS timestamps so replay/trends work
- prints evaluation/growth output
- includes a crisis false-positive sanity assertion

## 35. Testing

Test folder:

```text
tests/
```

Coverage areas:

```text
analytics
api
emotion
evaluation
graph
integration
llm
memory
nlp
orchestrator
profile
reports
retrieval
safety
storage
```

Recent API tests include:

- health endpoint
- chat happy path
- chat empty text validation
- crisis pass-through
- dashboard/support endpoints happy path
- empty-data dashboard/support path
- Cloudflare Pages CORS preflight for `/api/chat`

Command used recently:

```bash
.venv\Scripts\python.exe -m pytest tests/api/test_main.py -q
```

Recent result:

```text
7 passed
```

Frontend build command used recently:

```bash
npm run pages:build
```

Recent result:

```text
Next.js static build passed
```

Docs still mention "99 tests passing" in places; if reporting exact counts, run the full test suite first because the current count may have changed.

## 36. CI

Workflow:

```text
.github/workflows/tests.yml
```

The README/project docs describe CI as:

- install dev dependencies
- install spaCy model
- run ruff
- run mypy
- run pytest

Tests mock the LLM path, so CI should not require a real API key.

## 37. Requirements Files

Important files:

```text
requirements.txt
requirements-dev.txt
requirements-ml.txt
pyproject.toml
uv.lock
```

`requirements.txt` is optimized for runtime/deployment. Production uses lightweight embedding/emotion modes on Render.

`requirements-ml.txt` contains heavier ML dependencies for transformer-backed local operation.

`requirements-dev.txt` includes test/lint/typecheck dependencies.

`pyproject.toml` includes project metadata and broader Python dependencies for the workspace.

## 38. Data Stores and Generated Files

Default local generated paths:

```text
data/journal.db
data/latency_log.jsonl
data/text_files/
faiss_store/faiss.index
faiss_store/metadata.pkl
```

Render generated paths:

```text
/tmp/mind-shift-ai/data/journal.db
/tmp/mind-shift-ai/data/latency_log.jsonl
/tmp/mind-shift-ai/faiss_store/faiss.index
/tmp/mind-shift-ai/faiss_store/metadata.pkl
```

These are runtime data, not source-of-truth code.

Do not commit secrets, `.env`, generated DBs, or local FAISS stores.

## 39. Important Source Files by Responsibility

```text
app.py
```

Legacy/local Streamlit UI.

```text
backend/api/main.py
```

FastAPI REST API, CORS, dependency-injected singleton RAGService, endpoint definitions.

```text
backend/api/schemas.py
```

Pydantic API request/response models.

```text
backend/api/rag_service.py
```

Main pipeline orchestration.

```text
backend/config/settings.py
```

Environment-backed settings.

```text
backend/embedding/pipeline.py
```

Transformer or hashing embeddings and document chunking.

```text
backend/emotion/detector.py
```

Transformer or rule-based emotion detection.

```text
backend/llm/huggingface_client.py
```

Mistral API generation client, historical class name.

```text
backend/llm/prompt_builder.py
```

Constructs final prompt from memory, insights, packet, and current message.

```text
backend/retrieval/vector_store.py
```

FAISS index and metadata store.

```text
backend/retrieval/retriever.py
```

Hybrid semantic/emotion/recency reranking.

```text
backend/memory/manager.py
```

STM and LTM storage interface.

```text
backend/memory/replay_engine.py
```

Similar past experience replay.

```text
backend/storage/db.py
```

SQLite journal DB and migration/upsert logic.

```text
backend/nlp/text_processor.py
```

Entities, keywords, topics, habits, sentiment extraction.

```text
backend/analytics/
```

All deterministic analytics engines.

```text
backend/profile/
```

Persistent user profile.

```text
backend/orchestrator/
```

IntelligencePacket assembly.

```text
backend/graph/knowledge_graph.py
```

NetworkX personal knowledge graph.

```text
backend/reports/report_generator.py
```

Weekly PDF generation.

```text
frontend/src/lib/api.ts
```

Frontend API base URL and endpoint wrappers.

```text
frontend/src/app/chat/page.tsx
```

Deployed chat UI.

```text
frontend/src/app/dashboard/page.tsx
```

Deployed dashboard UI.

## 40. Known Quirks and Footguns

1. `HuggingFaceInferenceClient` actually calls Mistral. The name is historical.

2. `HF_MODEL` and `HF_API_TOKEN` remain supported as fallback names, but current preferred env vars are `MISTRAL_MODEL` and `MISTRAL_API_KEY`.

3. Cloudflare static builds bake `NEXT_PUBLIC_API_URL` into the bundle at build time.

4. Render free-tier storage is ephemeral. User data can reset.

5. Uploaded files in Streamlit are saved to `data/text_files` but are not automatically reindexed unless the vector store is rebuilt.

6. Long-term memory dedup ignores timestamp. Identical text will not create a second vector memory.

7. Full transformer local mode can download large models on first use.

8. Production uses rule-based emotion and hashing embeddings to fit small containers.

9. The project includes both Streamlit and Next.js surfaces. Do not accidentally fix only one UI if the issue is in the deployed frontend.

10. CORS symptoms can be misleading. Always inspect the actual request URL in browser DevTools.

11. A `405 preflight` seen in Cloudflare DevTools may mean the browser is calling Cloudflare `/api/chat` instead of Render `/api/chat`.

12. `out/`, `.next/`, node_modules, `.pytest_cache`, model caches, DBs, and FAISS files are generated/runtime artifacts.

13. Some README/old summary text may contain encoding artifacts from arrows or box drawing characters. Prefer ASCII in new docs.

## 41. Debugging Checklist for API/CORS Issues

1. Verify backend health:

```text
https://ai-reflection-intelligence-platform-eei6.onrender.com/api/health
```

Expected:

```json
{"status":"ok"}
```

2. In browser DevTools, inspect the failed `/api/chat` request URL.

Good:

```text
https://ai-reflection-intelligence-platform-eei6.onrender.com/api/chat
```

Bad:

```text
http://127.0.0.1:8501/api/chat
https://mind-shift-ai.pages.dev/api/chat
```

3. If URL is localhost, Cloudflare served stale bundle or env var was missing at build time.

4. If URL is Cloudflare `/api/chat`, frontend did not use `NEXT_PUBLIC_API_URL` correctly or served stale code.

5. If URL is Render and CORS fails, verify Render env:

```text
ALLOWED_ORIGIN=https://mind-shift-ai.pages.dev
ALLOWED_ORIGIN_REGEX=https://.*\.pages\.dev
```

6. Restart/redeploy Render after changing env vars.

7. Redeploy Cloudflare after changing frontend env vars or frontend code.

8. Confirm generated production frontend does not contain localhost:

```bash
rg "127\.0\.0\.1:8501" frontend/out
```

9. Confirm generated frontend does contain backend URL:

```bash
rg "ai-reflection-intelligence-platform-eei6\.onrender\.com" frontend/out
```

## 42. Local Development Commands

Backend/API local run example:

```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 8501
```

Streamlit local run:

```bash
make run
```

Frontend local run example:

```bash
cd frontend
npm run dev
```

If local frontend talks to local backend, set:

```text
NEXT_PUBLIC_API_URL=http://127.0.0.1:8501
```

Frontend production build:

```bash
cd frontend
npm run pages:build
```

API tests:

```bash
.venv\Scripts\python.exe -m pytest tests/api/test_main.py -q
```

Full test suite:

```bash
make test
```

## 43. Current Git/Deployment History Context

Recent relevant commits:

```text
531ec14 Update README with live deployment links
e334ae0 Fix production API fallback and CORS test
62642e9 Allow Cloudflare Pages CORS origins
7e11a53 Make Render blueprint free-tier compatible
4228827 Add deployment checklist
```

The current README has live app and API health links.

The production frontend fallback to Render backend was added after debugging repeated Cloudflare/localhost/CORS issues.

## 44. What Another Agent Should Do First

When taking over:

1. Run:

```bash
git status --short
```

2. Read:

```text
README.md
DEPLOYMENT.md
AI_AGENT_PROJECT_CONTEXT.md
backend/api/main.py
backend/api/rag_service.py
frontend/src/lib/api.ts
```

3. If working on deployment, inspect:

```text
render.yaml
Dockerfile
frontend/wrangler.toml
frontend/package.json
frontend/next.config.mjs
```

4. If working on pipeline behavior, inspect:

```text
backend/api/rag_service.py
backend/llm/prompt_builder.py
backend/memory/
backend/retrieval/
backend/analytics/
backend/storage/
```

5. Run focused tests before broad tests.

6. Do not revert user changes.

## 45. Safe Change Guidelines

Preserve these architectural constraints:

- Crisis detection must run first.
- Journal/memory storage should not be blocked by non-critical analytics failures.
- The LLM call should remain the final communication layer, not the analytics engine.
- The deployed frontend must call the Render backend origin.
- Secrets must stay on Render/backend, never in Cloudflare frontend code.
- Keep production lightweight defaults compatible with Render free tier.
- Preserve the API response shape unless updating both backend schemas and frontend types.
- Keep CORS tests if modifying backend middleware.
- Do not commit generated `frontend/out`, `.next`, DBs, FAISS stores, or local logs unless explicitly required.

## 46. Security and Privacy Notes

This is a journaling app, so user data is sensitive.

Current implementation is a single-user/demo architecture:

- no multi-user auth
- no per-user data isolation
- local/ephemeral SQLite and FAISS stores
- Render free-tier storage can reset

Do not market the current app as production-grade for private multi-user mental health data.

Do not expose Mistral API keys to the browser.

Do not send journal data to extra external services beyond the configured Mistral generation call unless the user explicitly approves.

## 47. Medical/Safety Scope

This is not a medical product.

Correct framing:

```text
reflection tool
journaling companion
pattern awareness
statistical signals
non-diagnostic insights
```

Incorrect framing:

```text
therapy replacement
clinical diagnosis
mental health treatment
suicide prevention service
medical advice
```

Crisis language detection is simple rule-based matching and should be treated as a safety net, not comprehensive crisis care.

## 48. Concise Interview Pitch

Mind Shift AI is a local-first journaling intelligence platform. A user writes a journal entry, the backend first checks safety, detects emotion, enriches the text with NLP, stores it in FAISS and SQLite, retrieves similar memories using semantic/emotion/recency scoring, runs deterministic analytics engines for patterns and insights, builds a user profile and intelligence packet, then uses one Mistral API call only to phrase a supportive response. The frontend is a static Next.js app on Cloudflare Pages, the backend is FastAPI on Render, and the architecture emphasizes explainability by keeping reasoning in testable Python rather than outsourcing all logic to the LLM.

## 49. Short GitHub Description

Local-first AI journaling assistant with RAG, emotion-aware memory, deterministic analytics, safety checks, FAISS + SQLite storage, FastAPI backend, and Cloudflare-deployed dashboard. Uses one Mistral call to generate empathetic reflections.

## 50. Recruiter Demo Mode

The deployed frontend defaults to `demo` mode for first-time visitors. This is designed
for recruiters and evaluators who should see a populated product immediately instead
of an empty chat/dashboard.

Demo mode behavior:

- `frontend/src/components/DemoModeProvider.tsx` stores `mode: "demo" | "live"` in
  `localStorage`.
- `frontend/src/components/AppShell.tsx` shows a persistent banner and a single CTA
  to switch between demo and live mode.
- In demo mode, the dashboard calls `/api/demo/dashboard/story`,
  `/api/demo/dashboard/timeline`, `/api/demo/diagnostics`, and `/api/demo/graph/people`.
- In demo mode, the chat page calls `/api/demo/chat-history` and renders a read-only
  transcript styled like the real chat UI.
- In live mode, the existing `/api/dashboard/*`, `/api/graph/*`, `/api/diagnostics`,
  and `/api/chat` routes are used.

Static demo data:

```text
backend/demo_data/
```

Important files:

```text
backend/api/demo.py
scripts/generate_demo_snapshot.py
backend/demo_data/dashboard_summary.json
backend/demo_data/dashboard_story.json
backend/demo_data/chat_transcript.json
```

Regeneration:

```bash
make demo-snapshot
```

The generator builds a 30-day synthetic persona in a throwaway SQLite database and
runs the real deterministic analytics engines before writing JSON. The static demo
endpoints do not instantiate `RAGService`, touch SQLite/FAISS, or call Mistral at
request time. This is intentional because Render free-tier storage is ephemeral.

If valid Mistral credentials and network access are available, the snapshot generator
can freeze live-generated sample replies. If not, it marks
`generated_with_live_llm: false` in `chat_transcript.json` and uses the offline
fallback transcript text from the script. This keeps the deployed demo reliable and
honest.
