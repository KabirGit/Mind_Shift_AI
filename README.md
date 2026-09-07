# Mind Shift AI

Mind Shift AI is a longitudinal personal reflection and decision-support system. It
turns journal entries into explainable emotional, behavioral, relationship, goal, and
memory signals, then uses those signals to produce either a reflective response or a
bounded recommendation for an ordinary life decision.

The product is intentionally positioned as a **personal AI reflection and
decision-support system with longitudinal memory**, not as a therapist, medical
authority, or source of diagnosis.

[Live app](https://mind-shift-ai.pages.dev) | [API health](https://ai-reflection-intelligence-platform-eei6.onrender.com/api/health)

## What the project does

Mind Shift AI combines a local, deterministic analysis layer with one hosted language
model boundary:

- Stores journal entries in SQLite and a FAISS-compatible vector store.
- Detects emotion and sentiment, extracts people, topics, habits, and keywords.
- Retrieves relevant personal history using semantic similarity, emotion compatibility,
  and recency.
- Computes recurring triggers, habits, relationships, temporal patterns, causal
  signals, alerts, forecasts, goals, timeline events, growth, and insights.
- Replays similar past situations and what happened afterward.
- Builds a user profile from longitudinal evidence.
- Routes ordinary entries to reflection, decision-shaped entries to structured guidance,
  and crisis or protected high-risk requests to safety responses.
- Exposes dashboard, graph, report, diagnostics, and observability APIs.

The important design boundary is that safety, retrieval, scoring, reversibility,
context limits, validation, and evaluation are application code. The hosted model
interprets decision text and phrases the final response; it is not the source of
analytics or system guarantees.

## Product surfaces

The deployed frontend is a static Next.js export with three main surfaces:

- **Chat (`/chat`)**: write a journal entry, attach a small text/Markdown/CSV file,
  view emotion metadata, inspect memory replay, and optionally inspect retrieved
  context and the generated prompt.
- **Insights (`/dashboard`)**: review the selected range through a narrative headline,
  evidence-backed insights, helpful and draining signals, people and relationship
  trends, weekly mood buckets, forecasts, goals, a timeline, a relationship graph,
  and a weekly PDF report.
- **Observability (`/observability`)**: inspect the demo system proof, request flow,
  route contracts, model boundary, context policy, capability evidence, redacted
  traces, evaluation coverage, and the synthetic journal evidence.

The home route redirects to `/dashboard`. On a fresh browser, the frontend starts in
**demo mode**. Demo mode is read-only and uses committed JSON snapshots, so recruiters
can explore the product without a live database, vector index, or model credential.
The banner switches the browser to **live mode**, where new entries use the real
backend pipeline.

## Architecture

```text
Next.js static frontend
  -> FastAPI REST API
       -> crisis and protected-risk gates
       -> local emotion + NLP enrichment
       -> SQLite journal record + short/long-term memory
       -> FAISS retrieval with emotion and recency ranking
       -> reflection path OR bounded decision-support path
       -> typed hosted-model call(s)
       -> redacted JSONL trace and latency record
```

### Reflection path

Normal journaling performs crisis detection, emotion detection, NLP enrichment,
persistence, retrieval, memory replay, deterministic analytics, profile updates,
prompt construction, and one final hosted-model completion. The response can include
validation, a gentle suggestion, and a reflective question.

### Guidance path

Conservative decision cues prevent ordinary journaling from paying for a parsing call.
When a message looks like a choice or recommendation request:

1. A structured model call extracts a bounded `DecisionState`.
2. A deterministic, read-only agent selects at most three tools from:
   `search_similar_memories`, `get_emotional_patterns`, `get_user_profile`, and
   `get_recent_decision_context`.
3. Context is deduplicated, source-capped, and packed into a global 1,500-token
   estimate.
4. Options are normalized and scored using goal alignment (30%), historical evidence
   (25%), feasibility (20%), reversibility (15%), and downside containment (10%).
5. Irreversible actions receive a cooling-off guard when negative emotion confidence is
   high.
6. A second structured model call phrases the validated result as validation,
   recommendation, evidence, uncertainty, and two to four next actions.

If evidence is weak, options are too close, or an action is irreversible, the system
does not force a recommendation. It returns uncertainty and information-gathering or
small reversible next steps.

### Safety path

The crisis detector runs first. Protected guidance risks such as medication changes,
immediate danger, domestic violence, severe mental-health symptoms, and similar
high-risk cues bypass guidance tools and recommendations. Crisis responses prepend
calm, non-diagnostic resources; protected-risk responses direct the user to qualified
or urgent support.

## Model and storage boundary

| Capability | Default local/deployed implementation |
| --- | --- |
| Embeddings | Local `all-MiniLM-L6-v2`; Render overrides this to deterministic `hashing` |
| Emotion | Local `SamLowe/roberta-base-go_emotions`; Render overrides this to `rule-based` |
| Sentiment | VADER |
| NLP enrichment | spaCy `en_core_web_sm`, keywords, topic and habit buckets |
| Vector retrieval | FAISS `IndexFlatL2` locally; persisted under `faiss_store/` |
| Structured records | SQLite `journal_records` plus a single user profile |
| Analytics | Pure Python and NumPy |
| Knowledge graph | In-memory NetworkX graph rebuilt from journal records |
| PDF report | `fpdf2` |
| Default hosted model | `arsoban/ocd-therapist-27b-v0.3` via Hugging Face Inference Providers |
| Default provider | Featherless AI through `https://router.huggingface.co/v1/chat/completions` |
| Compatibility model route | Mistral via `LLM_BACKEND=mistral` and `MISTRAL_API_KEY` |

The Hugging Face client uses bounded retries (maximum two attempts), typed request and
response models, JSON-mode validation for guidance, and a deterministic fallback when
the provider is unavailable or returns invalid structured output.

## Demo mode

Committed snapshots under `backend/demo_data/` back these routes:

```text
/api/demo/dashboard/summary
/api/demo/dashboard/story
/api/demo/dashboard/goals
/api/demo/dashboard/predictions
/api/demo/dashboard/timeline
/api/demo/dashboard/growth
/api/demo/graph/people
/api/demo/graph/query
/api/demo/diagnostics
/api/demo/observability
/api/demo/journal-entries
/api/demo/chat-history
```

The default snapshot contains 30 consecutive, detailed synthetic entries covering
career pressure, leadership, partnership, friendship, family care, money, health,
learning, creativity, habits, and eight recurring people. It is generated by the real
analytics and endpoint contracts, not by hand-written dashboard claims. It is demo and
evaluation data, not model fine-tuning data.

Regenerate it with:

```bash
make demo-snapshot
```

To ingest an external 30-entry persona dataset through the real pipeline:

```bash
make ingest-demo-persona DEMO_PERSONA_JSON=C:\path\to\demo_persona_journals.json
```

The ingestion script uses isolated temporary SQLite/FAISS files, backdates records,
exports raw endpoint responses under `debug/demo_persona_raw_output/`, and publishes
the same JSON contracts to `backend/demo_data/`.

## Quick start

### Backend

```bash
git clone https://github.com/KabirGit/Mind_Shift_AI.git
cd Mind_Shift_AI
copy .env.example .env
make install-dev
make run-api
```

The API runs at `http://127.0.0.1:8502` with the Makefile command. The Docker/Render
entry point listens on `PORT` and defaults to `8501`.

### Frontend

In a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open `http://localhost:3000`. Set `NEXT_PUBLIC_API_URL` when the API is not using the
frontend's development fallback.

### Optional local ML stack

The base requirements are enough for the lightweight deployment configuration. For
transformer embeddings and GoEmotions locally:

```bash
pip install -r requirements-ml.txt
```

The first local transformer run may download several hundred megabytes of model data.

## Configuration

Copy `.env.example` to `.env`. The default backend is Hugging Face:

```dotenv
LLM_BACKEND=huggingface
HF_TOKEN=...
HF_MODEL=arsoban/ocd-therapist-27b-v0.3
HF_INFERENCE_PROVIDER=featherless-ai
```

Useful configuration groups include:

- `DATA_DIR`, `SQLITE_PATH`, `VECTOR_STORE_DIR`, `LATENCY_LOG_PATH`
- `EMBEDDING_MODEL`, `EMOTION_MODEL`
- `HF_MAX_NEW_TOKENS`, `HF_TEMPERATURE`, `HF_MAX_ATTEMPTS`, timeout and backoff values
- `MISTRAL_API_KEY` and `MISTRAL_MODEL` for the compatibility backend
- `STM_SIZE`, `SESSION_STM_SIZE`, `LTM_MAX_ENTRIES`
- `RETRIEVAL_*` weights, half-life, and candidate pool
- `GUIDANCE_CONTEXT_TOKEN_BUDGET` and
  `GUIDANCE_CONTEXT_CHARACTERS_PER_TOKEN`
- `DASHBOARD_MIN_INSIGHT_CONFIDENCE`, `DASHBOARD_MIN_MENTION_COUNT`, and
  `DASHBOARD_MIN_ENTRY_COUNT`

Never expose `HF_TOKEN`, `HF_API_TOKEN`, `HUGGINGFACE_API_KEY`, or
`MISTRAL_API_KEY` to the browser or Cloudflare Pages.

## API overview

The FastAPI app is defined in `backend/api/main.py`:

| Route | Purpose |
| --- | --- |
| `GET /api/health` | Health check |
| `POST /api/chat` | Reflection, guidance, or safety pipeline |
| `GET /api/dashboard/summary` | Emotion timeline, patterns, triggers, habits, people, insights |
| `GET /api/dashboard/story` | Narrative dashboard composition |
| `GET /api/dashboard/goals` | Goal progress |
| `GET /api/dashboard/predictions` | Sentiment forecast and statistical burnout signal |
| `GET /api/dashboard/timeline` | Significant journal events |
| `GET /api/dashboard/growth` | Monthly growth snapshots and narrative |
| `GET /api/graph/query` | Search a topic, person, habit, or user node |
| `GET /api/graph/people` | Relationship graph nodes and edges |
| `GET /api/report/weekly` | Download a 30-day PDF report |
| `GET /api/diagnostics` | Retrieval, emotion confidence, latency, and trace metrics |
| `GET /api/observability` | Recruiter-facing operational proof snapshot |

The `/api/demo/*` routes mirror the read-only dashboard and diagnostics contracts
using committed JSON.

## Deployment

- **Frontend**: Cloudflare Pages, static Next.js export from `frontend/out`
- **Backend**: Render Docker web service using `render.yaml`
- **Frontend build**: `npm run pages:build`
- **Backend health**: `/api/health`
- **Public browser setting**: `NEXT_PUBLIC_API_URL` points to the backend origin
- **Backend CORS**: `ALLOWED_ORIGIN` and the Pages regex in `ALLOWED_ORIGIN_REGEX`

Render's free-tier configuration intentionally uses `/tmp/mind-shift-ai`, so SQLite,
FAISS, and latency/trace files are ephemeral across restarts and redeploys. The
frontend's default demo mode avoids depending on that state. Persistent user data
requires a storage plan and the disk configuration described in `DEPLOYMENT.md`.

## Development commands

```text
make install              Install runtime dependencies and spaCy
make install-dev          Install runtime and development dependencies
make run                  Start the legacy Streamlit entry point
make run-api              Start the FastAPI backend
make run-frontend         Start the Next.js frontend
make seed                 Seed local synthetic data through the real pipeline
make demo-snapshot        Regenerate committed demo JSON
make ingest-demo-persona  Ingest and publish an external 30-entry persona
make test                 Run the Python test suite
make lint                 Run Ruff
make typecheck            Run MyPy over backend/
make clean                Remove generated SQLite, FAISS, and latency files
```

`app.py` and `make run` are retained as a legacy Streamlit surface. The deployed
product and current frontend integration use `backend.api.main:app` plus `frontend/`.

## Testing and quality

The repository includes unit, API, integration, safety, storage, retrieval, LLM,
guidance, evaluation, analytics, graph, profile, and report tests. The GitHub Actions
workflow runs:

1. Python 3.11 dependency installation and spaCy model setup.
2. Ruff.
3. MyPy.
4. Pytest.
5. Frontend `npm ci`, ESLint, static export build, and high-severity npm audit.

The test suite mocks provider calls, so CI does not require a hosted-model credential.
Guidance evaluation checks route selection, tool limits, memory inclusion/exclusion,
structured fields, retry behavior, safety precedence, and termination contracts
without using an LLM judge.

## Important limitations

- This is a single-user application model; authentication and multi-user isolation are
  not implemented.
- The default SQLite and FAISS stores are local files. Render free tier is ephemeral.
- The knowledge graph is rebuilt in memory and is not a separate persistent graph DB.
- Burnout and forecast outputs are statistical signals, not clinical assessments.
- Crisis and protected-risk handling is conservative, rule-based, and not a substitute
  for emergency, medical, domestic-violence, or mental-health services.
- Identical normalized journal text is deduplicated in long-term vector memory.
- Uploaded text is placed into the composer; it is not a general-purpose document
  ingestion or multi-user file system.
- The deployed lightweight Render settings trade model richness for container size and
  startup reliability.

See `PROJECT_SUMMARY.md` for the detailed component and data-flow reference,
`DEPLOYMENT.md` for deployment steps, `Intent.md` for product intent, and
`docs/production-genai-architecture.md` for the decision-support hardening design.
