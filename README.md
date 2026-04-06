## Journaling AI Architecture

This app runs end-to-end in one Streamlit process and uses:

- Emotion detection
- Memory manager (STM + LTM)
- Hybrid retrieval (semantic + emotion + recency)
- Empathetic prompt builder
- Hugging Face Inference API generation (free tier / configurable)

Pipeline:

`emotion -> memory -> retrieval -> prompt -> llm`

## Hybrid Retrieval Strategy

Final score:

`combined = w_semantic * semantic_norm + w_emotion * emotion_sim + w_recency * recency_norm`

### Components

- **Semantic (`semantic_norm`)**
  - FAISS returns L2 distance.
  - Distances are min-max normalized per candidate set:
  - `semantic_norm = 1 - (d - d_min) / (d_max - d_min)`
- **Emotion (`emotion_sim`)**
  - `1.0` if query emotion == memory emotion
  - `0.5` if either side is neutral
  - `0.0` otherwise
- **Recency (`recency_norm`)**
  - Exponential decay with configurable half-life:
  - `recency_norm = 0.5 ** (age_hours / half_life_hours)`

### Configurable Weights

Use `.env`:

- `RETRIEVAL_SEMANTIC_WEIGHT` (default `0.6`)
- `RETRIEVAL_EMOTION_WEIGHT` (default `0.25`)
- `RETRIEVAL_RECENCY_WEIGHT` (default `0.15`)
- `RETRIEVAL_HALF_LIFE_HOURS` (default `72`)
- `RETRIEVAL_CANDIDATE_POOL` (default `20`)

## Prompt Builder Design

Prompt includes:

- Current user emotion and confidence
- Relevant past entries
- Pattern hints (emotion tendency + simple trigger phrase)
- Recent chat turns
- Safety and tone guidelines

Response rules:

- Validate emotions
- Avoid judgment
- Offer gentle reflection
- Avoid toxic positivity
- Keep tone consistent and safe

## Example Scoring Breakdown

Given:

- weights: semantic=`0.6`, emotion=`0.25`, recency=`0.15`
- candidate memory:
  - semantic_norm=`0.80`
  - emotion_sim=`1.00`
  - recency_norm=`0.40`

Combined:

`0.6*0.80 + 0.25*1.00 + 0.15*0.40 = 0.79`

## Streamlit Caching and State

Caching:

- `st.cache_resource` caches shared heavy components (vector store, embedding model, emotion classifier, HF client) so they load once per process.
- Emotion detection uses an internal lock for thread-safe inference.

Session state:

- `chat_history`: list of conversation turns
- `stm_entries`: session-local bounded deque for last N interactions
- `show_retrieved`: toggles showing retrieved context
- `debug_mode`: toggles showing detected emotion payload + constructed prompt

Reset:

- `Reset Session` clears chat + session STM without rebuilding cached models.
