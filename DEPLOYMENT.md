# Mind Shift AI Deployment Checklist

This repo is deployed as two services:

- Backend: FastAPI on Render, using Docker and a persistent disk for SQLite plus FAISS.
- Frontend: static Next.js export on Cloudflare Pages, calling the backend with `NEXT_PUBLIC_API_URL`.

Do not put `MISTRAL_API_KEY` or `HF_API_TOKEN` in Cloudflare. Browser code must only receive the public backend URL.

## 1. Render Backend

Use the GitHub repo `KabirGit/Mind_Shift_AI` and deploy from branch `codex/streamlit-to-frontend-backend`.

Recommended path: Render Blueprint, using the repo root `render.yaml`.

Render service settings from `render.yaml`:

| Setting | Value |
| --- | --- |
| Service type | Web Service |
| Runtime | Docker |
| Dockerfile | `./Dockerfile` |
| Docker context | `.` |
| Health check path | `/api/health` |
| Persistent disk name | `mind-shift-ai-data` |
| Persistent disk mount path | `/var/data` |
| Persistent disk size | `1 GB` |

Render environment variables to set manually:

| Key | Value |
| --- | --- |
| `MISTRAL_API_KEY` | Your Mistral API key from `https://console.mistral.ai` |
| `ALLOWED_ORIGIN` | Your Cloudflare Pages URL, for example `https://mind-shift-ai.pages.dev` |

Render environment variables already defined by `render.yaml`:

| Key | Value |
| --- | --- |
| `MISTRAL_MODEL` | `mistral-small` |
| `EMBEDDING_MODEL` | `hashing` |
| `EMOTION_MODEL` | `rule-based` |
| `DATA_DIR` | `/var/data/data` |
| `VECTOR_STORE_DIR` | `/var/data/faiss_store` |
| `SQLITE_PATH` | `/var/data/data/journal.db` |
| `LATENCY_LOG_PATH` | `/var/data/data/latency_log.jsonl` |
| `PYTHONUNBUFFERED` | `1` |

After Render deploys, verify:

```text
https://YOUR_RENDER_SERVICE.onrender.com/api/health
```

Expected response:

```json
{"status":"ok"}
```

## 2. Cloudflare Pages Frontend

Create a Cloudflare Pages project connected to the same GitHub repo and branch.

Cloudflare Pages build settings:

| Setting | Value |
| --- | --- |
| Project name | `mind-shift-ai` |
| Production branch | `codex/streamlit-to-frontend-backend` |
| Root directory | `frontend` |
| Framework preset | `Next.js (Static HTML Export)` |
| Build command | `npm run pages:build` |
| Build output directory | `out` |
| Node version | Uses `frontend/.node-version` (`22.16.0`) |

Cloudflare Pages environment variables to set:

| Key | Value |
| --- | --- |
| `NEXT_PUBLIC_API_URL` | Your Render backend URL, for example `https://ai-reflection-intelligence-platform.onrender.com` |

Use the backend origin only, with no trailing `/api`. The frontend code appends endpoint paths such as `/api/chat` and `/api/dashboard/summary`.

## 3. Final Cross-Service Wiring

After Cloudflare gives you the live `*.pages.dev` URL, go back to Render and update:

```text
ALLOWED_ORIGIN=https://YOUR_CLOUDFLARE_PROJECT.pages.dev
```

If you need to allow more than one frontend origin, use commas:

```text
ALLOWED_ORIGIN=https://mind-shift-ai.pages.dev,https://preview-url.pages.dev
```

Then redeploy or restart the Render service.

## 4. Smoke Test

1. Open the Cloudflare Pages URL.
2. Visit `/chat`.
3. Send a short non-crisis journal entry.
4. Confirm the response appears.
5. Visit `/dashboard`.
6. Confirm charts and summaries load.
7. Click weekly report download.
8. Confirm the PDF downloads from the Render backend.

If the frontend loads but API calls fail, check these first:

- `NEXT_PUBLIC_API_URL` in Cloudflare must be the Render backend origin.
- `ALLOWED_ORIGIN` in Render must exactly match the Cloudflare Pages origin.
- `/api/health` on Render must return `{"status":"ok"}`.
- `MISTRAL_API_KEY` must exist only on Render.
