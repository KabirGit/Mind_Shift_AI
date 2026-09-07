# Demo Mode Manual QA

## Fresh Load Defaults

1. Clear `localStorage` for the deployed or local frontend origin.
2. Open `/chat`.
3. Confirm the persistent banner shows the `Demo` badge and the CTA
   `Try it yourself with a live entry`.
4. Confirm the chat transcript is already populated from `/api/demo/chat-history`.
5. Confirm the composer is read-only and no `/api/chat` request is made.
6. Confirm the final decision turn is labeled `Guidance`, shows no more than three
   bounded tools, and exposes structured decision evidence.

## Dashboard Defaults

1. Open `/dashboard` with fresh `localStorage`.
2. Confirm the banner still shows `Demo`.
3. Confirm network requests go to:
   - `/api/demo/dashboard/story`
   - `/api/demo/dashboard/timeline`
   - `/api/demo/diagnostics`
   - `/api/demo/graph/people`
4. Confirm the dashboard renders populated hero, working/draining/people sections,
   weekly bars, forecast, goals area, timeline, diagnostics, and graph.

## Recruiter System Proof

1. Open `/observability` in demo mode.
2. Confirm it reports exactly 30 entries across 30 days and a 15/15 golden eval.
3. Confirm reflection, guidance, and safety route contracts are all visible.
4. Confirm the capability matrix covers storage, emotion, patterns, habits,
   relationships, goals, retrieval, guidance, safety, and redacted tracing.
5. Confirm recent traces show only operational metadata and never journal, prompt,
   retrieved-memory, or model-response text.
6. Expand `Inspect all 30 entries` and confirm the dated entries are detailed and
   tell a coherent career, habits, relationships, health, and finance narrative.

## Live Mode Switch

1. Click `Try it yourself with a live entry`.
2. Confirm the badge changes to `Live`.
3. Confirm `/chat` starts empty and the composer is enabled.
4. Submit a live entry and confirm it calls `/api/chat`.
5. Open `/dashboard` and confirm dashboard requests use live `/api/dashboard/*`,
   `/api/diagnostics`, and `/api/graph/*` endpoints.
6. Click `Return to demo` and confirm static demo content returns.
