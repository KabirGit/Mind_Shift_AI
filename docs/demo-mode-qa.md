# Demo Mode Manual QA

## Fresh Load Defaults

1. Clear `localStorage` for the deployed or local frontend origin.
2. Open `/chat`.
3. Confirm the persistent banner shows the `Demo` badge and the CTA
   `Try it yourself with a live entry`.
4. Confirm the chat transcript is already populated from `/api/demo/chat-history`.
5. Confirm the composer is read-only and no `/api/chat` request is made.

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

## Live Mode Switch

1. Click `Try it yourself with a live entry`.
2. Confirm the badge changes to `Live`.
3. Confirm `/chat` starts empty and the composer is enabled.
4. Submit a live entry and confirm it calls `/api/chat`.
5. Open `/dashboard` and confirm dashboard requests use live `/api/dashboard/*`,
   `/api/diagnostics`, and `/api/graph/*` endpoints.
6. Click `Return to demo` and confirm static demo content returns.
