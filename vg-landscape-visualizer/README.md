# VG Landscape Visualizer

Module 2 from the PhotoHDR MV v3 spec. Standalone Next.js tool for showing WTPS / VG clients what their property could look like with different landscaping styles before any work begins.

**No AutoHDR equivalent exists for this** — that's the case for building it.

## Stack

- Next.js 14 App Router + TypeScript + vanilla CSS
- Cloudinary for input/output storage
- Higgsfield Nano Banana Pro Inpaint via HTTPS API
- Same `HIGGSFIELD_SCHEMA_VERIFIED` gate as photo-editor — won't fire until a one-time live smoke test confirms the request/response schema

## Env vars

```
HIGGSFIELD_API_KEY=...
HIGGSFIELD_SCHEMA_VERIFIED=true      # set after smoke-testing the schema
HIGGSFIELD_INPAINT_MODEL=nano-banana-pro-inpaint  # override if needed
CLOUDINARY_CLOUD_NAME=...
CLOUDINARY_API_KEY=...
CLOUDINARY_API_SECRET=...
```

## Dev

```bash
cd vg-landscape-visualizer
npm install
npm run dev
```

## Status

Phase 4 scaffold. Front-end + API route + Higgsfield client are wired. The endpoint won't accept work until `HIGGSFIELD_SCHEMA_VERIFIED=true` — flip that after the first manual integration test confirms the live request schema matches the code.

Future work:
- Auth (Better-Auth PIN, same as photo-editor).
- Save/share results URL with clients.
- Side-by-side before/after comparison.
- Per-region masking (Phase 5? — let user paint over a specific area of the yard to restrict generation).
