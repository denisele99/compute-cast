# GPU Planner frontend

React + TypeScript + Tailwind planner for the Python API.

- `npm install` installs the frontend dependencies; `npm test` runs the form-logic tests; `npm run build` type-checks and builds.
- Leave `VITE_API_BASE_URL` blank to use the Vite `/api` proxy, or set it to an API origin for a deployed backend.
- All estimates come from the backend. The former `VITE_MOCK_MODE` demo data was removed with the old estimate contract.
- The form is rendered from `GET /api/schema` (GPT-2 MVP contract); `src/lib/form.ts` holds provenance, merge, visibility and lock rules (no silent overwrite of edits).
- `npm run e2e` drives both flows (upload and manual entry) in Chromium with Playwright against running backend + `npm run dev`. First run `npx playwright install chromium`. See the env vars at the top of `e2e/upload_flow.cjs` (`E2E_CACHE_DIR` must match the backend's `UPLOAD_CACHE_DIR`; set `BENCH_PYTHON` to a Python with torch to include the benchmark run/import step).
