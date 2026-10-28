# GPU Planner frontend

React + TypeScript + Tailwind planner for the Python API.

- `npm install` installs the frontend dependencies; `npm test` runs the form-logic tests; `npm run build` type-checks and builds.
- Leave `VITE_API_BASE_URL` blank to use the Vite `/api` proxy, or set it to an API origin for a deployed backend.
- All estimates come from the backend. The former `VITE_MOCK_MODE` demo data was removed with the old estimate contract.
- The form is rendered from `GET /api/schema`; `src/lib/form.ts` holds provenance and merge rules (no silent overwrite of edits).
