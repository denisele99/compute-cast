# GPU Planner frontend

The frontend is a React + TypeScript + Tailwind planner for the existing Python API.

- `npm install` installs the frontend dependencies.
- Set `VITE_MOCK_MODE=true` in a local `.env` file to preview the complete flow without the Python API. Mock results are labeled Demo data.
- Leave `VITE_API_BASE_URL` blank to use the Vite `/api` proxy, or set it to an API origin for a deployed backend.
- The live integration currently uses `GET /api/gpus` and `POST /api/estimate`.
- Workload parsing and benchmark submission remain visibly unimplemented backend endpoints; the upload only parses JSON in the browser and never executes uploaded Python.
