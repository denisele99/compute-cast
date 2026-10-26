# Backend

Start using the root README commands. Python 3.10+.
Optionally copy `.env.example` to `.env`; existing environment variables take precedence.

| File | Purpose |
| --- | --- |
| app/main.py | Create the FastAPI application |
| app/routes.py | HTTP routes and calls to services/clients |
| app/schemas.py | Pydantic input and response schemas |
| app/config.py | Server environment settings |
| app/services/estimation.py | All memory, runtime and pricing functions |
| app/clients/nebius.py | Future inference API client; currently raises NotImplementedError |
| app/clients/aws.py | Future AWS hardware/pricing client; currently raises NotImplementedError |
| tests/test_api.py | Regression checks |

Working endpoints: GET /api/health, POST /api/estimate, GET /api/gpus.
The GPU catalog contains illustrative capacity tiers only, with no live prices.
POST /api/workloads/parse and POST /api/benchmarks return HTTP 501.
The AWS client is a placeholder and is not used by routes yet.
There is no database, job queue or uploaded-code execution.

## Checks

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

Keep transport code in routes, calculations in the service, external API calls in clients.
Split files further only as they grow. For future background training, add an isolated worker.
