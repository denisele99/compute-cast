"""Future Token Factory adapter (separate from benchmark jobs). No API requests are made by this starter.

Cloud benchmark execution on Nebius AI Cloud Serverless Jobs lives in
app/services/benchmarks/nebius_client.py and uses its own restricted service account, never a
Token Factory key.
"""


def parse_workload(description: str) -> dict:
    # TODO: call the selected NVIDIA model via the provider API; validate output
    # against a workload schema and ask the user to review inferred settings.
    raise NotImplementedError("AI workload parsing is not implemented; use the structured form.")
