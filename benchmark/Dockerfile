# Compute Cast benchmark runner image (same runner as the local package).
# Base: official PyTorch image with CUDA 12.4 / cuDNN 9 runtime and torch 2.5.1.
# For repeatable cloud runs pin it by digest after pulling, e.g.
#   docker pull pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
#   docker inspect --format '{{index .RepoDigests 0}}' pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
# and replace the FROM line with pytorch/pytorch@sha256:<digest>.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BENCH_MAX_SECONDS=300

WORKDIR /app
# Only the trusted runner, registry and model implementations. No credentials, configs or caches.
COPY benchmark.py model_registry.py requirements.txt ./
COPY models/ ./models/

# The configuration is injected at /config/benchmark_config.json (Nebius FileInjection or a local
# bind mount); results.json and benchmark.log are written to /results (a mounted bucket path).
ENTRYPOINT ["python", "/app/benchmark.py"]
CMD ["--config", "/config/benchmark_config.json", "--output-dir", "/results"]
