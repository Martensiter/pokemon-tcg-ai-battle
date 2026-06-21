# Replay collector image -- pure-Python + numpy + the official Kaggle CLI.
# Multi-arch base (works on aarch64 / ARM as well as x86_64), so the same image
# runs on a Raspberry Pi, a NAS, a mini-PC, or any Docker-capable host.
#
# NO torch and NO engine binary live here -- collection only. Training stays
# offline on a separate machine (see docs/DEPLOY.md).
FROM python:3.11-slim

# Runtime-only deps. PYTHONPATH makes both `collector` (src/) and `agent`
# (repo root, for agent.features) importable without an editable install.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src:/app \
    COLLECTOR_DATA_DIR=/data \
    COLLECTOR_STATE_DIR=/state

WORKDIR /app

# numpy = collector runtime; kaggle = official CLI for real collection.
RUN pip install --no-cache-dir "numpy>=1.24" "kaggle>=1.6"

# Only the bits the collector needs at runtime (keep the image small; secrets and
# raw data are never copied -- see .dockerignore).
COPY src ./src
COPY agent ./agent
COPY pyproject.toml README.md ./

# Run unprivileged; persist data/state via volumes so the manifest survives
# restarts (idempotent + resumable).
RUN useradd -m runner && mkdir -p /data /state && chown -R runner /data /state /app
USER runner
VOLUME ["/data", "/state"]

# Liveness: the collector writes /state/status.json each pass. Consider it
# healthy if that file was updated recently (generous start period for the first
# discovery pass). Tune the freshness window to your COLLECTOR_LOOP_INTERVAL.
HEALTHCHECK --interval=5m --timeout=10s --start-period=20m --retries=3 \
  CMD python -c "import os,time,sys; p=os.path.join(os.environ['COLLECTOR_STATE_DIR'],'status.json'); \
sys.exit(0 if os.path.exists(p) and (time.time()-os.path.getmtime(p) < 3*3600) else 1)"

# Default = long-running loop (run_forever). For a one-off pass:
#   docker run --rm --env-file .env ptcg-collector:latest --once
ENTRYPOINT ["python", "-m", "collector"]
