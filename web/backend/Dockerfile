# Build context must be the REPO ROOT (not web/backend/) - this image needs
# src/ (the pipeline) and the root requirements.txt, not just this directory.
# On Render: set "Docker Build Context Directory" to the repo root and
# "Dockerfile Path" to web/backend/Dockerfile.
FROM python:3.11-slim

# git: clone_pr_branch shells out to it. semgrep: installed via pip below,
# but it needs a C toolchain on some platforms - slim image ships enough.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements.txt
COPY web/backend/requirements.txt web/backend/requirements.txt
RUN pip install --no-cache-dir -r web/backend/requirements.txt

COPY src/ src/
COPY web/backend/app.py web/backend/app.py

WORKDIR /app/web/backend
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
