# Spark model hub (web app + `fetch` / `render` CLI). Multi-arch: python:3.12-slim
# exists for linux/arm64 (DGX Spark) and linux/amd64; every wheel below has
# both. Build context: this directory.
#   docker build -t spark-model-hub docker/spark-stack/model-hub
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-deps -r requirements.txt && pip check

COPY catalog.yml /app/catalog.yml
COPY examples /app/examples
COPY model_hub /app/model_hub

# Runs as the compose `user:` (the owner of MODELS_DIR); 1000:1000 by default.
USER 1000:1000
EXPOSE 8090
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8090/healthz', timeout=2).status == 200 else 1)"]
CMD ["uvicorn", "--factory", "model_hub.app:main_app", "--host", "0.0.0.0", "--port", "8090", "--no-server-header", "--no-proxy-headers"]
