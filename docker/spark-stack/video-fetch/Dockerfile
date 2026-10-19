# Video fetch service (yt-dlp behind a small HTTP API) for the Spark stack.
# node is yt-dlp's JavaScript runtime for YouTube; ffmpeg merges the streams.
# Multi-arch (linux/arm64 for the DGX Spark, linux/amd64). Build context: this directory.
#   docker build -t spark-video-fetch docker/spark-stack/video-fetch
FROM node:22-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py /app/app.py

# Runs as the compose `user:` (the owner of the data volume); 1000:1000 by default.
USER 1000:1000
EXPOSE 8095
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8095/healthz', timeout=2).status == 200 else 1)"]
CMD ["uvicorn", "--factory", "app:main_app", "--host", "0.0.0.0", "--port", "8095", "--no-server-header", "--no-proxy-headers"]
