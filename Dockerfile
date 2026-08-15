FROM python:3.12-slim

# build-essential + cmake: fallback for building llama-cpp-python from source
# if no prebuilt CPU wheel matches this platform.
# fonts-dejavu-core: Unicode font for PDF export.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

COPY backend ./backend
COPY frontend ./frontend

ENV MODELS_DIR=/data/models
VOLUME /data/models

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
