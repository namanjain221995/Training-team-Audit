FROM python:3.11-slim

# ffmpeg (frames) + OpenCV/MediaPipe runtime libs + OpenMP (onnxruntime) + build tools (insightface)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        build-essential \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# InsightFace builds a Cython extension at install time -> needs numpy + cython present first
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "numpy>=1.24,<2.0" cython

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY analyzer ./analyzer
COPY config ./config

# Model caches live on the mounted volume so they download only once
ENV HF_HOME=/data/.cache
ENV CONFIG_DIR=/app/config

ENTRYPOINT ["python", "-m", "analyzer"]
