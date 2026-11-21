# DeepShield - one image that serves the API and the built React app on a single port.
# NOTE: written but not built/tested here. Build it once locally before relying on it:
#   docker build -t deepshield .
#   docker run -p 7860:7860 -e DF40_WEIGHTS_URL=<direct link to clip.pth> deepshield

# ---- 1) build the React frontend ----
FROM node:20-alpine AS frontend
WORKDIR /fe
COPY deepshield-frontend/package*.json ./
RUN npm ci
COPY deepshield-frontend/ ./
RUN npm run build

# ---- 2) python runtime ----
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=7860
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# CPU-only PyTorch keeps the image much smaller than the default CUDA build
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
COPY --from=frontend /fe/dist ./deepshield-frontend/dist

RUN useradd -m appuser && chown -R appuser /app
USER appuser
EXPOSE 7860

# fetch weights if missing, then start. The model loads ONCE here and stays in memory.
CMD ["sh", "-c", "python scripts/download_weights.py && python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
