# Placeholder - full image (frontend build + ffmpeg + models) is built in Phase 8.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend /app/backend
RUN pip install --no-cache-dir -e /app/backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app/backend"]
