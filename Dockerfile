# Container for the scoring API. Build: docker build -t churn-api .
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY api ./api
COPY models ./models

# Render/other hosts set $PORT; default to 8000 locally
ENV PORT=8000
ENV MODEL_PATH=/app/models/model.pkl
WORKDIR /app/api
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}