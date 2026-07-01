# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Metadane
LABEL maintainer="EnergyForecast MLOps"
LABEL description="Prognoza cen energii RDN — FastAPI + sklearn"

# Zmienne środowiskowe
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Warstwa zależności (kopiowana osobno → szybsze przebudowy gdy zmienia się tylko kod)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Kod aplikacji i frontend
COPY app/ ./app/
COPY frontend/ ./frontend/

# Katalogi danych i modeli (wypełniane przez wolumeny docker-compose)
RUN mkdir -p /app/data /app/models /app/outputs /app/scripts

# Healthcheck używa wbudowanego curl z obrazu slim
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
