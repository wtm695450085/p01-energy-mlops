# PROJECT STATE — EnergyForecast PL MLOps

## Status

| Etap | Status | Data |
|------|--------|------|
| Etap 1 — MVP FastAPI + Docker | ✅ Zakończony | 2026-06-13 |
| Etap 2 — Logowanie + Monitoring | ✅ Zakończony | 2026-06-13 |
| Etap 3 — Pełne MLOps | ✅ Zakończony | 2026-06-13 |

---

## ETAP 1 — Produkcyjne MVP

### Cel etapu
Uruchomienie notebooka, test parytetu modelu, FastAPI z 4 endpointami, strona HTML, Docker.

### Co zostało zrobione

#### Pliki aplikacji
- `app/features.py` — funkcje skopiowane z notebooka: `http_get_json`, `fetch_pse_day`, `fetch_pse_prices`, `fetch_weather`, `add_calendar_features`, `add_lag`, `build_tomorrow_features`
- `app/model_loader.py` — ładowanie modelu, obsługa braku pliku, hot-reload
- `app/schemas.py` — Pydantic 1:1 z 8 cechami + schema walidacji
- `app/db.py` — dostęp do timeseries.db
- `app/main.py` — FastAPI: GET /, GET /health, POST /predict, GET /predict/next-day

#### Frontend
- `frontend/index.html` — premium dark-mode UI z Chart.js:
  - Przycisk "Prognoza na jutro" → /predict/next-day → tabela + wykres
  - Sekcja "Tryb zaawansowany" (zwijana) → formularz ręczny → /predict
  - Stats row: średnia, min, max, godzina peak
  - Kolorowanie cen (zielony=niskie, żółty=peak, czerwony=wysokie)

#### Infrastruktura
- `requirements.txt` — przypięte wersje (sklearn 1.8.0)
- `Dockerfile` — python:3.11-slim, warstwy requirements→kod
- `docker-compose.yml` — port 8000, wolumeny models/data, healthcheck
- `.dockerignore` — wyklucza notebook, *.db, outputs

#### Skrypty
- `scripts/test_model_parity.py` — test parytetu (≥20 wierszy, rtol=1e-6)
- `scripts/smoke_test.sh` — automatyczny test bramki Etapu 1

### Jakie artefakty notebook powinien wygenerować
- `models/model.joblib` — kompletny pipeline sklearn
- `models/model_metadata.json` — 8 cech, sklearn_version, metryki
- `scripts/sample_payload.json` — realny rekord do testów
- `data/timeseries.db` — tabele prices, weather, energy_weather
- `data/reference_data.csv` — dane referencyjne pod dryft
- `outputs/prediction_next_day.csv` — wzorcowa prognoza

### Jak uruchomić aplikację

```bash
# 1. Wygeneruj model (jeśli nie masz artefaktów)
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1800 \
  --inplace notebook/model_cen_energii_RDN_Polska_POPRAWIONY.ipynb

# 2. Test parytetu (obowiązkowy!)
python scripts/test_model_parity.py

# 3. Uruchom w Dockerze
docker compose up -d --build

# 4. Smoke test
bash scripts/smoke_test.sh
```

### Jak przetestować /health i /predict

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @scripts/sample_payload.json
curl http://127.0.0.1:8000/predict/next-day
```

### Jak otworzyć stronę HTML
Przejdź do: http://127.0.0.1:8000

### Czego nie ruszać w Etapie 2
- `app/features.py` — logika 1:1 z notebooka (parytet)
- `models/model.joblib` — artefakt z notebooka
- Kontrakt odpowiedzi /predict (pola prediction, model_file, status)
- Wersja scikit-learn w requirements.txt

### Znane ograniczenia
- Prognoza może mieć mniej niż 24h rano, przed publikacją świeżych danych PSE
- Czas odpowiedzi /predict/next-day zależy od szybkości API PSE i Open-Meteo (~5-30s)
- Model trenowany od 2024-06-14 — brak danych starszych

---

## ETAP 2 — Logowanie i Monitoring

### Cel etapu
Zapisywanie predykcji do bazy SQLite, endpointy dla monitoringu operacyjnego (metryki Prometheus, logi) i monitoringu jakości modelu (dryft, ewaluacja).

### Co zostało zrobione

#### Aplikacja
- **Baza logów**: Dodano tworzenie `data/prediction_logs.db` z 3 tabelami w `app/db.py` (`prediction_logs`, `nextday_forecasts`, `forecast_accuracy`).
- **Logowanie**: Zmodyfikowano endpointy `/predict` i `/predict/next-day` w `app/main.py` do ciągłego logowania predykcji oraz błędów.
- **Monitoring endpoints**:
  - `GET /logs/recent` - zwraca 20 ostatnich logów predykcji.
  - `GET /monitoring/summary` - zwraca statystyki operacyjne.
  - `GET /metrics` - endpoint Prometheus z użyciem `prometheus_client`.
  - `GET /monitoring/drift` - zwraca najnowszy raport dryftu danych z pliku.
  - `GET /monitoring/accuracy` - zwraca historię ewaluacji dokładności z bazy.

#### Skrypty MLOps
- `src/monitoring/evaluate_forecasts.py`: Oblicza MAE/RMSE i porównuje z baseline dla zebranych predykcji.
- `src/monitoring/drift_report.py`: Porównuje średnie wartości 8 cech z ostatniego miesiąca do `reference_data.csv`.
- `scripts/generate_test_traffic.py`: Symuluje zapytania do endpointów API w celu testów monitoringu.
- `scripts/monitoring_test.sh`: Automatyczny test weryfikacyjny (smoke test) dla bramki Etapu 2.

### Jak przetestować Etap 2
```bash
# Wygenerowanie ruchu testowego
python scripts/generate_test_traffic.py 20

# Testy automatyczne (testują również endpointy monitorujące)
bash scripts/monitoring_test.sh
```

---

## ETAP 3 — Pełne MLOps (Retrening i Automatyzacja)

### Cel etapu
Pełna automatyzacja procesu: cykliczny retrening modelu z użyciem MLflow do logowania eksperymentów, harmonogramowanie zadań oraz gotowe narzędzia administracyjne.

### Co zostało zrobione

#### MLflow i Retrening
- **MLflow**: Dodano kontener MLflow server do `docker-compose.yml` przechowujący dane w `data/mlflow.db` (dostępny na porcie 5000).
- **Skrypt retreningu**: Utworzono `src/training/retrain.py`, który:
  1. Buduje najświeższy zestaw treningowy z bazy `timeseries.db` łącząc historię i nowo pobrane ceny z pogodą.
  2. Trenuje model HistGradientBoostingRegressor.
  3. Loguje metryki, parametry i sam model do bazy MLflow.
  4. Przy spełnieniu warunków jakościowych nadpisuje produkcyjny artefakt `models/model.joblib` i plik metadanych.

#### Orkiestracja zadań
- **Scheduler**: Dodano usługę `scheduler` do `docker-compose.yml` (skrypt `src/scheduler/scheduler.py` bazujący na bibliotece `schedule`), która uruchamia:
  - Aktualizację danych codziennie rano (09:15).
  - Ewaluację trafności i raport dryftu codziennie (10:00).
  - Retrening modelu raz w tygodniu (niedziela 02:00).
- Po ukończeniu retreningu Scheduler wysyła automatyczny request POST do kontenera `api` w celu przeładowania modelu bez restartowania aplikacji (Hot-reload).

#### Administracja i Obserwowalność
- **API Admin**: Dodano endpointy do ręcznego zarządzania modelem:
  - `POST /admin/reload-model` – przeładowanie modelu z dysku.
  - `POST /admin/retrain` – asynchroniczne odpalenie retreningu z poziomu API.
- **Grafana & Prometheus**: Podpięto w pełni skonfigurowane usługi do środowiska Docker Compose umożliwiając wizualizację danych wystawianych przez FastAPI na `/metrics`.

### Jak testować Etap 3
1. **Lokalny test retreningu**: Uruchom `python3 src/training/retrain.py` i sprawdź wyniki w terminalu. Zostanie stworzony plik `data/mlflow.db`.
2. **Dashboard MLflow**: Wejdź pod adres `http://127.0.0.1:5000` aby przejrzeć zarejestrowane eksperymenty (po uruchomieniu całego środowiska Docker Compose).
3. **Grafana**: Dostępna pod `http://127.0.0.1:3000` (wykorzystuje metryki zebrane przez `http://127.0.0.1:9090`).
