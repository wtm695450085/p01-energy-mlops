# EnergyForecast PL — Środowisko produkcyjne MLOps

Prognozowanie godzinowych cen energii elektrycznej na **Rynku Dnia Następnego** w Polsce (PLN/MWh).

Model: sklearn pipeline (HistGradientBoostingRegressor) · 8 cech · Dane: PSE + Open-Meteo

---

## Szybki start (lokalnie)

### Wymagania
- Python 3.11+
- Docker + Docker Compose
- Dostęp do internetu (PSE API + Open-Meteo)

### Krok 1 — Wygenerowanie modelu z notebooka

> Pierwszy raz pobiera ~2 lata danych (kilka minut). Wymaga internetu.

```bash
cd energy-mlops
pip install jupyter nbconvert scikit-learn pandas numpy requests holidays joblib matplotlib
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1800 \
  --inplace notebook/model_cen_energii_RDN_Polska_POPRAWIONY.ipynb
```

Sprawdź artefakty:
```bash
ls models/       # model.joblib, model_metadata.json
ls data/         # timeseries.db, reference_data.csv
ls scripts/      # sample_payload.json
ls outputs/      # prediction_next_day.csv
```

### Krok 2 — Test parytetu (obowiązkowy)

```bash
python scripts/test_model_parity.py
# Oczekiwane: ✅ PARYTET PRZESZEDŁ
```

### Krok 3 — Uruchomienie w Dockerze

```bash
docker compose up -d --build
```

Aplikacja dostępna pod: **http://127.0.0.1:8000**

---

## Weryfikacja działania

```bash
# Status aplikacji
curl http://127.0.0.1:8000/health

# Predykcja ręczna
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @scripts/sample_payload.json

# Prognoza na jutro (24h automatyczne)
curl http://127.0.0.1:8000/predict/next-day

# Smoke test (wszystkie powyższe automatycznie)
bash scripts/smoke_test.sh
```

---

## Wdrożenie na VPS

```bash
# Skopiuj katalog projektu (wraz z data/ i models/) na VPS
rsync -avz energy-mlops/ user@vps:/opt/energy-forecast/

# Na VPS:
cd /opt/energy-forecast
docker compose up -d --build
```

> ⚠️ Kontener wymaga dostępu do internetu: `api.raporty.pse.pl` i `open-meteo.com`

---

## Usługi i adresy

| Usługa | Adres | Opis |
|--------|-------|------|
| Frontend | http://127.0.0.1:8000/ | Strona HTML z prognozą |
| Health | http://127.0.0.1:8000/health | Status API i modelu |
| Predict | POST http://127.0.0.1:8000/predict | Predykcja ręczna |
| Next-day | GET http://127.0.0.1:8000/predict/next-day | Prognoza 24h na jutro |
| API docs | http://127.0.0.1:8000/docs | Swagger UI |

---

## Dane wejściowe modelu (8 cech)

| Cecha | Opis | Źródło |
|-------|------|--------|
| `hour` | Godzina doby (0–23) | kalendarz |
| `day_of_week` | Dzień tygodnia (0=pon) | kalendarz |
| `is_holiday_or_weekend` | 1 = weekend/święto | kalendarz + holidays PL |
| `temp_pl` | Temperatura krajowa [°C] | Open-Meteo, 10 miast |
| `wind_pl` | Prędkość wiatru [km/h] | Open-Meteo, 10 miast |
| `radiation_pl` | Nasłonecznienie [W/m²] | Open-Meteo, 10 miast |
| `price_lag_24h` | Cena RDN sprzed 24h [PLN/MWh] | PSE / timeseries.db |
| `price_lag_168h` | Cena RDN sprzed 168h [PLN/MWh] | PSE / timeseries.db |

---

## Struktura projektu

```
energy-mlops/
├── app/              # FastAPI: main.py, model_loader.py, schemas.py, features.py, db.py
├── data/             # timeseries.db, reference_data.csv (wolumen Docker)
├── frontend/         # index.html — interfejs użytkownika
├── models/           # model.joblib, model_metadata.json (wolumen Docker)
├── notebook/         # Źródłowy notebook Jupyter
├── outputs/          # Prognozy CSV
├── scripts/          # test_model_parity.py, smoke_test.sh, sample_payload.json
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Czego nie ruszać

- Logika funkcji w `app/features.py` — skopiowana 1:1 z notebooka (zasada parytetu)
- Wersja `scikit-learn` w `requirements.txt` — musi pasować do wersji z treningu
- Wolumeny `./models` i `./data` — zawierają model i bazę danych
