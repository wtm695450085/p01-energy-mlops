"""
Skrypt aktualizacji danych — Etap 2.
Idempotentny: dociąga wczorajsze/dzisiejsze ceny i pogodę do timeseries.db.
Uruchomienie: python scripts/update_data.py
"""
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.features import (
    DATA_DIR_PATH_FALLBACK,
    WEATHER_VARS,
    OM_HISTORY,
    fetch_pse_prices,
    fetch_weather,
    add_calendar_features,
    add_lag,
    FEATURES,
)

DB_PATH = ROOT / "data" / "timeseries.db"
if not DB_PATH.exists():
    DB_PATH = Path("/app/data/timeseries.db")

yesterday = date.today() - timedelta(days=1)
today     = date.today()

print(f"Aktualizacja danych do {yesterday}...")

# Dociągnij ceny
prices = fetch_pse_prices(yesterday, yesterday, db_path=DB_PATH)
print(f"Ceny: {len(prices)} rekordów dla {yesterday}")

# Dociągnij pogodę (ostatnie 3 dni)
start_w = yesterday - timedelta(days=2)
weather = fetch_weather(OM_HISTORY, start=start_w, end=yesterday)

with sqlite3.connect(DB_PATH) as con:
    # Dociągnij weather do bazy
    existing_w = None
    try:
        import pandas as pd
        existing_w = pd.read_sql("SELECT * FROM weather", con, parse_dates=["ts"])
        import pandas as pd
        new_rows = weather[~weather["ts"].isin(existing_w["ts"])]
        if len(new_rows) > 0:
            new_rows.to_sql("weather", con, if_exists="append", index=False)
            print(f"Dodano {len(new_rows)} godzin pogody")
        else:
            print("Pogoda: brak nowych rekordów")
    except Exception as e:
        weather.to_sql("weather", con, if_exists="replace", index=False)
        print(f"Pogoda zapisana od nowa: {e}")

print("Aktualizacja zakończona.")
