"""
Moduł cech — funkcje przeniesione z notebooka.

UWAGA: Logika jest IDENTYCZNA z notebookiem (zasada parytetu).
Szczególnie http_get_json — API PSE wymaga literalnego '$filter',
spacji jako %20 i literalnych apostrofów. NIE upraszczać do requests params.
"""
import json
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

try:
    import holidays as _hol
    _holidays_ok = True
except ImportError:
    _holidays_ok = False

# --- Konfiguracja -----------------------------------------------------------

CITIES = {
    "Warszawa":  (52.2297, 21.0122),
    "Krakow":    (50.0647, 19.9450),
    "Lodz":      (51.7592, 19.4560),
    "Wroclaw":   (51.1079, 17.0385),
    "Poznan":    (52.4064, 16.9252),
    "Gdansk":    (54.3520, 18.6466),
    "Szczecin":  (53.4285, 14.5528),
    "Lublin":    (51.2465, 22.5684),
    "Bialystok": (53.1325, 23.1688),
    "Katowice":  (50.2649, 19.0238),
}

WEATHER_VARS = ["temperature_2m", "wind_speed_10m", "shortwave_radiation"]

FEATURES = [
    "hour", "day_of_week", "is_holiday_or_weekend",
    "temp_pl", "wind_pl", "radiation_pl",
    "price_lag_24h", "price_lag_168h",
]
TARGET = "price_pln"

PSE_API_CSDAC = "https://api.raporty.pse.pl/api/csdac-pln"
PSE_API_RCE   = "https://api.raporty.pse.pl/api/rce-pln"
OM_HISTORY    = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OM_FORECAST   = "https://api.open-meteo.com/v1/forecast"


def _get_db_path() -> Path:
    """Zwraca ścieżkę do timeseries.db z obsługą środowiska (Docker / dev)."""
    # Docker: /app/data/timeseries.db   Dev: data/timeseries.db
    candidates = [
        Path("/app/data/timeseries.db"),
        Path("data/timeseries.db"),
    ]
    for p in candidates:
        if p.exists():
            return p
    # zwróć domyślną (zostanie utworzona)
    return candidates[1]


def _build_pl_holidays(years_ahead: int = 2) -> dict:
    if not _holidays_ok:
        return {}
    current_year = datetime.now().year
    try:
        return _hol.country_holidays("PL", years=range(2024, current_year + years_ahead + 1))
    except Exception:
        return {}


# Kalend. świąt PL — wczytany raz przy imporcie modułu
PL_HOLIDAYS: dict = _build_pl_holidays()


# --- Pomocnik HTTP ----------------------------------------------------------

def http_get_json(url: str, params: dict | None = None, tries: int = 3, timeout: int = 60):
    """GET z retry. URL budowany ręcznie: API PSE wymaga literalnego '$filter'
    (nie %24filter), spacji jako %20 i literalnych apostrofów."""
    if params:
        safe_chars = "',:-"
        qs = "&".join(f"{k}={quote(str(v), safe=safe_chars)}"
                      for k, v in params.items())
        url = f"{url}?{qs}"
    last_err = None
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 * attempt)
    raise RuntimeError(f"Nie udało się pobrać {url}: {last_err}")


# --- Ceny PSE ---------------------------------------------------------------

def _find_col(df: pd.DataFrame, substrings: list[str]) -> str | None:
    for sub in substrings:
        for c in df.columns:
            if sub in c.lower():
                return c
    return None


def _parse_pse_rows(rows: list) -> pd.DataFrame:
    raw = pd.DataFrame(rows)
    price_col = _find_col(raw, ["csdac_pln", "rce_pln", "pln"])
    time_col  = _find_col(raw, ["dtime", "udtczas", "period"])
    if price_col is None or time_col is None:
        raise RuntimeError(f"Nieznany format API PSE. Kolumny: {list(raw.columns)}")
    df = raw[[time_col, price_col]].copy()
    df.columns = ["ts_raw", "price_pln"]
    df["price_pln"] = pd.to_numeric(df["price_pln"], errors="coerce")
    df["ts"] = pd.to_datetime(df["ts_raw"], errors="coerce").dt.floor("h")
    return (df.dropna(subset=["ts", "price_pln"])
              .groupby("ts", as_index=False)["price_pln"].mean()
              .sort_values("ts"))


def fetch_pse_day(d: date) -> pd.DataFrame:
    """Godzinowe ceny RDN dla jednego dnia (csdac-pln → rce-pln jako zapas)."""
    for name, api in (("csdac-pln", PSE_API_CSDAC), ("rce-pln", PSE_API_RCE)):
        js = http_get_json(api, params={"$filter": f"business_date eq '{d}'"})
        rows = js.get("value", [])
        if rows:
            return _parse_pse_rows(rows)
    df = pd.DataFrame(columns=["ts", "price_pln"])
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def load_cached_prices(db_path: Path | None = None) -> pd.DataFrame:
    p = db_path or _get_db_path()
    if not p.exists():
        return pd.DataFrame(columns=["ts", "price_pln"])
    try:
        with sqlite3.connect(p) as con:
            return pd.read_sql("SELECT ts, price_pln FROM prices", con,
                               parse_dates=["ts"])
    except Exception:
        df = pd.DataFrame(columns=["ts", "price_pln"])
        df["ts"] = pd.to_datetime(df["ts"])
        return df


def fetch_pse_prices(start: date, end: date, db_path: Path | None = None) -> pd.DataFrame:
    """Ceny RDN [start, end]: cache z SQLite + dociągnięcie brakujących dni."""
    p = db_path or _get_db_path()
    cached = load_cached_prices(p)
    have_days = set(cached["ts"].dt.date) if len(cached) else set()
    all_days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    missing = [d for d in all_days if d not in have_days]

    new_frames = []
    for i, d in enumerate(missing, 1):
        df_day = fetch_pse_day(d)
        if not df_day.empty:
            new_frames.append(df_day)
        time.sleep(0.1)

    frames = []
    if len(cached) > 0:
        frames.append(cached)
    frames.extend(new_frames)
    
    if not frames:
        full = pd.DataFrame(columns=["ts", "price_pln"])
        full["ts"] = pd.to_datetime(full["ts"])
    else:
        full = (pd.concat(frames, ignore_index=True)
                  .drop_duplicates("ts").sort_values("ts").reset_index(drop=True))

    # Zapisz do cache
    p.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(p) as con:
        full_to_db = full.copy()
        full_to_db["ts"] = full_to_db["ts"].astype(str)
        full_to_db.to_sql("prices", con, if_exists="replace", index=False)

    mask = (full["ts"].dt.date >= start) & (full["ts"].dt.date <= end)
    return full[mask].reset_index(drop=True)


# --- Pogoda -----------------------------------------------------------------

def _weather_json_to_df(js_one_city: dict) -> pd.DataFrame:
    h = js_one_city["hourly"]
    df = pd.DataFrame({"ts": pd.to_datetime(h["time"])})
    for v in WEATHER_VARS:
        df[v] = h[v]
    return df


def fetch_weather(url: str, start: date | None = None, end: date | None = None,
                  forecast_days: int | None = None) -> pd.DataFrame:
    """Pogoda dla wszystkich miast; zwraca średnią krajową."""
    params: dict = {
        "latitude":  ",".join(str(lat) for lat, _ in CITIES.values()),
        "longitude": ",".join(str(lon) for _, lon in CITIES.values()),
        "hourly": ",".join(WEATHER_VARS),
        "timezone": "Europe/Warsaw",
    }
    if start is not None:
        params["start_date"] = str(start)
        params["end_date"]   = str(end)
    if forecast_days is not None:
        params["forecast_days"] = forecast_days

    js = http_get_json(url, params=params)
    cities_js = js if isinstance(js, list) else [js]
    per_city = [_weather_json_to_df(c) for c in cities_js]

    allc = pd.concat(per_city, ignore_index=True)
    national = (allc.groupby("ts", as_index=False)[WEATHER_VARS].mean()
                    .rename(columns={"temperature_2m": "temp_pl",
                                     "wind_speed_10m": "wind_pl",
                                     "shortwave_radiation": "radiation_pl"}))
    return national.sort_values("ts").reset_index(drop=True)


# --- Kalendarz --------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"]        = out["ts"].dt.hour
    out["day_of_week"] = out["ts"].dt.dayofweek   # 0 = poniedziałek
    is_weekend = out["day_of_week"] >= 5
    is_holiday = out["ts"].dt.date.map(lambda d: d in PL_HOLIDAYS)
    out["is_holiday_or_weekend"] = (is_weekend | is_holiday).astype(int)
    return out


# --- Lagi -------------------------------------------------------------------

def add_lag(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    """Dodaje kolumnę price_lag_{hours}h przez przesunięcie szeregu cen."""
    lag = df[["ts", "price_pln"]].copy()
    lag["ts"] = lag["ts"] + pd.Timedelta(hours=hours)
    lag = lag.rename(columns={"price_pln": f"price_lag_{hours}h"})
    return df.merge(lag, on="ts", how="left")


# --- Budowanie cech na jutro ------------------------------------------------

def build_tomorrow_features(db_path: Path | None = None) -> pd.DataFrame:
    """Automatyczne budowanie wektora cech dla 24 godzin jutrzejszego dnia.

    Identyczna logika jak w notebooku (parytet produkcyjny).
    """
    p = db_path or _get_db_path()
    tomorrow = date.today() + timedelta(days=1)

    # 1) Pogoda na jutro
    wf = fetch_weather(OM_FORECAST, forecast_days=2)
    wf = wf[wf["ts"].dt.date == tomorrow].copy()
    if len(wf) == 0:
        raise RuntimeError("Brak prognozy pogody na jutro z Open-Meteo")

    # 2) Kalendarz
    feat = add_calendar_features(wf)

    # 3) Historia cen z bazy + dociągnięcie najświeższych dni
    with sqlite3.connect(p) as con:
        hist = pd.read_sql("SELECT ts, price_pln FROM prices", con,
                           parse_dates=["ts"])
    last_day = hist["ts"].dt.date.max()
    if last_day < date.today():
        try:
            fresh = fetch_pse_prices(last_day, date.today(), db_path=p)
            hist = (pd.concat([hist, fresh]).drop_duplicates("ts")
                      .sort_values("ts").reset_index(drop=True))
            with sqlite3.connect(p) as con:
                hist_to_db = hist.copy()
                hist_to_db["ts"] = hist_to_db["ts"].astype(str)
                hist_to_db.to_sql("prices", con, if_exists="replace", index=False)
        except Exception as e:
            print(f"Uwaga: nie udało się dociągnąć najświeższych cen: {e}")

    # 4) Lagi
    s = (hist.dropna(subset=["price_pln"])
             .drop_duplicates("ts").set_index("ts")["price_pln"]
             .sort_index())
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="h")
    s = s.reindex(full_idx).ffill().bfill()

    def lag_value(t, hours: int) -> float:
        key = t - pd.Timedelta(hours=hours)
        if key in s.index:
            return float(s.loc[key])
        pos = s.index.searchsorted(key)
        pos = min(max(pos, 0), len(s) - 1)
        return float(s.iloc[pos])

    feat["price_lag_24h"]  = feat["ts"].map(lambda t: lag_value(t, 24))
    feat["price_lag_168h"] = feat["ts"].map(lambda t: lag_value(t, 168))

    out = feat.dropna(subset=FEATURES).reset_index(drop=True)
    return out
