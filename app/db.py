"""Dostęp do baz SQLite timeseries.db oraz prediction_logs.db."""
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional


def get_db_path() -> Path:
    candidates = [
        Path("/app/data/timeseries.db"),
        Path("data/timeseries.db"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[1]

def get_logs_db_path() -> Path:
    candidates = [
        Path("/app/data/prediction_logs.db"),
        Path("data/prediction_logs.db"),
    ]
    # Używamy ścieżki Docker jeśli to możliwe, w przeciwnym razie lokalnej
    for p in candidates:
        if p.parent.exists():
            return p
    return candidates[1]


def init_logs_db() -> None:
    """Inicjalizacja bazy logów (tworzenie tabel, jeśli nie istnieją)."""
    p = get_logs_db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(p) as con:
        con.execute('''
            CREATE TABLE IF NOT EXISTS prediction_logs (
                request_id TEXT PRIMARY KEY,
                timestamp TEXT,
                features_json TEXT,
                prediction REAL,
                model_file TEXT,
                status TEXT,
                error_message TEXT
            )
        ''')
        con.execute('''
            CREATE TABLE IF NOT EXISTS nextday_forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_prognozy TEXT,
                godzina INTEGER,
                predicted_price_pln REAL,
                model_file TEXT,
                created_at TEXT
            )
        ''')
        # Tabela na wyniki trafności z evaluate_forecasts.py
        con.execute('''
            CREATE TABLE IF NOT EXISTS forecast_accuracy (
                data_prognozy TEXT PRIMARY KEY,
                mae_model REAL,
                rmse_model REAL,
                mae_baseline REAL,
                evaluated_at TEXT
            )
        ''')


def get_last_price_date() -> str | None:
    """Zwraca datę ostatniego rekordu w tabeli prices."""
    p = get_db_path()
    if not p.exists():
        return None
    try:
        with sqlite3.connect(p) as con:
            row = con.execute("SELECT MAX(ts) FROM prices").fetchone()
            return row[0] if row and row[0] else None
    except Exception:
        return None

def log_prediction(request_id: str, features: dict, prediction: Optional[float], 
                   model_file: str, status: str, error_message: str = "") -> None:
    """Zapisuje pojedynczą predykcję do logów."""
    try:
        p = get_logs_db_path()
        with sqlite3.connect(p) as con:
            con.execute(
                "INSERT INTO prediction_logs (request_id, timestamp, features_json, prediction, model_file, status, error_message) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (request_id, datetime.now().isoformat(), json.dumps(features), prediction, model_file, status, error_message)
            )
    except Exception as e:
        print(f"Error logging prediction: {e}")

def log_nextday_forecast(data_prognozy: str, godzina: int, predicted_price: float, model_file: str) -> None:
    """Zapisuje pojedynczą prognozę 24h na jutro."""
    try:
        p = get_logs_db_path()
        with sqlite3.connect(p) as con:
            con.execute(
                "INSERT INTO nextday_forecasts (data_prognozy, godzina, predicted_price_pln, model_file, created_at) VALUES (?, ?, ?, ?, ?)",
                (data_prognozy, godzina, predicted_price, model_file, datetime.now().isoformat())
            )
    except Exception as e:
        print(f"Error logging nextday forecast: {e}")

# Inicjalizuj bazę logów przy imporcie modułu
init_logs_db()
