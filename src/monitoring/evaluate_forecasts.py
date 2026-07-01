"""Weryfikacja trafności: porównanie prognoz modelu z realnymi cenami."""
import sqlite3
import json
import sys
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.db import get_logs_db_path
from app.features import fetch_pse_prices, _get_db_path

def evaluate_forecasts():
    logs_db = get_logs_db_path()
    ts_db = _get_db_path()
    
    if not logs_db.exists():
        print("Baza logów nie istnieje.")
        return

    # Pobierz zrealizowane prognozy z bazy logów
    with sqlite3.connect(logs_db) as con:
        forecasts = pd.read_sql("SELECT data_prognozy, godzina, predicted_price_pln, model_file FROM nextday_forecasts", con)
        
    if forecasts.empty:
        print("Brak prognoz do oceny.")
        return

    # Skonwertuj typy
    forecasts["data_prognozy"] = pd.to_datetime(forecasts["data_prognozy"]).dt.date
    min_date = forecasts["data_prognozy"].min()
    max_date = forecasts["data_prognozy"].max()

    # Dociągnij ceny za dany okres
    prices = fetch_pse_prices(min_date - timedelta(days=7), max_date, db_path=ts_db)
    prices["data_prognozy"] = prices["ts"].dt.date
    prices["godzina"] = prices["ts"].dt.hour

    # Dodaj baseline (naiwny 168h wstecz)
    prices_with_lag = prices.copy()
    prices_with_lag["baseline_price"] = prices_with_lag["price_pln"].shift(168)

    # Złącz prognozy z realnymi cenami
    merged = pd.merge(forecasts, prices_with_lag, on=["data_prognozy", "godzina"], how="inner")
    
    if merged.empty:
        print("Brak rzeczywistych cen odpowiadających zapisanym prognozom (zbyt świeże?).")
        return

    # Zapisz wyniki dni po dniu
    accuracy_results = []
    
    for day, group in merged.groupby("data_prognozy"):
        if len(group) < 24:
            print(f"Dzień {day} ma niepełne dane: {len(group)} godz.")
            # Jeśli brak pełnej doby dla testów ok, ale można zignorować
        
        group_clean = group.dropna(subset=["price_pln", "predicted_price_pln", "baseline_price"])
        if group_clean.empty:
            continue
            
        mae_model = np.mean(np.abs(group_clean["predicted_price_pln"] - group_clean["price_pln"]))
        rmse_model = np.sqrt(np.mean((group_clean["predicted_price_pln"] - group_clean["price_pln"])**2))
        mae_baseline = np.mean(np.abs(group_clean["baseline_price"] - group_clean["price_pln"]))
        
        acc = {
            "data_prognozy": str(day),
            "mae_model": round(mae_model, 2),
            "rmse_model": round(rmse_model, 2),
            "mae_baseline": round(mae_baseline, 2),
            "evaluated_at": pd.Timestamp.now().isoformat()
        }
        accuracy_results.append(acc)

    if not accuracy_results:
        print("Nie można policzyć trafności z powodu braków w danych.")
        return

    # Zapisz do bazy logów (nadpisz dla danego dnia jeśli już istnieje)
    with sqlite3.connect(logs_db) as con:
        for acc in accuracy_results:
            con.execute('''
                INSERT OR REPLACE INTO forecast_accuracy 
                (data_prognozy, mae_model, rmse_model, mae_baseline, evaluated_at) 
                VALUES (?, ?, ?, ?, ?)
            ''', (acc["data_prognozy"], acc["mae_model"], acc["rmse_model"], acc["mae_baseline"], acc["evaluated_at"]))

    # Zapisz raport
    report_path = ROOT / "outputs" / "accuracy_report.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(accuracy_results, indent=2), encoding="utf-8")
    
    print(f"Wyliczono trafność dla {len(accuracy_results)} dni. Zapisano raport do outputs/accuracy_report.json")

if __name__ == "__main__":
    evaluate_forecasts()
