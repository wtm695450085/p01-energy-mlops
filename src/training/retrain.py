"""
Skrypt retreningu modelu.
Pobiera dane z bazy, uczy model, loguje eksperyment do MLflow i aktualizuje model produkcyjny.
"""
import sqlite3
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.db import get_db_path
from app.features import FEATURES, add_calendar_features, add_lag

# Ścieżki
DB_PATH = get_db_path()
MLFLOW_DB = ROOT / "data" / "mlflow.db"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

def get_training_data() -> pd.DataFrame:
    """Przygotowuje pełny zbiór danych treningowych na podstawie timeseries.db."""
    with sqlite3.connect(DB_PATH) as con:
        # Pobieranie cen (bez powtórzeń, uporządkowane)
        prices = pd.read_sql("SELECT ts, price_pln FROM prices", con, parse_dates=["ts"])
        prices = prices.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        
        # Pobieranie pogody (bez powtórzeń, uporządkowana)
        weather = pd.read_sql("SELECT ts, temp_pl, wind_pl, radiation_pl FROM weather", con, parse_dates=["ts"])
        weather = weather.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        
    if prices.empty or weather.empty:
        raise ValueError("Brak danych w bazie (prices lub weather jest puste).")

    # Merge po godzinach
    df = pd.merge(prices, weather, on="ts", how="inner")
    
    # Dodanie cech kalendarzowych
    df = add_calendar_features(df)
    
    # Dodanie opóźnień (lagów) z bazy cen
    df = add_lag(df, 24)
    df = add_lag(df, 168)
    
    # Usunięcie wierszy z brakami (szczególnie początek z powodu laga 168h)
    df = df.dropna().reset_index(drop=True)
    
    return df


def train_and_evaluate(df: pd.DataFrame):
    """Trenuje model i ocenia na zbiorze testowym, używając MLflow."""
    
    # Ostatnie 60 dni traktujemy jako test (zbliżone do założeń)
    test_days = 60
    cutoff = df["ts"].max() - pd.Timedelta(days=test_days)
    
    train_mask = df["ts"] < cutoff
    test_mask = df["ts"] >= cutoff
    
    df_train = df[train_mask]
    df_test = df[test_mask]
    
    if len(df_train) < 1000 or len(df_test) < 100:
        raise ValueError(f"Zbyt mało danych do treningu (train={len(df_train)}, test={len(df_test)}).")
        
    X_train = df_train[FEATURES]
    y_train = df_train["price_pln"].values
    
    X_test = df_test[FEATURES]
    y_test = df_test["price_pln"].values
    
    print(f"Rozpoczęcie treningu (train: {len(X_train)} wierszy, test: {len(X_test)} wierszy)...")
    
    # Model
    model = Pipeline([
        ('regressor', HistGradientBoostingRegressor(
            max_iter=200, 
            learning_rate=0.05, 
            max_leaf_nodes=31,
            random_state=42
        ))
    ])
    
    model.fit(X_train, y_train)
    
    # Ocena na zbiorze testowym
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Baseline (naiwny - opóźnienie o 168h)
    baseline_pred = df_test["price_lag_168h"].values
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    
    metrics = {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4)
    }
    
    print(f"Wyniki testu: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")
    print(f"Baseline MAE: {baseline_mae:.2f}")
    
    return model, metrics, baseline_mae, (str(df_train["ts"].min()), str(df_train["ts"].max())), (str(df_test["ts"].min()), str(df_test["ts"].max()))


def main():
    import sklearn
    MLFLOW_DB.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment("pl_day_ahead_power_price")
    
    with mlflow.start_run() as run:
        try:
            df = get_training_data()
            model, metrics, baseline_mae, train_range, test_range = train_and_evaluate(df)
            
            # Rejestracja parametrów i wyników w MLflow
            mlflow.log_param("champion", "HistGB")
            mlflow.log_param("sklearn_version", sklearn.__version__)
            mlflow.log_param("train_start", train_range[0])
            mlflow.log_param("train_end", train_range[1])
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            
            # Weryfikacja jakości przed promocją
            if metrics["R2"] < 0.50:
                print(f"Model zbyt słaby (R2={metrics['R2']}). Przerywam proces aktualizacji.")
                mlflow.log_param("status", "rejected")
                sys.exit(1)
                
            # Promocja modelu (nadpisanie produkcyjnych artefaktów)
            print("Promocja nowego modelu do produkcji...")
            joblib.dump(model, MODEL_PATH)
            
            meta = {
                "model_name": "pl_day_ahead_power_price",
                "champion": "HistGB",
                "target": "price_pln",
                "target_unit": "PLN/MWh",
                "features": FEATURES,
                "train_range": train_range,
                "test_range": test_range,
                "metrics_test": metrics,
                "baseline_naive_mae": round(baseline_mae, 2),
                "sklearn_version": sklearn.__version__,
                "exported_at": datetime.now().isoformat(),
                "mlflow_run_id": run.info.run_id
            }
            METADATA_PATH.write_text(json.dumps(meta, indent=2))
            mlflow.log_param("status", "promoted")
            
            print("Zakończono retrening sukcesem.")
            
        except Exception as e:
            mlflow.log_param("status", "failed")
            mlflow.log_param("error_message", str(e)[:250])
            print(f"Błąd podczas retreningu: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
