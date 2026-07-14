"""
FastAPI — Prognozowanie cen energii RDN (Polska)

Endpointy:
  GET  /                   — strona HTML
  GET  /health             — status aplikacji
  POST /predict            - prediction from 8 features (manual mode)
  GET  /predict/next-day   — prognoza 24h na jutro (automatyczne cechy)
"""
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.db import (
    get_last_price_date,
    log_prediction,
    log_nextday_forecast,
    get_logs_db_path
)
import sqlite3
import time
from app.features import (
    FEATURES,
    build_tomorrow_features,
    fetch_weather,
)
from app.monitoring import (
    get_metrics_response,
    mlops_predictions_total,
    mlops_prediction_errors_total,
    mlops_prediction_latency_seconds,
    mlops_last_prediction_value
)
from app.model_loader import load_model, model_state, reload_model
from app.schemas import (
    EnergyFeatures,
    NextDayForecastItem,
    NextDayForecastResponse,
    PredictionResponse,
)

# --- Inicjalizacja aplikacji ------------------------------------------------

app = FastAPI(
    title="RDN Energy Price Forecast",
    description="MLOps API - hourly energy price forecasts for Poland's Day-Ahead Market",
    version="1.0.0",
)

# Frontend (statyczne pliki)
_FRONTEND_DIR = Path("/app/frontend") if Path("/app/frontend").exists() else Path("frontend")
if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


@app.on_event("startup")
async def startup_event():
    load_model()


# --- Endpointy --------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """Serves the main HTML page."""
    html_path = _FRONTEND_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(
        str(html_path),
        media_type="text/html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/health")
async def health():
    """Application and model status."""
    return {
        "status": "ok",
        "model_loaded": model_state.loaded,
        "model_file": model_state.model_file,
        "model_error": model_state.error,
        "sklearn_version": model_state.metadata.get("sklearn_version"),
        "model_champion": model_state.metadata.get("champion"),
        "last_price_date": get_last_price_date(),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: EnergyFeatures):
    """Prediction from 8 input features - manual/test mode."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    mlops_predictions_total.inc()
    
    if not model_state.loaded:
        mlops_prediction_errors_total.inc()
        log_prediction(request_id, features.model_dump(), None, model_state.model_file or "", "error", "Model not loaded")
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {model_state.error}",
        )

    try:
        X = pd.DataFrame([features.model_dump()])
        X = X[FEATURES]
        pred = float(model_state.model.predict(X)[0])
        
        # Sukces
        latency = time.time() - start_time
        mlops_prediction_latency_seconds.observe(latency)
        mlops_last_prediction_value.set(pred)
        
        log_prediction(request_id, features.model_dump(), round(pred, 2), model_state.model_file or "", "ok")
        
    except Exception as e:
        mlops_prediction_errors_total.inc()
        log_prediction(request_id, features.model_dump(), None, model_state.model_file or "", "error", str(e))
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return PredictionResponse(
        prediction=round(pred, 2),
        model_file=model_state.model_file or "",
        status="ok",
        request_id=request_id,
    )


@app.get("/predict/next-day", response_model=NextDayForecastResponse)
async def predict_next_day():
    """MAIN function: automatic 24h forecast for tomorrow.

    Fetches the weather forecast from Open-Meteo, builds calendar features and price lags
    from timeseries.db, then returns 24 hourly RDN price forecasts.
    """
    if not model_state.loaded:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {model_state.error}",
        )

    try:
        feat_df = build_tomorrow_features()
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Could not build tomorrow features: {e}",
        )

    if feat_df.empty:
        raise HTTPException(
            status_code=422,
            detail="No data available for tomorrow forecast (empty feature table)",
        )

    try:
        X = feat_df[FEATURES]
        preds = model_state.model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    target_date = feat_df["ts"].dt.date.iloc[0].isoformat()
    forecasts = []

    for i, (_, row) in enumerate(feat_df.iterrows()):
        pred_val = float(preds[i])
        item = NextDayForecastItem(
            hour=int(row["hour"]),
            datetime=row["ts"].isoformat(),
            predicted_price_pln=round(pred_val, 2),
            temp_pl=round(float(row["temp_pl"]), 1),
            wind_pl=round(float(row["wind_pl"]), 1),
            radiation_pl=round(float(row["radiation_pl"]), 1),
            is_holiday_or_weekend=int(row["is_holiday_or_weekend"]),
        )
        forecasts.append(item)
        # Zapis do bazy
        log_nextday_forecast(target_date, int(row["hour"]), round(pred_val, 2), model_state.model_file or "")

    return NextDayForecastResponse(
        date=target_date,
        forecasts=forecasts,
        model_file=model_state.model_file or "",
        status="ok",
        generated_at=datetime.now().isoformat(),
    )


@app.get("/logs/recent")
async def logs_recent():
    """Returns the latest 20 prediction requests."""
    p = get_logs_db_path()
    if not p.exists():
        return {"logs": []}
    try:
        with sqlite3.connect(p) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT * FROM prediction_logs ORDER BY timestamp DESC LIMIT 20"
            ).fetchall()
            return {"logs": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring/summary")
async def monitoring_summary():
    """Returns the prediction summary from the log database."""
    p = get_logs_db_path()
    if not p.exists():
        return {
            "total_predictions": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "average_prediction": None,
            "min_prediction": None,
            "max_prediction": None,
            "last_prediction_timestamp": None
        }
    try:
        with sqlite3.connect(p) as con:
            con.row_factory = sqlite3.Row
            total = con.execute("SELECT COUNT(*) as c FROM prediction_logs").fetchone()["c"]
            succ = con.execute("SELECT COUNT(*) as c FROM prediction_logs WHERE status='ok'").fetchone()["c"]
            fail = con.execute("SELECT COUNT(*) as c FROM prediction_logs WHERE status!='ok'").fetchone()["c"]
            stats = con.execute("SELECT AVG(prediction) as avg, MIN(prediction) as min, MAX(prediction) as max, MAX(timestamp) as last_ts FROM prediction_logs WHERE status='ok'").fetchone()
            
            return {
                "total_predictions": total,
                "successful_predictions": succ,
                "failed_predictions": fail,
                "average_prediction": round(stats["avg"], 2) if stats["avg"] is not None else None,
                "min_prediction": round(stats["min"], 2) if stats["min"] is not None else None,
                "max_prediction": round(stats["max"], 2) if stats["max"] is not None else None,
                "last_prediction_timestamp": stats["last_ts"]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Endpoint dla Prometheusa."""
    return get_metrics_response()


@app.get("/monitoring/accuracy")
async def monitoring_accuracy():
    """Returns the daily forecast error history (model vs baseline)."""
    p = get_logs_db_path()
    if not p.exists():
        return {"history": []}
    try:
        with sqlite3.connect(p) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute("SELECT * FROM forecast_accuracy ORDER BY data_prognozy DESC").fetchall()
            return {"history": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring/drift")
async def monitoring_drift():
    """Returns the latest drift report."""
    import json
    report_path = Path("/app/outputs/drift_report.json")
    if not report_path.exists():
        report_path = Path("outputs/drift_report.json")
        
    if not report_path.exists():
        return {"status": "not_available", "message": "The drift report has not been generated yet."}
        
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report read error: {e}")


@app.post("/admin/reload-model")
async def admin_reload_model():
    """Wymusza wczytanie modelu z pliku z dysku."""
    try:
        reload_model()
        if model_state.loaded:
            return {"status": "ok", "message": "Model reloaded", "file": model_state.model_file}
        else:
            raise HTTPException(status_code=500, detail=model_state.error)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/retrain")
async def admin_retrain():
    """Asynchronicznie odpala proces retreningu."""
    import subprocess
    from fastapi import BackgroundTasks
    
    def run_retrain():
        try:
            # Uruchamia skrypt w tle
            subprocess.run(["python3", "src/training/retrain.py"], check=True)
            # After successful training, the model should be refreshed
            reload_model()
        except Exception as e:
            print(f"Background task error (retraining): {e}")

    # Asynchronous run - for testing we can run this synchronously for a small model or via BackgroundTasks, but FastAPI BackgroundTasks is cleaner.
    # To do this without BackgroundTasks in the argument, import it inside:
    # For the API, it is best to run this in a non-blocking subprocess
    subprocess.Popen(["python3", "src/training/retrain.py"])
    return {"status": "ok", "message": "The retraining job has been started in the background."}

@app.get("/architecture.html", include_in_schema=False)
async def architecture_page():
    return FileResponse(
        "frontend/architecture.html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )
