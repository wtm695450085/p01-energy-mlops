"""Raport dryftu danych na podstawie data/reference_data.csv"""
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.db import get_db_path
from app.features import FEATURES

def generate_drift_report():
    ref_path = ROOT / "data" / "reference_data.csv"
    ts_db = get_db_path()
    output_path = ROOT / "outputs" / "drift_report.json"
    
    if not ref_path.exists() or not ts_db.exists():
        print("Brak danych referencyjnych lub bieżącej bazy. Zaniechano raportu.")
        return

    # Pobierz dane referencyjne
    ref_df = pd.read_csv(ref_path)
    
    # Pobierz bieżące dane (ostatnie 30 dni)
    import sqlite3
    with sqlite3.connect(ts_db) as con:
        # Zakładamy, że data to ostatnie 30 dni od najnowszego rekordu
        curr_df = pd.read_sql("SELECT * FROM energy_weather ORDER BY ts DESC LIMIT 720", con) # 720 h = 30 dni
        
    if len(curr_df) < 100:
        print("Zbyt mało danych bieżących do oceny dryftu.")
        report = {"status": "not_enough_data"}
        output_path.parent.mkdir(exist_ok=True)
        output_path.write_text(json.dumps(report), encoding="utf-8")
        return

    report = {
        "status": "ok",
        "features": {},
        "drift_detected": False
    }

    # Proste statystyki i wykrywanie dryftu (ponad 20% zmiany średniej lub odchylenia standardowego dla ciągłych)
    drift_flag = False
    
    for f in FEATURES:
        if f not in ref_df.columns or f not in curr_df.columns:
            continue
            
        ref_mean = ref_df[f].mean()
        ref_std = ref_df[f].std()
        
        curr_mean = curr_df[f].mean()
        curr_std = curr_df[f].std()
        
        # Unikanie dzielenia przez zero
        if ref_mean == 0: ref_mean = 1e-6
        if ref_std == 0: ref_std = 1e-6
        
        mean_diff = abs(curr_mean - ref_mean) / abs(ref_mean)
        std_diff = abs(curr_std - ref_std) / abs(ref_std)
        
        is_drifted = mean_diff > 0.20 or std_diff > 0.20
        if is_drifted:
            drift_flag = True

        report["features"][f] = {
            "reference_mean": round(float(ref_df[f].mean()), 2),
            "current_mean": round(float(curr_mean), 2),
            "reference_std": round(float(ref_df[f].std()), 2),
            "current_std": round(float(curr_std), 2),
            "drift_detected": bool(is_drifted)
        }

    report["drift_detected"] = drift_flag
    if drift_flag:
        report["status"] = "drift_detected"

    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Zapisano raport dryftu do {output_path}. Dryft wykryty: {drift_flag}")

if __name__ == "__main__":
    generate_drift_report()
