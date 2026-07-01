"""
Test parytetu modelu — ETAP 1, krok obowiązkowy.

Sprawdza, czy wczytany models/model.joblib daje identyczne predykcje
co dane referencyjne (tolerancja rtol=1e-6) oraz dla sample_payload.json.

Uruchomienie: python scripts/test_model_parity.py
Wymaganie: models/model.joblib i data/reference_data.csv muszą istnieć.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# --- Ścieżki ----------------------------------------------------------------
ROOT = Path(__file__).parent.parent
MODEL_PATH   = ROOT / "models" / "model.joblib"
REF_DATA     = ROOT / "data" / "reference_data.csv"
PAYLOAD_PATH = ROOT / "scripts" / "sample_payload.json"
METADATA_PATH= ROOT / "models" / "model_metadata.json"

FEATURES = [
    "hour", "day_of_week", "is_holiday_or_weekend",
    "temp_pl", "wind_pl", "radiation_pl",
    "price_lag_24h", "price_lag_168h",
]

errors = []

def check(condition: bool, msg_ok: str, msg_fail: str):
    if condition:
        print(f"  ✓  {msg_ok}")
    else:
        print(f"  ✗  {msg_fail}")
        errors.append(msg_fail)

# --- 1. Sprawdzenie pliku modelu --------------------------------------------
print("\n=== TEST PARYTETU MODELU ===\n")
print("1. Pliki artefaktów")

check(MODEL_PATH.exists(),   f"model.joblib znaleziony ({MODEL_PATH})", f"Brak {MODEL_PATH}")
check(REF_DATA.exists(),     f"reference_data.csv znaleziony",           f"Brak {REF_DATA}")
check(PAYLOAD_PATH.exists(), f"sample_payload.json znaleziony",          f"Brak {PAYLOAD_PATH}")

if errors:
    print("\n❌ Brak wymaganych plików — uruchom notebook i spróbuj ponownie.")
    sys.exit(1)

# --- 2. Wczytanie modelu ----------------------------------------------------
print("\n2. Wczytanie modelu")
try:
    model = joblib.load(MODEL_PATH)
    check(True, f"Model wczytany z {MODEL_PATH}", "")
except Exception as e:
    check(False, "", f"Błąd wczytania modelu: {e}")
    sys.exit(1)

if METADATA_PATH.exists():
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    print(f"       Champion: {meta.get('champion')}")
    print(f"       sklearn:  {meta.get('sklearn_version')}")
    print(f"       MAE test: {meta.get('metrics_test', {}).get('MAE')} PLN/MWh")

# --- 3. Predykcje na danych referencyjnych (≥20 wierszy) -------------------
print("\n3. Test na danych referencyjnych (≥ 20 wierszy)")
ref = pd.read_csv(REF_DATA)
check("price_pln" in ref.columns, "Kolumna price_pln obecna", "Brak kolumny price_pln")
check(len(ref) >= 20, f"Danych referencyjnych: {len(ref)} (wymagane ≥ 20)", "Za mało danych referencyjnych")

# Bierzemy 50 ostatnich wierszy
sample = ref.tail(50).copy()
X_sample = sample[FEATURES]
y_true   = sample["price_pln"].values

pred_fresh = model.predict(X_sample)

# Parytet: wczytaj model jeszcze raz i porównaj
model2 = joblib.load(MODEL_PATH)
pred_reload = model2.predict(X_sample)

parity_ok = np.allclose(pred_fresh, pred_reload, rtol=1e-6)
check(parity_ok, "Parytet na 50 wierszach: ✓ (rtol=1e-6)", "PARYTET NIEZGODNY!")

mae = np.mean(np.abs(pred_fresh - y_true))
print(f"       MAE na próbce referencyjnej: {mae:.2f} PLN/MWh")

# --- 4. Test na sample_payload.json -----------------------------------------
print("\n4. Test na sample_payload.json")
payload = json.loads(PAYLOAD_PATH.read_text(encoding="utf-8"))
X_pay = pd.DataFrame([payload])[FEATURES]

pred_payload = float(model.predict(X_pay)[0])
pred_payload2 = float(model2.predict(X_pay)[0])

diff = abs(pred_payload - pred_payload2)
check(diff < 1e-6,
      f"Predykcja sample_payload: {pred_payload:.4f} PLN/MWh (parytet OK, diff={diff:.2e})",
      f"Parytet sample_payload NIEZGODNY: diff={diff:.2e}")

check(0 <= pred_payload <= 10000,
      f"Predykcja w sensownym zakresie: {pred_payload:.2f} PLN/MWh",
      f"Predykcja poza zakresem 0–10000: {pred_payload:.2f}")

# --- Podsumowanie -----------------------------------------------------------
print(f"\n{'='*40}")
if errors:
    print(f"❌ NIEPOWODZENIE — {len(errors)} błędy:")
    for e in errors:
        print(f"   • {e}")
    sys.exit(1)
else:
    print(f"✅ PARYTET PRZESZEDŁ — model gotowy do użycia w FastAPI")
    print(f"   Predykcja dla sample_payload: {pred_payload:.2f} PLN/MWh\n")
    sys.exit(0)
