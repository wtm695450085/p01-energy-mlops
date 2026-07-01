"""Generuje losowy ruch testowy na endpoint /predict do przetestowania monitoringu."""
import json
import random
import sys
import time
from pathlib import Path
import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

BASE_URL = "http://127.0.0.1:8000"

def get_reference_ranges():
    """Zwraca przybliżone zakresy dla cech na podstawie sample_payload.json."""
    payload_path = ROOT / "scripts" / "sample_payload.json"
    if not payload_path.exists():
        # Domyślne wartości jeśli brak pliku
        return {
            "temp_pl": (-5, 25),
            "wind_pl": (5, 30),
            "radiation_pl": (0, 800),
            "price_lag_24h": (100, 600),
            "price_lag_168h": (100, 600)
        }
    
    base = json.loads(payload_path.read_text())
    return {
        "temp_pl": (base["temp_pl"] - 10, base["temp_pl"] + 10),
        "wind_pl": (max(0, base["wind_pl"] - 10), base["wind_pl"] + 10),
        "radiation_pl": (max(0, base["radiation_pl"] - 200), base["radiation_pl"] + 300),
        "price_lag_24h": (base["price_lag_24h"] * 0.7, base["price_lag_24h"] * 1.3),
        "price_lag_168h": (base["price_lag_168h"] * 0.7, base["price_lag_168h"] * 1.3)
    }

def generate_request(ranges):
    return {
        "hour": random.randint(0, 23),
        "day_of_week": random.randint(0, 6),
        "is_holiday_or_weekend": random.randint(0, 1),
        "temp_pl": round(random.uniform(*ranges["temp_pl"]), 1),
        "wind_pl": round(random.uniform(*ranges["wind_pl"]), 1),
        "radiation_pl": round(random.uniform(*ranges["radiation_pl"]), 1),
        "price_lag_24h": round(random.uniform(*ranges["price_lag_24h"]), 2),
        "price_lag_168h": round(random.uniform(*ranges["price_lag_168h"]), 2)
    }

def main(n_requests=20):
    print(f"Generowanie {n_requests} losowych zapytań do {BASE_URL}/predict ...")
    ranges = get_reference_ranges()
    
    success = 0
    errors = 0
    
    for i in range(n_requests):
        payload = generate_request(ranges)
        
        # Co 10 zapytanie niepoprawne, aby zasymulować błędy (dla logów)
        if i % 10 == 9:
            payload["hour"] = 99
            
        try:
            r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)
            if r.status_code == 200:
                success += 1
            else:
                errors += 1
        except Exception as e:
            print(f"Błąd połączenia: {e}")
            errors += 1
            
        time.sleep(0.1)
        
    print(f"Zakończono. Sukces: {success}, Błędy API: {errors}")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(n)
