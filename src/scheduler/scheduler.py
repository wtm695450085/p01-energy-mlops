"""
Serwis harmonogramu zadań (Scheduler).
Wykorzystuje bibliotekę `schedule` do cyklicznego uruchamiania skryptów MLOps.
"""
import schedule
import time
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import requests

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

def run_script(script_path: str):
    """Pomocnicza funkcja do uruchamiania skryptów."""
    print(f"[{datetime.now().isoformat()}] Uruchamianie {script_path}...")
    try:
        result = subprocess.run(["python3", script_path], check=True, capture_output=True, text=True)
        print(f"[{datetime.now().isoformat()}] Sukces: {script_path}\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now().isoformat()}] Błąd wykonania {script_path}:\n{e.stderr}")
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] Wyjątek podczas {script_path}: {e}")

def task_update_data():
    """Codzienne pobieranie najnowszych danych i aktualizacja bazy."""
    print(f"[{datetime.now().isoformat()}] ROZPOCZĘCIE: Aktualizacja danych")
    run_script(str(ROOT / "scripts" / "update_data.py"))
    
def task_evaluate_drift_and_accuracy():
    """Codzienna ewaluacja trafności i generacja raportu dryftu."""
    print(f"[{datetime.now().isoformat()}] ROZPOCZĘCIE: Monitorowanie jakości")
    run_script(str(ROOT / "src" / "monitoring" / "evaluate_forecasts.py"))
    run_script(str(ROOT / "src" / "monitoring" / "drift_report.py"))

def task_retrain_model():
    """Tygodniowy retrening modelu."""
    print(f"[{datetime.now().isoformat()}] ROZPOCZĘCIE: Retrening modelu")
    run_script(str(ROOT / "src" / "training" / "retrain.py"))
    
    # Powiadom API o nowym modelu
    try:
        r = requests.post("http://api:8000/admin/reload-model", timeout=10)
        print(f"[{datetime.now().isoformat()}] Przeładowanie modelu w API: {r.status_code}")
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] Nie udało się powiadomić API: {e}")

def main():
    print("Inicjalizacja Schedulera MLOps...")
    
    # Definicja harmonogramu
    # 1. Pobieranie danych codziennie rano po 9:00 (kiedy PSE zazwyczaj ma zaktualizowane dane dla RDN)
    schedule.every().day.at("09:15").do(task_update_data)
    
    # 2. Ewaluacja monitoringu po 10:00 (gdy dane są na pewno pobrane)
    schedule.every().day.at("10:00").do(task_evaluate_drift_and_accuracy)
    
    # 3. Retrening modelu raz w tygodniu, np. w niedzielę w nocy (02:00)
    schedule.every().sunday.at("02:00").do(task_retrain_model)
    
    print("Harmonogram ustawiony. Oczekuję na zadania...")
    
    # Główna pętla
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
