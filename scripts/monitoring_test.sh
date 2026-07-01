#!/usr/bin/env bash
# monitoring_test.sh — Automatyczne testy bramki Etapu 2
# Użycie: bash scripts/monitoring_test.sh
# Wymaga: działającej aplikacji pod http://127.0.0.1:8000

set -e
BASE="http://127.0.0.1:8000"
PASS=0
FAIL=0
ERRORS=()

green() { echo -e "\033[32m✓\033[0m $1"; }
red()   { echo -e "\033[31m✗\033[0m $1"; }

check() {
  local desc="$1"; local cmd="$2"; local expect="$3"
  result=$(eval "$cmd" 2>/dev/null || true)
  if echo "$result" | grep -q "$expect"; then
    green "$desc"
    PASS=$((PASS+1))
  else
    red "$desc (oczekiwano: '$expect', dostałem: '${result:0:120}')"
    FAIL=$((FAIL+1))
    ERRORS+=("$desc")
  fi
}

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   MONITORING TEST — EnergyForecast API   ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Upewnijmy się, że ruch został wygenerowany
echo "Generowanie ruchu testowego..."
python3 scripts/generate_test_traffic.py 20

# 1. /logs/recent
echo ""
echo "--- 1. GET /logs/recent ---"
check "Endpoint logs zwraca JSON" "curl -sf $BASE/logs/recent" '"logs"'

# 2. /monitoring/summary
echo ""
echo "--- 2. GET /monitoring/summary ---"
check "Zwraca całkowitą liczbę predykcji" "curl -sf $BASE/monitoring/summary" '"total_predictions"'
check "Są zapisane predykcje (>0)" "curl -sf $BASE/monitoring/summary" '"total_predictions":[1-9]'

# 3. /metrics (Prometheus)
echo ""
echo "--- 3. GET /metrics ---"
check "Format prometheus" "curl -sf $BASE/metrics" 'mlops_predictions_total'
check "Metryka errors" "curl -sf $BASE/metrics" 'mlops_prediction_errors_total'

# 4. Raport dryftu
echo ""
echo "--- 4. Raport dryftu ---"
echo "Generowanie raportu dryftu..."
python3 src/monitoring/drift_report.py
check "Raport dostępny przez API" "curl -sf $BASE/monitoring/drift" '"status"'

# 5. Ewaluacja prognoz
echo ""
echo "--- 5. Ewaluacja prognoz ---"
echo "Uruchamianie evaluate_forecasts.py..."
python3 src/monitoring/evaluate_forecasts.py
check "Endpoint accuracy zwraca historię" "curl -sf $BASE/monitoring/accuracy" '"history"'

# Podsumowanie
echo ""
echo "════════════════════════════════════════"
echo "  WYNIK: ${PASS} passed, ${FAIL} failed"
echo "════════════════════════════════════════"
if [ ${#ERRORS[@]} -gt 0 ]; then
  echo "  Błędy:"
  for e in "${ERRORS[@]}"; do echo "    • $e"; done
  echo ""
  exit 1
fi
echo ""
exit 0
